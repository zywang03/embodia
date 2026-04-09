"""Tests for inferaxis's optional remote helpers."""

from __future__ import annotations

import unittest

import inferaxis as em
from inferaxis.contrib import remote as em_remote

from helpers import assert_array_equal


class RemoteTests(unittest.TestCase):
    """Coverage for lightweight remote conversion and remote-policy helpers."""

    def test_actions_to_action_plan_uses_explicit_target(self) -> None:
        plan = em_remote.actions_to_action_plan(
            {"actions": [[1, 2, 3], [4, 5, 6]]},
            target="arm",
            command="joint_position",
            ref_frame="tool",
        )

        first = plan[0].get_command("arm")
        assert first is not None
        self.assertEqual(first.command, "joint_position")
        assert_array_equal(self, first.value, [1.0, 2.0, 3.0])
        self.assertEqual(first.ref_frame, "tool")

    def test_response_from_action_plan_preserves_metadata(self) -> None:
        response = em_remote.response_from_action_plan(
            [
                em.Action.single(
                    target="arm",
                    command="joint_position",
                    value=[0.1, 0.2, 0.3],
                    ref_frame="tool",
                ),
                em.Action.single(
                    target="arm",
                    command="joint_position",
                    value=[0.4, 0.5, 0.6],
                    ref_frame="tool",
                ),
            ]
        )

        self.assertEqual(response["actions"][0], [0.1, 0.2, 0.3])
        self.assertEqual(response["inferaxis"]["action_target"], "arm")
        self.assertEqual(response["inferaxis"]["action_command"], "joint_position")

    def test_remote_transform_converts_obs_and_actions(self) -> None:
        transform = em_remote.RemoteTransform(
            command="joint_position",
            action_target="arm",
        )

        frame = transform.build_frame(
            {
                "timestamp_ns": 1,
                "images": {},
                "state": {"arm": [0.0, 1.0, 2.0]},
            }
        )
        action = transform.first_action_from_response({"actions": [[1, 2, 3]]})
        command = action.get_command("arm")
        assert command is not None

        assert_array_equal(self, frame.state["arm"], [0.0, 1.0, 2.0])
        self.assertEqual(command.command, "joint_position")
        assert_array_equal(self, command.value, [1.0, 2.0, 3.0])

    def test_build_policy_adapter_serves_grouped_action_policy(self) -> None:
        class DemoPolicy(em.PolicyMixin):
            def _get_spec_impl(self) -> dict[str, object]:
                return {
                    "name": "demo_model",
                    "required_image_keys": [],
                    "required_state_keys": ["arm"],
                    "required_task_keys": [],
                    "outputs": [
                        {
                            "target": "arm",
                            "command": "joint_position",
                            "dim": 3,
                        }
                    ],
                }

            def _reset_impl(self) -> None:
                return None

            def _step_impl(self, frame: em.Frame) -> em.Action:
                del frame
                return em.Action.single(
                    target="arm",
                    command="joint_position",
                    value=[1.0, 2.0, 3.0],
                )

        adapter = em_remote.build_remote_policy_adapter(DemoPolicy())
        response = adapter.infer(
            {
                "timestamp_ns": 1,
                "images": {},
                "state": {"arm": [0.0, 0.0, 0.0]},
            }
        )

        self.assertEqual(response["actions"][0], [1.0, 2.0, 3.0])
        self.assertIn("policy_spec", adapter.get_server_metadata()["inferaxis"])

    def test_remote_policy_source_can_drive_run_step(self) -> None:
        class DemoRobot(em.RobotMixin):
            ROBOT_SPEC = {
                "name": "demo_robot",
                "image_keys": [],
                "components": [
                    {
                        "name": "arm",
                        "type": "arm",
                        "dof": 3,
                        "command": ["joint_position"],
                    }
                ],
            }

            def __init__(self) -> None:
                self.last_action: em.Action | None = None

            def _observe_impl(self) -> dict[str, object]:
                return {
                    "timestamp_ns": 1,
                    "images": {},
                    "state": {"arm": [0.0, 0.0, 0.0]},
                }

            def _act_impl(self, action: em.Action) -> None:
                self.last_action = action

            def _reset_impl(self) -> dict[str, object]:
                return self._observe_impl()

        class StubRunner:
            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                del obs
                return {
                    "actions": [[0.4, 0.5, 0.6]],
                    "inferaxis": {
                        "action_target": "arm",
                        "action_command": "joint_position",
                    },
                }

        robot = DemoRobot()
        policy = em_remote.RemotePolicy(
            runner=StubRunner(),
        )
        result = em.run_step(robot, source=policy)
        action = result.action
        command = action.get_command("arm")
        assert command is not None

        assert_array_equal(self, command.value, [0.4, 0.5, 0.6])
        assert_array_equal(self, robot.last_action.get_command("arm").value, [0.4, 0.5, 0.6])  # type: ignore[union-attr]

    def test_remote_policy_can_infer_action_shape_from_policy_spec(self) -> None:
        class StubRunner:
            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                del obs
                return {"actions": [[0.7, 0.8, 0.9]]}

            def get_server_metadata(self) -> dict[str, object]:
                return {
                    "inferaxis": {
                        "policy_spec": {
                            "name": "remote_policy",
                            "required_image_keys": [],
                            "required_state_keys": ["arm"],
                            "required_task_keys": [],
                            "outputs": [
                                {
                                    "target": "arm",
                                    "command": "joint_position",
                                    "dim": 3,
                                }
                            ],
                        }
                    }
                }

        policy = em_remote.RemotePolicy(runner=StubRunner())
        action = policy.infer(
            {
                "timestamp_ns": 1,
                "images": {},
                "state": {"arm": [0.0, 0.0, 0.0]},
            }
        )
        command = action.get_command("arm")
        assert command is not None
        self.assertEqual(command.command, "joint_position")
        assert_array_equal(self, command.value, [0.7, 0.8, 0.9])

    def test_remote_policy_runner_rejects_when_disabled(self) -> None:
        runner = em_remote.RemotePolicyRunner(enabled=False)

        with self.assertRaises(em.InterfaceValidationError):
            runner.infer({"timestamp_ns": 1, "images": {}, "state": {}})


if __name__ == "__main__":
    unittest.main()
