"""Tests for embodia's optional OpenPI-compatible remote helpers."""

from __future__ import annotations

import unittest
from unittest import mock

import embodia as em
from embodia.contrib import openpi_remote as em_openpi_remote


class OpenPIRemoteTests(unittest.TestCase):
    """Coverage for lightweight OpenPI conversion and remote-policy helpers."""

    def test_openpi_actions_to_action_plan_uses_explicit_target(self) -> None:
        plan = em_openpi_remote.openpi_actions_to_action_plan(
            {"actions": [[1, 2, 3], [4, 5, 6]]},
            target="arm",
            kind="joint_position",
            ref_frame="tool",
        )

        first = plan[0].get_command("arm")
        assert first is not None
        self.assertEqual(first.kind, "joint_position")
        self.assertEqual(first.value, [1.0, 2.0, 3.0])
        self.assertEqual(first.ref_frame, "tool")

    def test_openpi_response_from_action_plan_preserves_metadata(self) -> None:
        response = em_openpi_remote.openpi_response_from_action_plan(
            [
                em.Action.single(
                    target="arm",
                    kind="joint_position",
                    value=[0.1, 0.2, 0.3],
                    ref_frame="tool",
                ),
                em.Action.single(
                    target="arm",
                    kind="joint_position",
                    value=[0.4, 0.5, 0.6],
                    ref_frame="tool",
                ),
            ]
        )

        self.assertEqual(response["actions"][0], [0.1, 0.2, 0.3])
        self.assertEqual(response["embodia"]["action_target"], "arm")
        self.assertEqual(response["embodia"]["action_kind"], "joint_position")

    def test_openpi_transform_converts_obs_and_actions(self) -> None:
        transform = em_openpi_remote.OpenPITransform(
            command_kind="joint_position",
            action_target="arm",
        )

        frame = transform.build_frame(
            {
                "timestamp_ns": 1,
                "images": {},
                "state": {"joint_positions": [0.0, 1.0, 2.0]},
            }
        )
        action = transform.first_action_from_response({"actions": [[1, 2, 3]]})
        command = action.get_command("arm")
        assert command is not None

        self.assertEqual(frame.state["joint_positions"], [0.0, 1.0, 2.0])
        self.assertEqual(command.kind, "joint_position")
        self.assertEqual(command.value, [1.0, 2.0, 3.0])

    def test_build_policy_adapter_serves_grouped_action_policy(self) -> None:
        class DemoPolicy(em.PolicyMixin):
            def _get_spec_impl(self) -> dict[str, object]:
                return {
                    "name": "demo_model",
                    "required_image_keys": [],
                    "required_state_keys": ["joint_positions"],
                    "required_task_keys": [],
                    "outputs": [
                        {
                            "target": "arm",
                            "command_kind": "joint_position",
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
                    kind="joint_position",
                    value=[1.0, 2.0, 3.0],
                )

        adapter = em_openpi_remote.build_policy_adapter(DemoPolicy())
        response = adapter.infer(
            {
                "timestamp_ns": 1,
                "images": {},
                "state": {"joint_positions": [0.0, 0.0, 0.0]},
            }
        )

        self.assertEqual(response["actions"][0], [1.0, 2.0, 3.0])
        self.assertIn("policy_spec", adapter.get_server_metadata()["embodia"])

    def test_robot_mixin_can_use_hidden_remote_policy_backend(self) -> None:
        class DemoRobot(em.RobotMixin):
            ROBOT_SPEC = {
                "name": "demo_robot",
                "image_keys": [],
                "components": [
                    {
                        "name": "arm",
                        "kind": "arm",
                        "dof": 3,
                        "supported_command_kinds": ["joint_position"],
                        "state_keys": ["joint_positions"],
                    }
                ],
            }

            def __init__(self) -> None:
                self.last_action: em.Action | None = None

            def _observe_impl(self) -> dict[str, object]:
                return {
                    "timestamp_ns": 1,
                    "images": {},
                    "state": {"joint_positions": [0.0, 0.0, 0.0]},
                }

            def _act_impl(self, action: em.Action) -> None:
                self.last_action = action

            def _reset_impl(self) -> dict[str, object]:
                return self._observe_impl()

        class StubRunner:
            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                del obs
                return {"actions": [[0.4, 0.5, 0.6]]}

        robot = DemoRobot()
        with mock.patch.object(em_openpi_remote, "RemotePolicyRunner", return_value=StubRunner()):
            em_openpi_remote.configure_robot_remote_policy(
                robot,
                obs_builder=em.frame_to_dict,
                action_target="arm",
                command_kind="joint_position",
            )

        action = robot.request_remote_policy_action(robot.reset())
        result = em.run_step(robot)
        command = action.get_command("arm")
        assert command is not None

        self.assertEqual(command.value, [0.4, 0.5, 0.6])
        self.assertEqual(result.action.get_command("arm").value, [0.4, 0.5, 0.6])  # type: ignore[union-attr]
        self.assertEqual(robot.last_action.get_command("arm").value, [0.4, 0.5, 0.6])  # type: ignore[union-attr]

    def test_remote_policy_runner_rejects_when_disabled(self) -> None:
        runner = em_openpi_remote.RemotePolicyRunner(enabled=False)

        with self.assertRaises(em.InterfaceValidationError):
            runner.infer({"timestamp_ns": 1, "images": {}, "state": {}})


if __name__ == "__main__":
    unittest.main()
