"""Tests for embodia's grouped-command schema and core runtime flow."""

from __future__ import annotations

import time
import unittest

import embodia as em

from helpers import DummyPolicy, DummyRobot


class InterfaceTests(unittest.TestCase):
    """Coverage for the schema, mixins, and single-step runtime helpers."""

    def test_action_roundtrip_uses_grouped_commands(self) -> None:
        action = em.coerce_action(
            {
                "arm": {
                    "kind": "cartesian_pose_delta",
                    "value": [0.1, 0.2, 0.3],
                    "ref_frame": "tool",
                },
                "gripper": {
                    "kind": "gripper_position",
                    "value": [0.8],
                },
            }
        )

        self.assertEqual(
            action.get_command("arm").kind,  # type: ignore[union-attr]
            "cartesian_pose_delta",
        )
        self.assertEqual(action.get_command("gripper").value, [0.8])  # type: ignore[union-attr]
        self.assertEqual(
            em.action_to_dict(action),
            {
                "arm": {
                    "kind": "cartesian_pose_delta",
                    "value": [0.1, 0.2, 0.3],
                    "ref_frame": "tool",
                },
                "gripper": {
                    "kind": "gripper_position",
                    "value": [0.8],
                },
            },
        )

        action_with_meta = em.coerce_action(
            {
                "commands": {
                    "arm": {
                        "kind": "cartesian_pose_delta",
                        "value": [0.1, 0.2, 0.3],
                        "ref_frame": "tool",
                    },
                    "gripper": {
                        "kind": "gripper_position",
                        "value": [0.8],
                    },
                },
                "meta": {"source": "test"},
            }
        )

        self.assertEqual(
            em.action_to_dict(action_with_meta),
            {
                "commands": {
                    "arm": {
                        "kind": "cartesian_pose_delta",
                        "value": [0.1, 0.2, 0.3],
                        "ref_frame": "tool",
                    },
                    "gripper": {
                        "kind": "gripper_position",
                        "value": [0.8],
                    },
                },
                "meta": {"source": "test"},
            },
        )

        self.assertEqual(
            em.action_to_dict(
                action_with_meta,
                compact=False,
                commands_as_mapping=False,
            ),
            {
                "commands": [
                    {
                        "target": "arm",
                        "kind": "cartesian_pose_delta",
                        "value": [0.1, 0.2, 0.3],
                        "ref_frame": "tool",
                        "meta": {},
                    },
                    {
                        "target": "gripper",
                        "kind": "gripper_position",
                        "value": [0.8],
                        "ref_frame": None,
                        "meta": {},
                    },
                ],
                "meta": {"source": "test"},
            },
        )

    def test_coerce_action_rejects_duplicate_targets_in_legacy_list(self) -> None:
        action = {
            "commands": [
                {
                    "target": "arm",
                    "kind": "cartesian_pose_delta",
                    "value": [0.0] * 6,
                },
                {
                    "target": "arm",
                    "kind": "joint_position",
                    "value": [0.0] * 6,
                },
            ]
        }

        with self.assertRaises(em.InterfaceValidationError) as ctx:
            em.coerce_action(action)

        self.assertIn("duplicate target", str(ctx.exception))

    def test_command_kind_registry_exposes_builtins(self) -> None:
        spec = em.get_command_kind_spec("joint_position")
        self.assertEqual(spec.name, "joint_position")
        self.assertTrue(em.is_known_command_kind("joint_position"))

    def test_register_command_kind_rejects_duplicate_name(self) -> None:
        spec = em.CommandKindSpec(
            name="custom:test_duplicate_registration",
            description="test kind",
            default_dim=2,
            allowed_component_kinds=["custom"],
        )
        em.register_command_kind(spec)

        with self.assertRaises(ValueError):
            em.register_command_kind(
                em.CommandKindSpec(name="custom:test_duplicate_registration")
            )

    def test_validate_command_accepts_unregistered_custom_kind(self) -> None:
        em.validate_command(
            em.Command(
                kind="custom:my_lab_synergy",
                value=[0.1, 0.2],
            )
        )

    def test_validate_command_rejects_unknown_non_custom_kind(self) -> None:
        with self.assertRaises(em.InterfaceValidationError):
            em.validate_command(
                em.Command(
                    kind="definitely_unknown_kind",
                    value=[0.0],
                )
            )

    def test_dummy_components_pass_pair_check(self) -> None:
        robot = DummyRobot()
        policy = DummyPolicy()

        sample_frame = robot.reset()
        em.check_robot(robot)
        em.check_policy(policy, sample_frame=sample_frame)
        em.check_pair(robot, policy, sample_frame=sample_frame)

    def test_check_pair_reports_group_mismatch(self) -> None:
        robot = DummyRobot()

        class IncompatiblePolicy(em.PolicyMixin):
            def _get_spec_impl(self) -> dict[str, object]:
                return {
                    "name": "bad_model",
                    "required_image_keys": ["front_rgb"],
                    "required_state_keys": ["joint_positions"],
                    "required_task_keys": [],
                    "outputs": [
                        {
                            "target": "gripper",
                            "command_kind": "gripper_position",
                            "dim": 1,
                        }
                    ],
                }

            def _reset_impl(self) -> None:
                return None

            def _step_impl(self, frame: em.Frame) -> em.Action:
                del frame
                return em.Action.single(
                    target="gripper",
                    kind="gripper_position",
                    value=[1.0],
                )

        policy = IncompatiblePolicy()
        with self.assertRaises(em.InterfaceValidationError) as ctx:
            em.check_pair(robot, policy, sample_frame=robot.reset())

        self.assertIn("missing required component", str(ctx.exception))

    def test_mixin_must_be_leftmost_direct_base(self) -> None:
        class VendorRobot:
            pass

        with self.assertRaises(TypeError):

            class WrongOrder(VendorRobot, em.RobotMixin):
                pass

    def test_robot_and_policy_mixins_remap_targets_modes_and_state(self) -> None:
        class VendorRobot:
            def __init__(self) -> None:
                self.last_action: em.Action | None = None

            def capture(self) -> dict[str, object]:
                return {
                    "timestamp_ns": time.time_ns(),
                    "images": {"front_rgb": None},
                    "state": {"qpos": [1.0] * 6, "gripper_pos": 0.5},
                    "task": {"prompt": "fold"},
                }

            def send_command(self, action: em.Action) -> None:
                self.last_action = action

            def home(self) -> dict[str, object]:
                return self.capture()

        class YourRobot(em.RobotMixin, VendorRobot):
            ROBOT_SPEC = {
                "name": "vendor_robot",
                "image_keys": ["front_rgb"],
                "components": [
                    {
                        "name": "arm",
                        "kind": "arm",
                        "dof": 6,
                        "supported_command_kinds": ["cartesian_pose_delta"],
                        "state_keys": ["joint_positions"],
                    },
                    {
                        "name": "gripper",
                        "kind": "gripper",
                        "dof": 1,
                        "supported_command_kinds": ["gripper_position"],
                        "state_keys": ["position"],
                    },
                ],
            }
            METHOD_ALIASES = {
                "observe": "capture",
                "act": "send_command",
                "reset": "home",
            }
            MODALITY_MAPS = {
                em.CONTROL_TARGETS: {
                    "vendor_arm": "arm",
                    "vendor_gripper": "gripper",
                },
                em.STATE_KEYS: {
                    "qpos": "joint_positions",
                    "gripper_pos": "position",
                },
                em.COMMAND_KINDS: {
                    "cartesian_delta": "cartesian_pose_delta",
                    "gripper_position": "gripper_position",
                },
            }

        class VendorPolicy:
            def clear_state(self) -> None:
                return None

            def infer(self, frame: em.Frame) -> dict[str, object]:
                self.seen = frame
                return {
                    "vendor_arm": {
                        "kind": "cartesian_delta",
                        "value": [0.1] * 6,
                    },
                    "vendor_gripper": {
                        "kind": "gripper_position",
                        "value": [0.2],
                    },
                }

        class YourPolicy(em.PolicyMixin, VendorPolicy):
            POLICY_SPEC = {
                "name": "vendor_model",
                "required_image_keys": ["front_rgb"],
                "required_state_keys": ["joint_positions", "position"],
                "required_task_keys": ["prompt"],
                "outputs": [
                    {
                        "target": "arm",
                        "command_kind": "cartesian_pose_delta",
                        "dim": 6,
                    },
                    {
                        "target": "gripper",
                        "command_kind": "gripper_position",
                        "dim": 1,
                    },
                ],
            }
            METHOD_ALIASES = {
                "reset": "clear_state",
            }
            MODALITY_MAPS = {
                em.CONTROL_TARGETS: {
                    "vendor_arm": "arm",
                    "vendor_gripper": "gripper",
                },
                em.STATE_KEYS: {
                    "qpos": "joint_positions",
                    "gripper_pos": "position",
                },
                em.COMMAND_KINDS: {
                    "cartesian_delta": "cartesian_pose_delta",
                    "gripper_position": "gripper_position",
                },
            }

        robot = YourRobot()
        policy = YourPolicy()
        result = em.run_step(robot, policy)

        self.assertEqual(
            result.action.get_command("arm").kind,  # type: ignore[union-attr]
            "cartesian_pose_delta",
        )
        self.assertEqual(result.action.get_command("gripper").value, [0.2])  # type: ignore[union-attr]
        self.assertEqual(
            robot.last_action.get_command("vendor_arm").kind,  # type: ignore[union-attr]
            "cartesian_delta",
        )
        self.assertEqual(policy.seen.state["qpos"], [1.0] * 6)
        self.assertEqual(policy.seen.task["prompt"], "fold")

    def test_run_step_accepts_action_function(self) -> None:
        robot = DummyRobot()

        def scripted(frame: em.Frame) -> em.Action:
            del frame
            return em.Action.single(
                target="arm",
                kind="cartesian_pose_delta",
                value=[1.0] * 6,
            )

        result = em.run_step(robot, action_fn=scripted)
        self.assertEqual(result.action.get_command("arm").value, [1.0] * 6)  # type: ignore[union-attr]
        self.assertEqual(robot.last_action.get_command("arm").value, [1.0] * 6)  # type: ignore[union-attr]

    def test_run_step_prefers_robot_returned_action(self) -> None:
        class ReturningRobot(em.RobotMixin):
            def __init__(self) -> None:
                self.last_action: em.Action | None = None

            def _get_spec_impl(self) -> dict[str, object]:
                return {
                    "name": "returning_robot",
                    "image_keys": ["front_rgb"],
                    "components": [
                        {
                            "name": "arm",
                            "kind": "arm",
                            "dof": 6,
                            "supported_command_kinds": ["cartesian_pose_delta"],
                            "state_keys": ["joint_positions"],
                        }
                    ],
                }

            def _observe_impl(self) -> dict[str, object]:
                return {
                    "timestamp_ns": time.time_ns(),
                    "images": {"front_rgb": None},
                    "state": {"joint_positions": [0.0] * 6},
                }

            def _act_impl(self, action: em.Action) -> em.Action:
                del action
                self.last_action = em.Action.single(
                    target="arm",
                    kind="cartesian_pose_delta",
                    value=[0.25] * 6,
                )
                return self.last_action

            def _reset_impl(self) -> dict[str, object]:
                return self._observe_impl()

        robot = ReturningRobot()

        def scripted(frame: em.Frame) -> em.Action:
            del frame
            return em.Action.single(
                target="arm",
                kind="cartesian_pose_delta",
                value=[1.0] * 6,
            )

        result = em.run_step(robot, action_fn=scripted)
        self.assertEqual(result.action.get_command("arm").value, [0.25] * 6)  # type: ignore[union-attr]
        self.assertEqual(robot.last_action.get_command("arm").value, [0.25] * 6)  # type: ignore[union-attr]

    def test_run_step_result_can_be_exported(self) -> None:
        robot = DummyRobot()
        policy = DummyPolicy()
        result = em.run_step(robot, policy)

        self.assertEqual(em.frame_to_dict(result.frame)["state"]["joint_positions"], [0.0] * 6)
        self.assertEqual(
            em.action_to_dict(result.action)["arm"]["kind"],
            "cartesian_pose_delta",
        )

    def test_run_step_can_report_embodia_timing(self) -> None:
        robot = DummyRobot()
        policy = DummyPolicy()

        result = em.run_step(robot, policy, measure_timing=True)

        self.assertIsNotNone(result.timing)
        assert result.timing is not None
        self.assertGreaterEqual(result.timing.total_s, 0.0)
        self.assertGreaterEqual(result.timing.embodia_overhead_s, 0.0)
        self.assertGreaterEqual(result.timing.source_call_s, 0.0)
        self.assertGreaterEqual(result.timing.observe_call_s, 0.0)
        self.assertGreaterEqual(
            result.timing.total_s,
            result.timing.embodia_overhead_s,
        )


if __name__ == "__main__":
    unittest.main()
