"""Tests for embodia's grouped-command schema and core runtime flow."""

from __future__ import annotations

import time
import unittest

import embodia as em

from .helpers import DummyModel, DummyRobot


class InterfaceTests(unittest.TestCase):
    """Coverage for the schema, mixins, and single-step runtime helpers."""

    def test_action_roundtrip_uses_grouped_commands(self) -> None:
        action = em.coerce_action(
            {
                "commands": [
                    {
                        "target": "arm",
                        "mode": "ee_delta",
                        "value": [0.1, 0.2, 0.3],
                        "ref_frame": "tool",
                    },
                    {
                        "target": "gripper",
                        "mode": "scalar_position",
                        "value": [0.8],
                    },
                ],
                "dt": 0.05,
                "meta": {"source": "test"},
            }
        )

        self.assertEqual(action.get_command("arm").mode, "ee_delta")  # type: ignore[union-attr]
        self.assertEqual(action.get_command("gripper").value, [0.8])  # type: ignore[union-attr]
        self.assertEqual(
            em.action_to_dict(action),
            {
                "commands": [
                    {
                        "target": "arm",
                        "mode": "ee_delta",
                        "value": [0.1, 0.2, 0.3],
                        "ref_frame": "tool",
                        "meta": {},
                    },
                    {
                        "target": "gripper",
                        "mode": "scalar_position",
                        "value": [0.8],
                        "ref_frame": None,
                        "meta": {},
                    },
                ],
                "dt": 0.05,
                "meta": {"source": "test"},
            },
        )

    def test_validate_action_rejects_duplicate_targets(self) -> None:
        action = em.Action(
            commands=[
                em.Command(target="arm", mode="ee_delta", value=[0.0] * 6),
                em.Command(target="arm", mode="joint_position", value=[0.0] * 6),
            ]
        )

        with self.assertRaises(em.InterfaceValidationError) as ctx:
            em.validate_action(action)

        self.assertIn("duplicate target", str(ctx.exception))

    def test_dummy_components_pass_pair_check(self) -> None:
        robot = DummyRobot()
        model = DummyModel()

        sample_frame = robot.reset()
        em.check_robot(robot)
        em.check_model(model, sample_frame=sample_frame)
        em.check_pair(robot, model, sample_frame=sample_frame)

    def test_check_pair_reports_group_mismatch(self) -> None:
        robot = DummyRobot()

        class IncompatibleModel(em.ModelMixin):
            def _get_spec_impl(self) -> dict[str, object]:
                return {
                    "name": "bad_model",
                    "required_image_keys": ["front_rgb"],
                    "required_state_keys": ["joint_positions"],
                    "required_task_keys": [],
                    "outputs": [
                        {
                            "target": "gripper",
                            "mode": "scalar_position",
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
                    mode="scalar_position",
                    value=[1.0],
                )

        model = IncompatibleModel()
        with self.assertRaises(em.InterfaceValidationError) as ctx:
            em.check_pair(robot, model, sample_frame=robot.reset())

        self.assertIn("missing required control group", str(ctx.exception))

    def test_mixin_must_be_leftmost_direct_base(self) -> None:
        class VendorRobot:
            pass

        with self.assertRaises(TypeError):

            class WrongOrder(VendorRobot, em.RobotMixin):
                pass

    def test_robot_and_model_mixins_remap_targets_modes_and_state(self) -> None:
        class VendorRobot:
            ROBOT_SPEC = {
                "name": "vendor_robot",
                "image_keys": ["front_rgb"],
                "groups": [
                    {
                        "name": "arm",
                        "kind": "arm",
                        "dof": 6,
                        "action_modes": ["ee_delta"],
                        "state_keys": ["joint_positions"],
                    },
                    {
                        "name": "gripper",
                        "kind": "gripper",
                        "dof": 1,
                        "action_modes": ["scalar_position"],
                        "state_keys": ["position"],
                    },
                ],
                "task_keys": ["prompt"],
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
                em.ACTION_MODES: {
                    "cartesian_delta": "ee_delta",
                    "gripper_position": "scalar_position",
                },
            }

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
            pass

        class VendorModel:
            MODEL_SPEC = {
                "name": "vendor_model",
                "required_image_keys": ["front_rgb"],
                "required_state_keys": ["joint_positions", "position"],
                "required_task_keys": ["prompt"],
                "outputs": [
                    {"target": "arm", "mode": "ee_delta", "dim": 6},
                    {"target": "gripper", "mode": "scalar_position", "dim": 1},
                ],
            }
            METHOD_ALIASES = {
                "reset": "clear_state",
                "step": "infer",
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
                em.ACTION_MODES: {
                    "cartesian_delta": "ee_delta",
                    "gripper_position": "scalar_position",
                },
            }

            def clear_state(self) -> None:
                return None

            def infer(self, frame: em.Frame) -> dict[str, object]:
                self.seen = frame
                return {
                    "commands": [
                        {
                            "target": "vendor_arm",
                            "mode": "cartesian_delta",
                            "value": [0.1] * 6,
                        },
                        {
                            "target": "vendor_gripper",
                            "mode": "gripper_position",
                            "value": [0.2],
                        },
                    ],
                    "dt": 0.05,
                }

        class YourModel(em.ModelMixin, VendorModel):
            pass

        robot = YourRobot()
        model = YourModel()
        result = em.run_step(robot, model)

        self.assertEqual(result.action.get_command("arm").mode, "ee_delta")  # type: ignore[union-attr]
        self.assertEqual(result.action.get_command("gripper").value, [0.2])  # type: ignore[union-attr]
        self.assertEqual(robot.last_action.get_command("vendor_arm").mode, "cartesian_delta")  # type: ignore[union-attr]
        self.assertEqual(model.seen.state["qpos"], [1.0] * 6)
        self.assertEqual(model.seen.task["prompt"], "fold")

    def test_run_step_accepts_action_function(self) -> None:
        robot = DummyRobot()

        def scripted(frame: em.Frame) -> em.Action:
            del frame
            return em.Action.single(
                target="arm",
                mode="ee_delta",
                value=[1.0] * 6,
            )

        result = em.run_step(robot, action_fn=scripted)
        self.assertEqual(result.action.get_command("arm").value, [1.0] * 6)  # type: ignore[union-attr]
        self.assertEqual(robot.last_action.get_command("arm").value, [1.0] * 6)  # type: ignore[union-attr]

    def test_run_step_result_can_be_exported(self) -> None:
        robot = DummyRobot()
        model = DummyModel()
        result = em.run_step(robot, model)

        self.assertEqual(em.frame_to_dict(result.frame)["state"]["joint_positions"], [0.0] * 6)
        self.assertEqual(
            em.action_to_dict(result.action)["commands"][0]["target"],
            "arm",
        )


if __name__ == "__main__":
    unittest.main()
