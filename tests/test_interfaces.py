"""Basic tests for the embodia package."""

from __future__ import annotations

import unittest
from unittest import mock

from embodia import (
    Action,
    ACTION_MODES,
    Frame,
    IMAGE_KEYS,
    InterfaceValidationError,
    MethodAliasKey,
    ModelMixin,
    ModelSpecKey,
    RobotMixin,
    RobotSpecKey,
    STATE_KEYS,
    StepResult,
    action_to_dict,
    check_model,
    check_pair,
    check_robot,
    coerce_action,
    frame_to_dict,
    remap_frame,
    run_step,
    validate_action,
    validate_frame,
)
from embodia.core import arraylike as em_arraylike
from tests.helpers import DummyModel, DummyRobot


class InterfaceTests(unittest.TestCase):
    """Smoke tests for the minimal runtime interface package."""

    def test_dummy_components_pass_checks(self) -> None:
        robot = DummyRobot()
        model = DummyModel()
        frame = robot.observe()

        check_robot(robot)
        check_model(model, sample_frame=frame)
        check_pair(robot, model)

    def test_validate_frame_rejects_negative_timestamp(self) -> None:
        frame = Frame(timestamp_ns=-1, images={}, state={})

        with self.assertRaises(InterfaceValidationError):
            validate_frame(frame)

    def test_validate_action_rejects_bad_mode(self) -> None:
        action = Action(mode="ee_delta", value=[0.0])
        action.mode = "not_real_mode"  # type: ignore[assignment]

        with self.assertRaises(InterfaceValidationError):
            validate_action(action)

    def test_check_pair_reports_missing_keys(self) -> None:
        class IncompatibleModel(DummyModel):
            def get_spec(self):  # type: ignore[override]
                spec = super().get_spec()
                spec.required_state_keys.append("ee_pose")
                return spec

        robot = DummyRobot()
        model = IncompatibleModel()

        with self.assertRaises(InterfaceValidationError) as ctx:
            check_pair(robot, model)

        self.assertIn("missing state keys", str(ctx.exception))

    def test_transform_helpers_normalize_mapping(self) -> None:
        action = coerce_action(
            {
                "mode": "ee_delta",
                "value": (0.0, 1.0),
                "frame": "tool",
            }
        )

        self.assertIsInstance(action, Action)
        self.assertEqual(action.value, [0.0, 1.0])
        self.assertEqual(action.ref_frame, "tool")

    def test_coerce_action_accepts_optional_numpy_like_value(self) -> None:
        class FakeNdArray:
            def __init__(self, values: list[float]) -> None:
                self._values = values

            def tolist(self) -> list[float]:
                return list(self._values)

        with mock.patch.object(
            em_arraylike,
            "numpy_ndarray_type",
            return_value=FakeNdArray,
        ):
            action = coerce_action(
                {
                    "mode": "ee_delta",
                    "value": FakeNdArray([0.0, 1.0, 2.0]),
                }
            )

        self.assertEqual(action.value, [0.0, 1.0, 2.0])

    def test_coerce_action_accepts_optional_torch_like_value(self) -> None:
        class FakeDevice:
            type = "cuda"

        class FakeTensor:
            def __init__(self, values: list[float]) -> None:
                self._values = values
                self.device = FakeDevice()
                self.detached = False
                self.moved_to_cpu = False

            def detach(self) -> "FakeTensor":
                self.detached = True
                return self

            def cpu(self) -> "FakeTensor":
                self.moved_to_cpu = True
                self.device = type("CpuDevice", (), {"type": "cpu"})()
                return self

            def tolist(self) -> list[float]:
                return list(self._values)

        tensor = FakeTensor([0.0, 1.0, 2.0])
        with mock.patch.object(
            em_arraylike,
            "torch_tensor_type",
            return_value=FakeTensor,
        ):
            action = coerce_action(
                {
                    "mode": "ee_delta",
                    "value": tensor,
                }
            )

        self.assertEqual(action.value, [0.0, 1.0, 2.0])
        self.assertTrue(tensor.detached)
        self.assertTrue(tensor.moved_to_cpu)

    def test_action_to_dict_exports_ref_frame_name(self) -> None:
        exported = action_to_dict(
            Action(
                mode="ee_delta",
                value=[0.0] * 6,
                ref_frame="tool",
            )
        )

        self.assertIn("ref_frame", exported)
        self.assertEqual(exported["ref_frame"], "tool")
        self.assertNotIn("frame", exported)

    def test_frame_to_dict_exports_dataclass(self) -> None:
        frame = Frame(timestamp_ns=1, images={"front_rgb": None}, state={})

        exported = frame_to_dict(frame)

        self.assertEqual(exported["timestamp_ns"], 1)
        self.assertIn("front_rgb", exported["images"])

    def test_remap_frame_renames_vendor_keys(self) -> None:
        frame = remap_frame(
            {
                "timestamp_ns": 1,
                "images": {"rgb_front": None},
                "state": {"qpos": [0.0] * 6},
            },
            image_key_map={"rgb_front": "front_rgb"},
            state_key_map={"qpos": "joint_positions"},
        )

        self.assertIn("front_rgb", frame.images)
        self.assertIn("joint_positions", frame.state)

    def test_run_step_result_can_be_exported_with_plain_dicts(self) -> None:
        robot = DummyRobot()

        def scripted_action(frame: Frame) -> dict[str, object]:
            return {"mode": "ee_delta", "value": [frame.timestamp_ns * 0.0] * 6}

        result = run_step(robot, action_fn=scripted_action)
        exported = {
            "frame": frame_to_dict(result.frame),
            "action": action_to_dict(result.action),
            "meta": {"collector": "custom"},
        }

        self.assertEqual(exported["meta"]["collector"], "custom")
        self.assertIn("front_rgb", exported["frame"]["images"])
        self.assertEqual(exported["action"]["mode"], "ee_delta")

    def test_robot_mixin_wraps_existing_robot_class(self) -> None:
        class VendorRobot:
            def __init__(self) -> None:
                self.last_action = None

            def get_spec(self):
                return {
                    "name": "vendor_robot",
                    "action_modes": ["cartesian_delta"],
                    "image_keys": ["rgb_front"],
                    "state_keys": ["qpos"],
                }

            def observe(self):
                return {
                    "timestamp_ns": 1,
                    "images": {"rgb_front": None},
                    "state": {"qpos": [0.0] * 6},
                }

            def act(self, action):
                self.last_action = action

            def reset(self):
                return self.observe()

        class CompatibleRobot(RobotMixin, VendorRobot):
            def get_image_key_map(self):
                return {"rgb_front": "front_rgb"}

            def get_state_key_map(self):
                return {"qpos": "joint_positions"}

            def get_action_mode_map(self):
                return {"cartesian_delta": "ee_delta"}

        robot = CompatibleRobot()
        check_robot(robot)
        robot.act({"mode": "ee_delta", "value": [0.0] * 6})

        self.assertIsInstance(robot.last_action, Action)
        self.assertEqual(robot.last_action.mode, "cartesian_delta")
        self.assertIn("front_rgb", robot.observe().images)

    def test_mixin_supports_method_aliases_and_class_attribute_maps(self) -> None:
        class VendorRobot:
            def __init__(self) -> None:
                self.last_action = None

            def describe(self):
                return {
                    "name": "vendor_robot",
                    "action_modes": ["cartesian_delta"],
                    "image_keys": ["rgb_front"],
                    "state_keys": ["qpos"],
                }

            def capture(self):
                return {
                    "timestamp_ns": 1,
                    "images": {"rgb_front": None},
                    "state": {"qpos": [0.0] * 6},
                }

            def send_command(self, action):
                self.last_action = action

            def home(self):
                return self.capture()

        class VendorModel:
            def describe(self):
                return {
                    "name": "vendor_model",
                    "required_image_keys": ["rgb_front"],
                    "required_state_keys": ["qpos"],
                    "output_action_mode": "cartesian_delta",
                }

            def clear_state(self):
                return None

            def infer(self, frame):
                self.seen_frame = frame
                return {"mode": "cartesian_delta", "value": [0.0] * 6}

        class CompatibleRobot(RobotMixin, VendorRobot):
            GET_SPEC_METHOD = "describe"
            OBSERVE_METHOD = "capture"
            ACT_METHOD = "send_command"
            RESET_METHOD = "home"
            MODALITY_MAPS = {
                "images": {"rgb_front": "front_rgb"},
                "state": {"qpos": "joint_positions"},
                "action_modes": {"cartesian_delta": "ee_delta"},
            }

        class CompatibleModel(ModelMixin, VendorModel):
            GET_SPEC_METHOD = "describe"
            RESET_METHOD = "clear_state"
            STEP_METHOD = "infer"
            MODALITY_MAPS = {
                "images": {"rgb_front": "front_rgb"},
                "state": {"qpos": "joint_positions"},
                "action_modes": {"cartesian_delta": "ee_delta"},
            }

        robot = CompatibleRobot()
        model = CompatibleModel()
        result = run_step(robot, model)

        self.assertEqual(result.action.mode, "ee_delta")
        self.assertEqual(robot.last_action.mode, "cartesian_delta")
        self.assertIn("rgb_front", model.seen_frame.images)

    def test_model_mixin_wraps_existing_model_class(self) -> None:
        class VendorModel:
            def get_spec(self):
                return {
                    "name": "vendor_model",
                    "required_image_keys": ["rgb_front"],
                    "required_state_keys": ["qpos"],
                    "output_action_mode": "cartesian_delta",
                }

            def reset(self):
                return None

            def step(self, frame):
                self.seen_frame = frame
                return {"mode": "cartesian_delta", "value": [0.0] * 6}

        class CompatibleModel(ModelMixin, VendorModel):
            def get_image_key_map(self):
                return {"rgb_front": "front_rgb"}

            def get_state_key_map(self):
                return {"qpos": "joint_positions"}

            def get_action_mode_map(self):
                return {"cartesian_delta": "ee_delta"}

        model = CompatibleModel()
        frame = Frame(
            timestamp_ns=1,
            images={"front_rgb": None},
            state={"joint_positions": [0.0] * 6},
        )
        check_model(model, sample_frame=frame)
        action = model.step(frame)

        self.assertIsInstance(action, Action)
        self.assertIn("rgb_front", model.seen_frame.images)
        self.assertEqual(action.mode, "ee_delta")

    def test_mixin_supports_declarative_specs_and_method_alias_table(self) -> None:
        class VendorRobot:
            def __init__(self) -> None:
                self.last_action = None

            def capture(self):
                return {
                    "timestamp_ns": 1,
                    "images": {"rgb_front": None},
                    "state": {"qpos": [0.0] * 6},
                }

            def send_command(self, action):
                self.last_action = action

            def home(self):
                return self.capture()

        class VendorModel:
            def clear_state(self):
                return None

            def infer(self, frame):
                self.seen_frame = frame
                return {"mode": "cartesian_delta", "value": [0.0] * 6}

        class CompatibleRobot(RobotMixin, VendorRobot):
            ROBOT_SPEC = {
                RobotSpecKey.NAME: "vendor_robot",
                RobotSpecKey.ACTION_MODES: ["cartesian_delta"],
                RobotSpecKey.IMAGE_KEYS: ["rgb_front"],
                RobotSpecKey.STATE_KEYS: ["qpos"],
            }
            METHOD_ALIASES = {
                MethodAliasKey.OBSERVE: "capture",
                MethodAliasKey.ACT: "send_command",
                MethodAliasKey.RESET: "home",
            }
            MODALITY_MAPS = {
                IMAGE_KEYS: {"rgb_front": "front_rgb"},
                STATE_KEYS: {"qpos": "joint_positions"},
                ACTION_MODES: {"cartesian_delta": "ee_delta"},
            }

        class CompatibleModel(ModelMixin, VendorModel):
            MODEL_SPEC = {
                ModelSpecKey.NAME: "vendor_model",
                ModelSpecKey.REQUIRED_IMAGE_KEYS: ["rgb_front"],
                ModelSpecKey.REQUIRED_STATE_KEYS: ["qpos"],
                ModelSpecKey.OUTPUT_ACTION_MODE: "cartesian_delta",
            }
            METHOD_ALIASES = {
                MethodAliasKey.RESET: "clear_state",
                MethodAliasKey.STEP: "infer",
            }
            MODALITY_MAPS = {
                IMAGE_KEYS: {"rgb_front": "front_rgb"},
                STATE_KEYS: {"qpos": "joint_positions"},
                ACTION_MODES: {"cartesian_delta": "ee_delta"},
            }

        robot = CompatibleRobot()
        model = CompatibleModel()

        check_pair(robot, model, sample_frame=robot.reset())
        result = run_step(robot, model)

        self.assertEqual(result.action.mode, "ee_delta")
        self.assertEqual(robot.last_action.mode, "cartesian_delta")
        self.assertIn("rgb_front", model.seen_frame.images)

    def test_mixin_supports_editing_outer_class_in_place(self) -> None:
        class VendorRobot(RobotMixin):
            ROBOT_SPEC = {
                "name": "vendor_robot",
                "action_modes": ["cartesian_delta"],
                "image_keys": ["rgb_front"],
                "state_keys": ["qpos"],
            }
            METHOD_ALIASES = {
                "observe": "capture",
                "act": "send_command",
                "reset": "home",
            }
            MODALITY_MAPS = {
                IMAGE_KEYS: {"rgb_front": "front_rgb"},
                STATE_KEYS: {"qpos": "joint_positions"},
                ACTION_MODES: {"cartesian_delta": "ee_delta"},
            }

            def __init__(self) -> None:
                self.last_action = None

            def capture(self):
                return {
                    "timestamp_ns": 1,
                    "images": {"rgb_front": None},
                    "state": {"qpos": [0.0] * 6},
                }

            def send_command(self, action):
                self.last_action = action

            def home(self):
                return self.capture()

        class VendorModel(ModelMixin):
            MODEL_SPEC = {
                "name": "vendor_model",
                "required_image_keys": ["rgb_front"],
                "required_state_keys": ["qpos"],
                "output_action_mode": "cartesian_delta",
            }
            METHOD_ALIASES = {
                "reset": "clear_state",
                "step": "infer",
            }
            MODALITY_MAPS = {
                IMAGE_KEYS: {"rgb_front": "front_rgb"},
                STATE_KEYS: {"qpos": "joint_positions"},
                ACTION_MODES: {"cartesian_delta": "ee_delta"},
            }

            def clear_state(self):
                return None

            def infer(self, frame):
                self.seen_frame = frame
                return {"mode": "cartesian_delta", "value": [0.0] * 6}

        robot = VendorRobot()
        model = VendorModel()

        frame = robot.reset()
        check_pair(robot, model, sample_frame=frame)
        result = run_step(robot, model, frame=frame)

        self.assertEqual(result.action.mode, "ee_delta")
        self.assertEqual(robot.last_action.mode, "cartesian_delta")
        self.assertIn("rgb_front", model.seen_frame.images)

    def test_from_config_can_attach_runtime_interface_config(self) -> None:
        class YourRobot(RobotMixin):
            def __init__(self) -> None:
                self.last_action = None

            def capture(self):
                return {
                    "timestamp_ns": 1,
                    "images": {"rgb_front": None},
                    "state": {"qpos": [0.0] * 6},
                }

            def send_command(self, action):
                self.last_action = action

            def home(self):
                return self.capture()

        class YourModel(ModelMixin):
            def clear_state(self):
                return None

            def infer(self, frame):
                self.seen_frame = frame
                return {"mode": "cartesian_delta", "value": [0.0] * 6}

        robot = YourRobot.from_config(
            robot_spec={
                "name": "vendor_robot",
                "action_modes": ["cartesian_delta"],
                "image_keys": ["rgb_front"],
                "state_keys": ["qpos"],
            },
            method_aliases={
                "observe": "capture",
                "act": "send_command",
                "reset": "home",
            },
            modality_maps={
                IMAGE_KEYS: {"rgb_front": "front_rgb"},
                STATE_KEYS: {"qpos": "joint_positions"},
                ACTION_MODES: {"cartesian_delta": "ee_delta"},
            },
        )
        model = YourModel.from_config(
            model_spec={
                "name": "vendor_model",
                "required_image_keys": ["rgb_front"],
                "required_state_keys": ["qpos"],
                "output_action_mode": "cartesian_delta",
            },
            method_aliases={
                "reset": "clear_state",
                "step": "infer",
            },
            modality_maps={
                IMAGE_KEYS: {"rgb_front": "front_rgb"},
                STATE_KEYS: {"qpos": "joint_positions"},
                ACTION_MODES: {"cartesian_delta": "ee_delta"},
            },
        )

        check_pair(robot, model, sample_frame=robot.reset())
        result = run_step(robot, model)

        self.assertEqual(result.action.mode, "ee_delta")
        self.assertEqual(robot.last_action.mode, "cartesian_delta")
        self.assertIn("rgb_front", model.seen_frame.images)

    def test_from_config_validates_runtime_interface_config(self) -> None:
        class YourRobot(RobotMixin):
            pass

        with self.assertRaises(InterfaceValidationError) as ctx:
            YourRobot.from_config(
                method_aliases={"observe": 123},  # type: ignore[dict-item]
            )

        self.assertIn("method_aliases", str(ctx.exception))

    def test_from_config_rejects_bad_config_before_init(self) -> None:
        class YourRobot(RobotMixin):
            init_calls = 0

            def __init__(self) -> None:
                type(self).init_calls += 1

        with self.assertRaises(InterfaceValidationError) as ctx:
            YourRobot.from_config(
                method_aliases={"not_a_real_method": "capture"},
            )

        self.assertIn("unsupported key", str(ctx.exception))
        self.assertEqual(YourRobot.init_calls, 0)

    def test_from_config_rejects_bad_remote_policy_before_init(self) -> None:
        class YourRobot(RobotMixin):
            init_calls = 0

            def __init__(self) -> None:
                type(self).init_calls += 1

        with self.assertRaises(InterfaceValidationError) as ctx:
            YourRobot.from_config(
                remote_policy={"dt": 0.0},
            )

        self.assertIn("remote_policy.dt", str(ctx.exception))
        self.assertEqual(YourRobot.init_calls, 0)

    def test_mixin_must_be_leftmost_direct_base(self) -> None:
        class VendorRobot:
            pass

        class VendorModel:
            pass

        with self.assertRaises(TypeError):
            class BrokenRobot(VendorRobot, RobotMixin):
                pass

        with self.assertRaises(TypeError):
            class BrokenModel(VendorModel, ModelMixin):
                pass

    def test_run_step_unifies_robot_model_data_flow(self) -> None:
        class VendorRobot:
            def __init__(self) -> None:
                self.last_action = None

            def get_spec(self):
                return {
                    "name": "vendor_robot",
                    "action_modes": ["cartesian_delta"],
                    "image_keys": ["rgb_front"],
                    "state_keys": ["qpos"],
                }

            def observe(self):
                return {
                    "timestamp_ns": 1,
                    "images": {"rgb_front": None},
                    "state": {"qpos": [0.0] * 6},
                }

            def act(self, action):
                self.last_action = action

            def reset(self):
                return self.observe()

        class VendorModel:
            def get_spec(self):
                return {
                    "name": "vendor_model",
                    "required_image_keys": ["rgb_front"],
                    "required_state_keys": ["qpos"],
                    "output_action_mode": "cartesian_delta",
                }

            def reset(self):
                return None

            def step(self, frame):
                self.seen_frame = frame
                return {"mode": "cartesian_delta", "value": [0.0] * 6}

        class CompatibleRobot(RobotMixin, VendorRobot):
            def get_image_key_map(self):
                return {"rgb_front": "front_rgb"}

            def get_state_key_map(self):
                return {"qpos": "joint_positions"}

            def get_action_mode_map(self):
                return {"cartesian_delta": "ee_delta"}

        class CompatibleModel(ModelMixin, VendorModel):
            def get_image_key_map(self):
                return {"rgb_front": "front_rgb"}

            def get_state_key_map(self):
                return {"qpos": "joint_positions"}

            def get_action_mode_map(self):
                return {"cartesian_delta": "ee_delta"}

        robot = CompatibleRobot()
        model = CompatibleModel()

        result = run_step(robot, model)

        self.assertIsInstance(result, StepResult)
        self.assertIn("front_rgb", result.frame.images)
        self.assertEqual(result.action.mode, "ee_delta")
        self.assertIn("rgb_front", model.seen_frame.images)
        self.assertEqual(robot.last_action.mode, "cartesian_delta")

    def test_run_step_accepts_external_action_fn(self) -> None:
        robot = DummyRobot()
        seen_frames: list[Frame] = []

        def remote_policy(frame: Frame) -> dict[str, object]:
            seen_frames.append(frame)
            return {"mode": "ee_delta", "value": [1.0] * 6, "dt": 0.1}

        result = run_step(robot, remote_policy)

        self.assertIsInstance(result, StepResult)
        self.assertEqual(result.action.value, [1.0] * 6)
        self.assertEqual(robot.last_action.value, [1.0] * 6)
        self.assertEqual(len(seen_frames), 1)

    def test_run_step_rejects_multiple_action_sources(self) -> None:
        robot = DummyRobot()
        model = DummyModel()

        with self.assertRaises(InterfaceValidationError) as ctx:
            run_step(
                robot,
                model,
                action_fn=lambda frame: {
                    "mode": "ee_delta",
                    "value": [0.0] * 6,
                },
            )

        self.assertIn("either a model/callable source", str(ctx.exception))

    def test_run_step_rejects_missing_source_without_remote_policy(self) -> None:
        robot = DummyRobot()

        with self.assertRaises(InterfaceValidationError) as ctx:
            run_step(robot)

        self.assertIn("configured remote policy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
