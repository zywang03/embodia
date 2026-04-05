"""Basic tests for the embodia package."""

from __future__ import annotations

import json
import tempfile
import unittest

from embodia import (
    Action,
    Episode,
    EpisodeStep,
    Frame,
    InterfaceValidationError,
    ModelMixin,
    RobotMixin,
    StepResult,
    check_model,
    check_pair,
    check_robot,
    collect_episode,
    coerce_action,
    episode_step_to_dict,
    episode_to_dict,
    frame_to_dict,
    record_step,
    remap_frame,
    run_step,
    validate_action,
    validate_frame,
)
from embodia.contrib import lerobot as em_lerobot
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
        action = coerce_action({"mode": "ee_delta", "value": (0.0, 1.0)})

        self.assertIsInstance(action, Action)
        self.assertEqual(action.value, [0.0, 1.0])

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

    def test_record_step_collects_observation_only(self) -> None:
        robot = DummyRobot()

        step = record_step(robot)

        self.assertIsInstance(step, EpisodeStep)
        self.assertEqual(step.action, None)
        self.assertIn("front_rgb", step.frame.images)

    def test_collect_episode_robot_only_allows_scripted_action_source(self) -> None:
        robot = DummyRobot()

        def scripted_action(frame: Frame) -> dict[str, object]:
            return {"mode": "ee_delta", "value": [frame.timestamp_ns * 0.0] * 6}

        episode = collect_episode(
            robot,
            steps=2,
            action_fn=scripted_action,
            execute_actions=True,
            reset_robot=True,
            include_reset_frame=True,
            episode_meta={"source": "scripted"},
        )

        self.assertIsInstance(episode, Episode)
        self.assertEqual(len(episode.steps), 3)
        self.assertEqual(episode.steps[0].meta["source"], "reset")
        self.assertIsNotNone(robot.last_action)
        self.assertEqual(episode.meta["source"], "scripted")

    def test_collect_episode_with_model_records_standardized_actions(self) -> None:
        robot = DummyRobot()
        model = DummyModel()

        episode = collect_episode(
            robot,
            steps=2,
            model=model,
            execute_actions=False,
            reset_model=True,
        )

        self.assertEqual(len(episode.steps), 2)
        self.assertEqual(episode.steps[0].action.mode, "ee_delta")
        self.assertIsNone(robot.last_action)
        self.assertEqual(episode.model_spec.output_action_mode, "ee_delta")

    def test_episode_export_helpers_return_plain_dicts(self) -> None:
        robot = DummyRobot()
        step = record_step(robot, step_meta={"collector": "test"})
        episode = Episode(robot_spec=robot.get_spec(), steps=[step])

        exported_step = episode_step_to_dict(step)
        exported_episode = episode_to_dict(episode)

        self.assertEqual(exported_step["meta"]["collector"], "test")
        self.assertEqual(exported_episode["robot_spec"]["name"], "dummy_robot")
        self.assertEqual(len(exported_episode["steps"]), 1)

    def test_lerobot_bridge_converts_episode_to_records(self) -> None:
        robot = DummyRobot()
        episode = collect_episode(robot, steps=2)

        records = em_lerobot.episode_to_lerobot_records(episode, episode_index=3)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["episode_index"], 3)
        self.assertIn("observation.images", records[0])
        self.assertFalse(records[0]["next.done"])
        self.assertTrue(records[-1]["next.done"])

    def test_lerobot_bridge_writes_jsonl(self) -> None:
        robot = DummyRobot()
        episode = collect_episode(robot, steps=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = em_lerobot.write_lerobot_jsonl(
                episode,
                f"{tmpdir}/episode_0001.jsonl",
                episode_index=1,
            )
            with open(path, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode_index"], 1)
        self.assertIn("observation.state", rows[0])

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
            IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
            STATE_KEY_MAP = {"qpos": "joint_positions"}
            ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

        class CompatibleModel(ModelMixin, VendorModel):
            GET_SPEC_METHOD = "describe"
            RESET_METHOD = "clear_state"
            STEP_METHOD = "infer"
            IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
            STATE_KEY_MAP = {"qpos": "joint_positions"}
            ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

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
            IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
            STATE_KEY_MAP = {"qpos": "joint_positions"}
            ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

        class CompatibleModel(ModelMixin, VendorModel):
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
            IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
            STATE_KEY_MAP = {"qpos": "joint_positions"}
            ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

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
            IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
            STATE_KEY_MAP = {"qpos": "joint_positions"}
            ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

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
            IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
            STATE_KEY_MAP = {"qpos": "joint_positions"}
            ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

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


if __name__ == "__main__":
    unittest.main()
