"""Tests for embodia's optional YAML-based runtime configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from embodia import (
    InterfaceValidationError,
    ModelMixin,
    RobotMixin,
    check_pair,
    load_component_yaml_config,
    load_yaml_config,
    require_yaml,
)
from embodia.core import config_io


class _JsonYamlModule:
    """Small stand-in parser so tests do not depend on real PyYAML."""

    @staticmethod
    def safe_load(text: str) -> object:
        return json.loads(text)


class YamlConfigTests(unittest.TestCase):
    """Coverage for optional config-file loading and mixin construction."""

    def _write_config(self, payload: object) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "config.yml"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_yaml_config_reads_top_level_mapping(self) -> None:
        path = self._write_config(
            {
                "robot_spec": {
                    "name": "robot",
                    "action_modes": ["ee_delta"],
                    "image_keys": ["front_rgb"],
                    "state_keys": ["joint_positions"],
                }
            }
        )

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            loaded = load_yaml_config(path)

        self.assertEqual(loaded["robot_spec"]["name"], "robot")

    def test_load_yaml_config_reads_named_section(self) -> None:
        path = self._write_config(
            {
                "robot": {"method_aliases": {"observe": "capture"}},
                "model": {"method_aliases": {"step": "infer"}},
            }
        )

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            loaded = load_yaml_config(path, section="model")

        self.assertEqual(loaded["method_aliases"]["step"], "infer")

    def test_load_yaml_config_rejects_missing_section(self) -> None:
        path = self._write_config({"robot": {"name": "only_robot"}})

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            with self.assertRaises(InterfaceValidationError) as ctx:
                load_yaml_config(path, section="model")

        self.assertIn("missing section", str(ctx.exception))

    def test_load_component_yaml_config_reads_component_section(self) -> None:
        path = self._write_config(
            {
                "robot": {"method_aliases": {"observe": "capture"}},
                "model": {"method_aliases": {"step": "infer"}},
            }
        )

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            loaded = load_component_yaml_config(path, component="robot")

        self.assertEqual(loaded["method_aliases"]["observe"], "capture")

    def test_load_component_yaml_config_accepts_direct_mapping(self) -> None:
        path = self._write_config(
            {
                "robot_spec": {
                    "name": "robot",
                    "action_modes": ["ee_delta"],
                    "image_keys": ["front_rgb"],
                    "state_keys": ["joint_positions"],
                }
            }
        )

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            loaded = load_component_yaml_config(path, component="robot")

        self.assertEqual(loaded["robot_spec"]["name"], "robot")

    def test_require_yaml_reports_missing_optional_dependency(self) -> None:
        with mock.patch.object(config_io.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(InterfaceValidationError) as ctx:
                require_yaml()

        self.assertIn("embodia[yaml]", str(ctx.exception))

    def test_from_yaml_builds_runtime_config_and_merges_init_overrides(self) -> None:
        path = self._write_config(
            {
                "robot": {
                    "init": {"label": "from_yaml"},
                    "robot_spec": {
                        "name": "vendor_robot",
                        "action_modes": ["cartesian_delta"],
                        "image_keys": ["rgb_front"],
                        "state_keys": ["qpos"],
                    },
                    "method_aliases": {
                        "observe": "capture",
                        "act": "send_command",
                        "reset": "home",
                    },
                    "modality_maps": {
                        "images": {"rgb_front": "front_rgb"},
                        "state": {"qpos": "joint_positions"},
                        "action_modes": {"cartesian_delta": "ee_delta"},
                    },
                },
                "model": {
                    "init": {"gain": 0.25},
                    "model_spec": {
                        "name": "vendor_model",
                        "required_image_keys": ["rgb_front"],
                        "required_state_keys": ["qpos"],
                        "output_action_mode": "cartesian_delta",
                    },
                    "method_aliases": {
                        "reset": "clear_state",
                        "step": "infer",
                    },
                    "modality_maps": {
                        "images": {"rgb_front": "front_rgb"},
                        "state": {"qpos": "joint_positions"},
                        "action_modes": {"cartesian_delta": "ee_delta"},
                    },
                },
            }
        )

        class YourRobot(RobotMixin):
            def __init__(self, label: str = "default") -> None:
                self.label = label
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
            def __init__(self, gain: float = 0.0) -> None:
                self.gain = gain

            def clear_state(self):
                return None

            def infer(self, frame):
                self.seen_frame = frame
                return {
                    "mode": "cartesian_delta",
                    "value": [self.gain] * 6,
                }

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            robot = YourRobot.from_yaml(path)
            model = YourModel.from_yaml(path, gain=0.5)

        check_pair(robot, model, sample_frame=robot.reset())

        self.assertEqual(robot.label, "from_yaml")
        self.assertEqual(model.gain, 0.5)
        self.assertIn("front_rgb", robot.observe().images)
        self.assertEqual(model.step(robot.observe()).mode, "ee_delta")
        self.assertIn("rgb_front", model.seen_frame.images)

    def test_from_yaml_rejects_non_mapping_init_field(self) -> None:
        path = self._write_config(
            {
                "robot": {
                    "init": ["not", "a", "mapping"],
                }
            }
        )

        class YourRobot(RobotMixin):
            pass

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            with self.assertRaises(InterfaceValidationError) as ctx:
                YourRobot.from_yaml(path)

        self.assertIn("init", str(ctx.exception))

    def test_from_yaml_rejects_unknown_runtime_field_before_init(self) -> None:
        path = self._write_config(
            {
                "robot": {
                    "robot_spec": {
                        "name": "robot",
                        "action_modes": ["ee_delta"],
                        "image_keys": ["front_rgb"],
                        "state_keys": ["joint_positions"],
                    },
                    "method_aliasess": {
                        "observe": "capture",
                    },
                }
            }
        )

        class YourRobot(RobotMixin):
            init_calls = 0

            def __init__(self) -> None:
                type(self).init_calls += 1

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            with self.assertRaises(InterfaceValidationError) as ctx:
                YourRobot.from_yaml(path)

        self.assertIn("unsupported field", str(ctx.exception))
        self.assertEqual(YourRobot.init_calls, 0)


if __name__ == "__main__":
    unittest.main()
