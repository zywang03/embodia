"""Tests for embodia's optional YAML-based runtime configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import embodia as em
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
        path = self._write_config({"robot": {"method_aliases": {"observe": "capture"}}})

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            loaded = em.load_yaml_config(path)

        self.assertIn("robot", loaded)

    def test_load_yaml_config_reads_named_section(self) -> None:
        path = self._write_config(
            {
                "robot": {"method_aliases": {"observe": "capture"}},
                "model": {"method_aliases": {"step": "infer"}},
            }
        )

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            loaded = em.load_yaml_config(path, section="model")

        self.assertEqual(loaded["method_aliases"]["step"], "infer")

    def test_from_yaml_builds_grouped_specs_with_shared_schema(self) -> None:
        path = self._write_config(
            {
                "schema": {
                    "images": ["front_rgb"],
                    "groups": {
                        "arm": {
                            "kind": "arm",
                            "dof": 6,
                            "state": ["joint_positions"],
                            "command_kinds": ["cartesian_pose_delta"],
                        },
                        "gripper": {
                            "kind": "gripper",
                            "dof": 1,
                            "state": ["position"],
                            "command_kinds": ["gripper_position"],
                        },
                    },
                    "task": ["prompt"],
                },
                "robot": {
                    "name": "demo_robot",
                    "method_aliases": {
                        "observe": "capture",
                        "act": "send_command",
                        "reset": "home",
                    },
                },
                "model": {
                    "name": "demo_model",
                    "requires": {
                        "images": ["front_rgb"],
                        "state": ["joint_positions", "position"],
                        "task": ["prompt"],
                    },
                    "outputs": {
                        "arm": "cartesian_pose_delta",
                        "gripper": "gripper_position",
                    },
                    "method_aliases": {
                        "reset": "clear_state",
                        "step": "infer",
                    },
                },
            }
        )

        class YourRobot(em.RobotMixin):
            def __init__(self, label: str = "default") -> None:
                self.label = label
                self.last_action = None

            def capture(self):
                return {
                    "timestamp_ns": 1,
                    "images": {"front_rgb": None},
                    "state": {
                        "joint_positions": [0.0] * 6,
                        "position": 0.5,
                    },
                    "task": {"prompt": "fold the cloth"},
                }

            def send_command(self, action):
                self.last_action = action

            def home(self):
                return self.capture()

        class YourModel(em.ModelMixin):
            def __init__(self, gain: float = 0.0) -> None:
                self.gain = gain
                self.seen_frame = None

            def clear_state(self):
                return None

            def infer(self, frame):
                self.seen_frame = frame
                return {
                    "commands": [
                        {
                            "target": "arm",
                            "kind": "cartesian_pose_delta",
                            "value": [self.gain] * 6,
                        },
                        {
                            "target": "gripper",
                            "kind": "gripper_position",
                            "value": [0.3],
                        },
                    ]
                }

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            robot = YourRobot.from_yaml(path, label="from_yaml")
            model = YourModel.from_yaml(path, gain=0.5)

        sample_frame = robot.reset()
        em.check_pair(robot, model, sample_frame=sample_frame)
        result = em.run_step(robot, model)

        self.assertEqual(robot.label, "from_yaml")
        self.assertEqual(model.gain, 0.5)
        self.assertEqual(result.action.get_command("arm").value, [0.5] * 6)  # type: ignore[union-attr]
        self.assertEqual(robot.last_action.get_command("gripper").value, [0.3])  # type: ignore[union-attr]
        self.assertIn("joint_positions", model.seen_frame.state)
        self.assertEqual(model.seen_frame.task["prompt"], "fold the cloth")

    def test_from_yaml_rejects_unknown_group_output(self) -> None:
        path = self._write_config(
            {
                "schema": {
                    "groups": {
                        "arm": {
                            "kind": "arm",
                            "dof": 6,
                            "state": ["joint_positions"],
                            "command_kinds": ["cartesian_pose_delta"],
                        }
                    }
                },
                "model": {
                    "name": "demo_model",
                    "outputs": {
                        "gripper": "gripper_position",
                    },
                },
            }
        )

        class YourModel(em.ModelMixin):
            pass

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            with self.assertRaises(em.InterfaceValidationError) as ctx:
                YourModel.from_yaml(path)

        self.assertIn("unknown target", str(ctx.exception))

    def test_from_yaml_rejects_unknown_runtime_field_before_init(self) -> None:
        path = self._write_config(
            {
                "schema": {
                    "groups": {
                        "arm": {
                            "kind": "arm",
                            "dof": 6,
                            "state": ["joint_positions"],
                            "command_kinds": ["cartesian_pose_delta"],
                        }
                    }
                },
                "robot": {
                    "name": "robot",
                    "unsupported": True,
                },
            }
        )

        class YourRobot(em.RobotMixin):
            pass

        with mock.patch.object(config_io, "_import_yaml", return_value=_JsonYamlModule):
            with self.assertRaises(em.InterfaceValidationError):
                YourRobot.from_yaml(path)


if __name__ == "__main__":
    unittest.main()
