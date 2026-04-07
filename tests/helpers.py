"""Shared dummy implementations for tests."""

from __future__ import annotations

import time

from embodia import Action, Frame, ModelMixin, RobotMixin


class DummyRobot(RobotMixin):
    def __init__(self) -> None:
        self.last_action: Action | None = None

    def _get_spec_impl(self) -> dict[str, object]:
        return {
            "name": "dummy_robot",
            "image_keys": ["front_rgb"],
            "groups": [
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

    def _act_impl(self, action: Action) -> None:
        self.last_action = action

    def _reset_impl(self) -> dict[str, object]:
        return self._observe_impl()


class DummyModel(ModelMixin):
    def _get_spec_impl(self) -> dict[str, object]:
        return {
            "name": "dummy_model",
            "required_image_keys": ["front_rgb"],
            "required_state_keys": ["joint_positions"],
            "required_task_keys": [],
            "outputs": [
                {
                    "target": "arm",
                    "command_kind": "cartesian_pose_delta",
                    "dim": 6,
                }
            ],
        }

    def _reset_impl(self) -> None:
        return None

    def _step_impl(self, frame: Frame) -> dict[str, object]:
        return {
            "commands": [
                {
                    "target": "arm",
                    "kind": "cartesian_pose_delta",
                    "value": [0.0] * 6,
                }
            ]
        }
