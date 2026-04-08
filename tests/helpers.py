"""Shared dummy implementations for tests."""

from __future__ import annotations

import time

import numpy as np

from embodia import Action, Frame, PolicyMixin, RobotMixin


def demo_image() -> np.ndarray:
    """Return one small deterministic image tensor for tests/examples."""

    return np.zeros((2, 2, 3), dtype=np.uint8)


def assert_array_equal(testcase: object, actual: object, expected: object) -> None:
    """Assert one array-like payload matches the expected numeric values."""

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


class DummyRobot(RobotMixin):
    def __init__(self) -> None:
        self.last_action: Action | None = None

    def _get_spec_impl(self) -> dict[str, object]:
        return {
            "name": "dummy_robot",
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
            "images": {"front_rgb": demo_image()},
            "state": {"joint_positions": np.zeros(6, dtype=np.float64)},
        }

    def _act_impl(self, action: Action) -> None:
        self.last_action = action

    def _reset_impl(self) -> dict[str, object]:
        return self._observe_impl()


class DummyPolicy(PolicyMixin):
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
            "arm": {
                "kind": "cartesian_pose_delta",
                "value": [0.0] * 6,
            }
        }
