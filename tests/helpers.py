"""Shared dummy implementations for tests."""

from __future__ import annotations

import numpy as np

from inferaxis import Action, ChunkRequest, Frame, PolicySpec, RobotSpec
from inferaxis.core.schema import ComponentSpec, PolicyOutputSpec


def demo_image() -> np.ndarray:
    """Return one small deterministic image tensor for tests/examples."""

    return np.zeros((2, 2, 3), dtype=np.uint8)


def assert_array_equal(testcase: object, actual: object, expected: object) -> None:
    """Assert one array-like payload matches the expected numeric values."""

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


class DummyRobot:
    def __init__(self) -> None:
        self.last_action: Action | None = None

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="dummy_robot",
            image_keys=["front_rgb"],
            components=[
                ComponentSpec(
                    name="arm",
                    type="arm",
                    dof=6,
                    command=["cartesian_pose_delta"],
                )
            ],
        )

    def get_obs(self) -> Frame:
        return Frame(
            images={"front_rgb": demo_image()},
            state={"arm": np.zeros(6, dtype=np.float64)},
        )

    def send_action(self, action: Action) -> None:
        self.last_action = action

    def reset(self) -> Frame:
        return self.get_obs()


class DummyPolicy:
    def get_spec(self) -> PolicySpec:
        return PolicySpec(
            name="dummy_model",
            required_image_keys=["front_rgb"],
            required_state_keys=["arm"],
            outputs=[
                PolicyOutputSpec(
                    target="arm",
                    command="cartesian_pose_delta",
                    dim=6,
                )
            ],
        )

    def reset(self) -> None:
        return None

    def infer(
        self,
        obs: Frame,
        request: ChunkRequest,
    ) -> Action:
        del obs, request
        return Action.single(
            target="arm",
            command="cartesian_pose_delta",
            value=np.zeros(6, dtype=np.float64),
        )
