"""Shared dummy implementations for tests."""

from __future__ import annotations

import time

from embodia import Action, Frame, ModelBase, ModelSpec, RobotBase, RobotSpec


class DummyRobot(RobotBase):
    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="dummy_robot",
            action_modes=["ee_delta"],
            image_keys=["front_rgb"],
            state_keys=["joint_positions"],
        )

    def observe(self) -> Frame:
        return Frame(
            timestamp_ns=time.time_ns(),
            images={"front_rgb": None},
            state={"joint_positions": [0.0] * 6},
        )

    def act(self, action: Action) -> None:
        return None

    def reset(self) -> Frame:
        return self.observe()


class DummyModel(ModelBase):
    def get_spec(self) -> ModelSpec:
        return ModelSpec(
            name="dummy_model",
            required_image_keys=["front_rgb"],
            required_state_keys=["joint_positions"],
            output_action_mode="ee_delta",
        )

    def reset(self) -> None:
        return None

    def step(self, frame: Frame) -> Action:
        return Action(mode="ee_delta", value=[0.0] * 6)
