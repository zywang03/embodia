"""Basic embodia usage example.

Run this after installing the package in editable mode:

    pip install -e .
    python examples/basic_usage.py
"""

from __future__ import annotations

import time

from embodia import (
    Action,
    Frame,
    ModelBase,
    ModelSpec,
    RobotBase,
    RobotSpec,
    check_model,
    check_pair,
    check_robot,
)


class ExampleRobot(RobotBase):
    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="example_robot",
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
        print(f"[ExampleRobot] action={action}")

    def reset(self) -> Frame:
        return self.observe()


class ExampleModel(ModelBase):
    def get_spec(self) -> ModelSpec:
        return ModelSpec(
            name="example_model",
            required_image_keys=["front_rgb"],
            required_state_keys=["joint_positions"],
            output_action_mode="ee_delta",
        )

    def reset(self) -> None:
        return None

    def step(self, frame: Frame) -> Action:
        return Action(
            mode="ee_delta",
            value=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper=0.0,
            frame="tool",
            dt=0.1,
        )


def main() -> None:
    robot = ExampleRobot()
    model = ExampleModel()
    frame = robot.observe()

    check_robot(robot)
    check_model(model, sample_frame=frame)
    check_pair(robot, model)

    action = model.step(frame)
    robot.act(action)
    print("embodia basic usage example passed.")


if __name__ == "__main__":
    main()
