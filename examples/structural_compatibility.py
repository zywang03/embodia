"""Example showing Protocol-style compatibility without inheriting base classes."""

from __future__ import annotations

import time

from embodia import Action, Frame, ModelSpec, RobotSpec, check_model, check_pair, check_robot


class ForeignRobot:
    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="foreign_robot",
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
        print(f"[ForeignRobot] action={action}")

    def reset(self) -> Frame:
        return self.observe()


class ForeignModel:
    def get_spec(self) -> ModelSpec:
        return ModelSpec(
            name="foreign_model",
            required_image_keys=["front_rgb"],
            required_state_keys=["joint_positions"],
            output_action_mode="ee_delta",
        )

    def reset(self) -> None:
        return None

    def step(self, frame: Frame) -> Action:
        return Action(mode="ee_delta", value=[0.0] * 6)


def main() -> None:
    robot = ForeignRobot()
    model = ForeignModel()
    frame = robot.observe()

    check_robot(robot)
    check_model(model, sample_frame=frame)
    check_pair(robot, model)
    print("Structural compatibility example passed.")


if __name__ == "__main__":
    main()
