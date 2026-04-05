"""Example 1: implement a fresh robot/model pair using only embodia mixins.

Run with:

    PYTHONPATH=src python examples/01_mixin_from_scratch.py
"""

from __future__ import annotations

import time

import embodia as em


class DemoRobot(em.RobotMixin):
    """Fresh robot implementation using only embodia mixin hooks."""

    def _get_spec_impl(self) -> dict[str, object]:
        return {
            "name": "demo_robot",
            "action_modes": ["ee_delta"],
            "image_keys": ["front_rgb"],
            "state_keys": ["joint_positions", "ee_pose"],
        }

    def _observe_impl(self) -> dict[str, object]:
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"front_rgb": None},
            "state": {
                "joint_positions": [0.0] * 6,
                "ee_pose": [0.0] * 7,
            },
            "meta": {"source": "robot.observe"},
        }

    def _act_impl(self, action: em.Action) -> None:
        print(f"[DemoRobot] execute {action}")

    def _reset_impl(self) -> dict[str, object]:
        return self._observe_impl()


class DemoModel(em.ModelMixin):
    """Fresh model implementation using only embodia mixin hooks."""

    def _get_spec_impl(self) -> dict[str, object]:
        return {
            "name": "demo_model",
            "required_image_keys": ["front_rgb"],
            "required_state_keys": ["joint_positions"],
            "output_action_mode": "ee_delta",
        }

    def _reset_impl(self) -> None:
        return None

    def _step_impl(self, frame: em.Frame) -> dict[str, object]:
        return {
            "mode": "ee_delta",
            "value": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gripper": 0.0,
            "frame": "tool",
            "dt": 0.1,
        }


def main() -> None:
    robot = DemoRobot()
    model = DemoModel()

    reset_frame = robot.reset()
    print("reset_frame:", em.frame_to_dict(reset_frame))

    em.check_robot(robot)
    em.check_model(model, sample_frame=reset_frame)
    em.check_pair(robot, model)

    result = em.run_step(robot, model)
    print("normalized_frame:", em.frame_to_dict(result.frame))
    print("normalized_action:", em.action_to_dict(result.action))
    print("example 1 passed.")


if __name__ == "__main__":
    main()
