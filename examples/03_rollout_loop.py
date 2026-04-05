"""Example 3: run a short standardized rollout and collect a trajectory.

Run with:

    PYTHONPATH=src python examples/03_rollout_loop.py
"""

from __future__ import annotations

import time

import embodia as em


class LoopRobot:
    def __init__(self) -> None:
        self.step_count = 0
        self.last_native_action: object | None = None

    def observe(self) -> dict[str, object]:
        self.step_count += 1
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"rgb_front": None},
            "state": {"qpos": [float(self.step_count)] * 6},
            "meta": {"step_count": self.step_count},
        }

    def act(self, action: object) -> None:
        self.last_native_action = action

    def reset(self) -> dict[str, object]:
        self.step_count = 0
        return self.observe()


class LoopModel:
    def reset(self) -> None:
        return None

    def step(self, frame: object) -> dict[str, object]:
        qpos = frame.state["qpos"]
        return {
            "mode": "cartesian_delta",
            "value": [qpos[0] * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dt": 0.1,
        }


class UnifiedLoopRobot(em.RobotMixin, LoopRobot):
    ROBOT_SPEC = {
        "name": "loop_robot",
        "action_modes": ["cartesian_delta"],
        "image_keys": ["rgb_front"],
        "state_keys": ["qpos"],
    }
    IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
    STATE_KEY_MAP = {"qpos": "joint_positions"}
    ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}


class UnifiedLoopModel(em.ModelMixin, LoopModel):
    MODEL_SPEC = {
        "name": "loop_model",
        "required_image_keys": ["rgb_front"],
        "required_state_keys": ["qpos"],
        "output_action_mode": "cartesian_delta",
    }
    IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
    STATE_KEY_MAP = {"qpos": "joint_positions"}
    ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}


def main() -> None:
    robot = UnifiedLoopRobot()
    model = UnifiedLoopModel()
    em.check_pair(robot, model, sample_frame=robot.reset())

    trajectory: list[dict[str, object]] = []
    for _ in range(3):
        result = em.run_step(robot, model)
        trajectory.append(
            {
                "frame": em.frame_to_dict(result.frame),
                "action": em.action_to_dict(result.action),
            }
        )

    print("trajectory_length:", len(trajectory))
    print("first_step:", trajectory[0])
    print("last_native_action:", robot.last_native_action)
    print("example 3 passed.")


if __name__ == "__main__":
    main()
