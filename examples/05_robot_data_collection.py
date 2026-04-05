"""Example 5: robot-only data collection as an embodia sub-application.

Run with:

    PYTHONPATH=src python examples/05_robot_data_collection.py
"""

from __future__ import annotations

import time

import embodia as em


class YourRobot(em.RobotMixin):
    """A small robot that already exposes native methods and names."""

    ROBOT_SPEC = {
        "name": "your_robot",
        "action_modes": ["cartesian_delta"],
        "image_keys": ["rgb_front"],
        "state_keys": ["qpos"],
    }
    METHOD_ALIASES = {
        "observe": "capture",
        "act": "send_command",
        "reset": "home",
    }
    IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
    STATE_KEY_MAP = {"qpos": "joint_positions"}
    ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

    def __init__(self) -> None:
        self.step_count = 0
        self.last_native_action: object | None = None

    def capture(self) -> dict[str, object]:
        self.step_count += 1
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"rgb_front": None},
            "state": {"qpos": [float(self.step_count)] * 6},
            "meta": {"sensor_source": "demo_camera"},
        }

    def send_command(self, action: object) -> None:
        self.last_native_action = action

    def home(self) -> dict[str, object]:
        self.step_count = 0
        return self.capture()


def scripted_action(frame: em.Frame) -> dict[str, object]:
    """Pretend this came from teleop or a scripted collector."""

    joint_0 = float(frame.state["joint_positions"][0])
    return {
        "mode": "ee_delta",
        "value": [joint_0 * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dt": 0.1,
    }


def main() -> None:
    robot = YourRobot()
    em.check_robot(robot, call_observe=False)

    one_step = em.record_step(robot)
    episode = em.collect_episode(
        robot,
        steps=3,
        action_fn=scripted_action,
        execute_actions=True,
        reset_robot=True,
        include_reset_frame=True,
        episode_meta={"collector": "demo_script"},
    )

    print("one_step:", em.episode_step_to_dict(one_step))
    print("episode_length:", len(episode.steps))
    print("first_episode_step:", em.episode_to_dict(episode)["steps"][0])
    print("last_native_action:", robot.last_native_action)
    print("example 5 passed.")


if __name__ == "__main__":
    main()
