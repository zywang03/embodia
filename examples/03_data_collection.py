"""Example 3: robot-only data collection with plain Python save logic.

Run with:

    PYTHONPATH=src python examples/03_data_collection.py
"""

from __future__ import annotations

import json
from pathlib import Path

import embodia as em
import numpy as np


class YourRobot(em.RobotMixin):
    """Pretend this is your original outer robot class after one small edit."""

    def __init__(self) -> None:
        self.last_native_action: object | None = None

    def YOUR_OWN_get_obs(self) -> dict[str, object]:
        return {
            "images": {"YOUR_OWN_front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            "state": {
                "YOUR_OWN_arm": np.full(6, 0.25, dtype=np.float64),
                "YOUR_OWN_gripper": np.array([0.5], dtype=np.float64),
            },
        }

    def YOUR_OWN_send_action(self, action: object) -> object:
        """Pretend the robot controller returns the final accepted action."""

        accepted = em.coerce_action(action)
        self.last_native_action = accepted
        return accepted

    def YOUR_OWN_reset(self) -> dict[str, object]:
        return self.YOUR_OWN_get_obs()


class DemoTeleop:
    """Tiny stand-in for teleop input or an external data-collection source."""

    def __init__(self) -> None:
        self.cursor = 0

    script = [
        {
            "YOUR_OWN_arm": {
                "command": "cartesian_pose_delta",
                "value": np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            },
            "YOUR_OWN_gripper": {
                "command": "gripper_position",
                "value": np.array([1.0], dtype=np.float64),
            },
        },
        {
            "YOUR_OWN_arm": {
                "command": "cartesian_pose_delta",
                "value": np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            },
            "YOUR_OWN_gripper": {
                "command": "gripper_position",
                "value": np.array([0.5], dtype=np.float64),
            },
        },
        {
            "YOUR_OWN_arm": {
                "command": "cartesian_pose_delta",
                "value": np.array([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            },
            "YOUR_OWN_gripper": {
                "command": "gripper_position",
                "value": np.array([0.0], dtype=np.float64),
            },
        },
    ]

    def next_action(self, frame: em.Frame) -> dict[str, object]:
        """Return one scripted action without exposing runtime metadata."""

        del frame
        index = min(self.cursor, len(self.script) - 1)
        self.cursor += 1
        return self.script[index]


def main() -> None:
    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    teleop = DemoTeleop()
    output_path = Path("tmp") / "episode_0000.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    reset_frame = robot.reset()
    records.append({"frame": em.frame_to_dict(reset_frame), "action": None})

    for _ in range(3):
        result = em.run_step(robot, source=teleop)
        records.append(
            {
                "frame": em.frame_to_dict(result.frame),
                "action": em.action_to_dict(result.action),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            # JSON serialization converts numpy arrays into plain lists on disk.
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")

    print("record_count:", len(records))
    print("jsonl_path:", output_path)
    print("last_native_action:", robot.last_native_action)
    print("example 3 passed.")


if __name__ == "__main__":
    main()
