"""Example 4: replay collected actions back into the robot.

Run with:

    PYTHONPATH=src python examples/04_replay_collected_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import inferaxis as infra
import numpy as np


class YourRobot(infra.RobotMixin):
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

        accepted = infra.coerce_action(action)
        self.last_native_action = accepted
        return accepted

    def YOUR_OWN_reset(self) -> dict[str, object]:
        return self.YOUR_OWN_get_obs()


class DemoTeleop:
    """Tiny stand-in for recorded operator input when bootstrapping the demo."""

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

    def next_action(self, frame: infra.Frame) -> dict[str, object]:
        """Return one scripted action without exposing runtime metadata."""

        del frame
        index = min(self.cursor, len(self.script) - 1)
        self.cursor += 1
        return self.script[index]


def main() -> None:
    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    episode_path = Path("tmp") / "episode_0000.jsonl"
    if not episode_path.exists():
        teleop = DemoTeleop()
        records: list[dict[str, object]] = [
            {"frame": infra.frame_to_dict(robot.reset()), "action": None}
        ]
        for _ in range(3):
            result = infra.run_step(robot, source=teleop)
            records.append(
                {
                    "frame": infra.frame_to_dict(result.frame),
                    "action": infra.action_to_dict(result.action),
                }
            )

        episode_path.parent.mkdir(parents=True, exist_ok=True)
        with episode_path.open("w", encoding="utf-8") as handle:
            for record in records:
                # JSON serialization converts numpy arrays into plain lists on disk.
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")

    replayed = 0
    with episode_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = record.get("frame")
            action = record.get("action")
            if frame is None or action is None:
                continue
            infra.run_step(
                robot,
                frame=frame,
                action_fn=lambda _frame, recorded_action=action: recorded_action,
            )
            replayed += 1

    print("episode_path:", episode_path)
    print("replayed_actions:", replayed)
    print("last_native_action:", robot.last_native_action)
    print("example 4 passed.")


if __name__ == "__main__":
    main()
