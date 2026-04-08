"""Example 4: replay collected actions back into the robot.

Run with:

    PYTHONPATH=src python examples/04_replay_collected_data.py
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
    """Tiny stand-in for recorded operator input when bootstrapping the demo."""

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
        """Return the scripted action indexed by embodia-managed sequence_id."""

        sequence_id = int(frame.sequence_id or 0)
        return self.script[min(sequence_id, len(self.script) - 1)]


def ensure_demo_episode(path: Path, robot: YourRobot) -> None:
    """Create one small recording first so the replay example is self-contained."""

    if path.exists():
        return

    teleop = DemoTeleop()
    records: list[dict[str, object]] = [
        {"frame": em.frame_to_dict(robot.reset()), "action": None}
    ]
    for _ in range(3):
        result = em.run_step(robot, source=teleop)
        records.append(
            {
                "frame": em.frame_to_dict(result.frame),
                "action": em.action_to_dict(result.action),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            # JSON serialization converts numpy arrays into plain lists on disk.
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def main() -> None:
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    episode_path = Path("tmp") / "episode_0000.jsonl"
    ensure_demo_episode(episode_path, robot)

    replayed = 0
    with episode_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = record.get("frame")
            action = record.get("action")
            if frame is None or action is None:
                continue
            em.run_step(
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
