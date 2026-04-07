"""Example 4: replay collected actions back into the robot.

Run with:

    PYTHONPATH=src python examples/04_replay_collected_data.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import embodia as em


class YourRobot(em.RobotMixin):
    """Pretend this is your original outer robot class after one small edit."""

    def __init__(self) -> None:
        self.step_index = 0
        self.last_native_action: object | None = None

    def capture(self) -> dict[str, object]:
        self.step_index += 1
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"rgb_front": None},
            "state": {
                "qpos": [float(self.step_index)] * 6,
                "gripper_pos": min(self.step_index * 0.1, 1.0),
            },
        }

    def send_command(self, action: object) -> None:
        self.last_native_action = action

    def home(self) -> dict[str, object]:
        self.step_index = 0
        return self.capture()


def scripted_action(frame: em.Frame) -> dict[str, object]:
    """Generate a tiny demo recording when the replay file is missing."""

    qpos0 = float(frame.state["joint_positions"][0])
    gripper_pos = float(frame.state["position"])
    return {
        "commands": [
            {
                "target": "arm",
                "mode": "ee_delta",
                "value": [qpos0 * 0.01] * 6,
            },
            {
                "target": "gripper",
                "mode": "scalar_position",
                "value": [max(0.0, min(1.0, 1.0 - gripper_pos))],
            },
        ],
        "dt": 0.1,
    }


def ensure_demo_episode(path: Path, robot: YourRobot) -> None:
    """Create one small recording first so the replay example is self-contained."""

    if path.exists():
        return

    records: list[dict[str, object]] = [
        {"frame": em.frame_to_dict(robot.reset()), "action": None}
    ]
    for _ in range(3):
        result = em.run_step(robot, action_fn=scripted_action)
        records.append(
            {
                "frame": em.frame_to_dict(result.frame),
                "action": em.action_to_dict(result.action),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
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
            action = record.get("action")
            if action is None:
                continue
            robot.act(action)
            replayed += 1

    print("episode_path:", episode_path)
    print("replayed_actions:", replayed)
    print("last_native_action:", robot.last_native_action)
    print("example 4 passed.")


if __name__ == "__main__":
    main()
