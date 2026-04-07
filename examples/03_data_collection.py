"""Example 3: robot-only data collection with plain Python save logic.

Run with:

    PYTHONPATH=src python examples/03_data_collection.py
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
            "meta": {"step_index": self.step_index},
        }

    def send_command(self, action: object) -> None:
        self.last_native_action = action

    def home(self) -> dict[str, object]:
        self.step_index = 0
        return self.capture()


def scripted_action(frame: em.Frame) -> dict[str, object]:
    """Pretend this action comes from teleop or a scripted collector."""

    qpos0 = float(frame.state["joint_positions"][0])
    gripper_pos = float(frame.state["position"])
    return {
        "commands": [
            {
                "target": "arm",
                "kind": "cartesian_pose_delta",
                "value": [qpos0 * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "target": "gripper",
                "kind": "gripper_position",
                "value": [max(0.0, min(1.0, 1.0 - gripper_pos))],
            },
        ],
        "dt": 0.1,
    }


def main() -> None:
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    output_path = Path("tmp") / "episode_0000.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    reset_frame = robot.reset()
    records.append({"frame": em.frame_to_dict(reset_frame), "action": None})

    for _ in range(3):
        result = em.run_step(robot, action_fn=scripted_action)
        records.append(
            {
                "frame": em.frame_to_dict(result.frame),
                "action": em.action_to_dict(result.action),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")

    print("record_count:", len(records))
    print("jsonl_path:", output_path)
    print("last_native_action:", robot.last_native_action)
    print("example 3 passed.")


if __name__ == "__main__":
    main()
