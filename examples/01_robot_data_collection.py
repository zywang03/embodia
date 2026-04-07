"""Example 1: robot-only collection with your own lightweight save logic.

Run with:

    PYTHONPATH=src python examples/01_robot_data_collection.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import embodia as em


class YourRobot(em.RobotMixin):
    """A small robot that already exposes native methods and names."""

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
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    em.check_robot(robot, call_observe=False)

    records: list[dict[str, object]] = []
    reset_frame = robot.reset()
    records.append(
        {
            "frame": em.frame_to_dict(reset_frame),
            "action": None,
            "meta": {"source": "reset"},
        }
    )

    for _ in range(3):
        result = em.run_step(robot, action_fn=scripted_action)
        records.append(
            {
                "frame": em.frame_to_dict(result.frame),
                "action": em.action_to_dict(result.action),
                "meta": {"source": "scripted_action"},
            }
        )

    output_path = Path("tmp") / "robot_records.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")

    print("record_count:", len(records))
    print("first_record:", records[0])
    print("jsonl_path:", output_path)
    print("last_native_action:", robot.last_native_action)
    print("example 1 passed.")


if __name__ == "__main__":
    main()
