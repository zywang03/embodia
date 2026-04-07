"""Example 4: one possible LeRobot-style export built on plain run_step() data.

Run with:

    PYTHONPATH=src python examples/04_lerobot_bridge.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import embodia as em


class YourRobot(em.RobotMixin):
    """Pretend this is your original outer robot class after one small edit."""

    def __init__(self) -> None:
        self.step_count = 0
        self.last_native_action: object | None = None

    def capture(self) -> dict[str, object]:
        self.step_count += 1
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"rgb_front": None},
            "state": {"qpos": [float(self.step_count)] * 6},
        }

    def send_command(self, action: object) -> None:
        self.last_native_action = action

    def home(self) -> dict[str, object]:
        self.step_count = 0
        return self.capture()


def scripted_action(frame: em.Frame) -> dict[str, object]:
    """Pretend this action came from teleop or a scripted expert."""

    qpos0 = float(frame.state["joint_positions"][0])
    return {"mode": "ee_delta", "value": [qpos0 * 0.01] * 6, "dt": 0.1}


def to_lerobot_record(
    result: em.StepResult,
    *,
    episode_index: int,
    frame_index: int,
    is_last: bool,
) -> dict[str, Any]:
    """Convert one embodia step result into one LeRobot-style row."""

    frame = em.frame_to_dict(result.frame)
    return {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp": result.frame.timestamp_ns / 1_000_000_000.0,
        "timestamp_ns": result.frame.timestamp_ns,
        "next.done": is_last,
        "observation.images": dict(frame["images"]),
        "observation.state": dict(frame["state"]),
        "task": frame["task"],
        "action": em.action_to_dict(result.action),
        "embodia.meta": {"frame_meta": frame["meta"]},
    }


def main() -> None:
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    results = [em.run_step(robot, action_fn=scripted_action) for _ in range(3)]

    records = [
        to_lerobot_record(
            result,
            episode_index=7,
            frame_index=index,
            is_last=index == len(results) - 1,
        )
        for index, result in enumerate(results)
    ]
    output_path = Path("tmp") / "episode_0007.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")

    print("first_record:", records[0])
    print("jsonl_path:", output_path)
    print("example 4 passed.")


if __name__ == "__main__":
    main()
