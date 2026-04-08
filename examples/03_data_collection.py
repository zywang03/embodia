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
            "images": {"front_rgb": None},
            "state": {
                "joint_positions": [float(self.step_index)] * 6,
                "position": min(self.step_index * 0.1, 1.0),
            },
            "meta": {"step_index": self.step_index},
        }

    def send_command(self, action: object) -> object:
        """Pretend the robot controller returns the final accepted action."""

        accepted = em.coerce_action(action)
        self.last_native_action = accepted
        return accepted

    def home(self) -> dict[str, object]:
        self.step_index = 0
        return self.capture()


class DemoTeleop:
    """Tiny stand-in for teleop input or an external data-collection source."""

    def __init__(self) -> None:
        self.step_index = 0
        self.script = [
            {
                "arm": {
                    "kind": "cartesian_pose_delta",
                    "value": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "gripper": {
                    "kind": "gripper_position",
                    "value": [1.0],
                },
            },
            {
                "arm": {
                    "kind": "cartesian_pose_delta",
                    "value": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "gripper": {
                    "kind": "gripper_position",
                    "value": [0.5],
                },
            },
            {
                "arm": {
                    "kind": "cartesian_pose_delta",
                    "value": [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "gripper": {
                    "kind": "gripper_position",
                    "value": [0.0],
                },
            },
        ]

    def next_action(self, frame: em.Frame) -> dict[str, object]:
        """Return the next operator-provided action, independent of robot state."""

        del frame
        action = self.script[min(self.step_index, len(self.script) - 1)]
        self.step_index += 1
        return action


def main() -> None:
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    teleop = DemoTeleop()
    output_path = Path("tmp") / "episode_0000.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    reset_frame = robot.reset()
    records.append({"frame": em.frame_to_dict(reset_frame), "action": None})

    for _ in range(3):
        result = em.run_step(robot, action_fn=teleop.next_action)
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
