"""Example 3: robot-only data collection with plain Python save logic.

Run with:

    PYTHONPATH=src python examples/03_data_collection.py
"""

from __future__ import annotations

import json
from pathlib import Path

import inferaxis as infra
import numpy as np


class YourRobot:
    """Plain local executor used by the collection loop."""

    def __init__(self) -> None:
        self.last_native_action: object | None = None

    def get_obs(self) -> infra.Frame:
        return infra.Frame(
            images={"YOUR_OWN_front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={
                "YOUR_OWN_arm": np.full(6, 0.25, dtype=np.float64),
                "YOUR_OWN_gripper": np.array([0.5], dtype=np.float64),
            },
        )

    def send_action(self, action: infra.Action) -> infra.Action:
        """Pretend the robot controller returns the final accepted action."""

        self.last_native_action = action
        return action

    def reset(self) -> infra.Frame:
        return self.get_obs()


class DemoTeleop:
    """Tiny stand-in for teleop input or an external data-collection source."""

    def __init__(self) -> None:
        self.cursor = 0

    script = [
        infra.Action(
            commands={
                "YOUR_OWN_arm": infra.Command(
                    command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                    value=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                ),
                "YOUR_OWN_gripper": infra.Command(
                    command=infra.BuiltinCommandKind.GRIPPER_POSITION,
                    value=np.array([1.0], dtype=np.float64),
                ),
            }
        ),
        infra.Action(
            commands={
                "YOUR_OWN_arm": infra.Command(
                    command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                    value=np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                ),
                "YOUR_OWN_gripper": infra.Command(
                    command=infra.BuiltinCommandKind.GRIPPER_POSITION,
                    value=np.array([0.5], dtype=np.float64),
                ),
            }
        ),
        infra.Action(
            commands={
                "YOUR_OWN_arm": infra.Command(
                    command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                    value=np.array([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                ),
                "YOUR_OWN_gripper": infra.Command(
                    command=infra.BuiltinCommandKind.GRIPPER_POSITION,
                    value=np.array([0.0], dtype=np.float64),
                ),
            }
        ),
    ]

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> infra.Action:
        """Return one scripted action and ignore chunk request metadata."""

        del obs, request
        index = min(self.cursor, len(self.script) - 1)
        self.cursor += 1
        return self.script[index]


def main() -> None:
    robot = YourRobot()
    teleop = DemoTeleop()
    output_path = Path("tmp") / "episode_0000.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    reset_frame = robot.reset()
    records.append({"frame": infra.frame_to_dict(reset_frame), "action": None})

    for _ in range(3):
        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=teleop.infer,
        )
        records.append(
            {
                "frame": infra.frame_to_dict(result.frame),
                "action": infra.action_to_dict(result.action),
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
