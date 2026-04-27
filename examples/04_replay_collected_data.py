"""Example 4: replay collected actions back into the robot.

Run with:

    PYTHONPATH=src python examples/04_replay_collected_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import inferaxis as infra
import numpy as np
from inferaxis.core.transform import (
    action_to_dict,
    coerce_action,
    coerce_frame,
    frame_to_dict,
)


class YourRobot:
    """Plain local executor used by the replay loop."""

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
    """Tiny stand-in for recorded operator input when bootstrapping the demo."""

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
    episode_path = Path("tmp") / "episode_0000.jsonl"
    if not episode_path.exists():
        teleop = DemoTeleop()
        records: list[dict[str, object]] = [
            {"frame": frame_to_dict(robot.reset()), "action": None}
        ]
        for _ in range(3):
            result = infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=teleop.infer,
            )
            records.append(
                {
                    "frame": frame_to_dict(result.frame),
                    "action": action_to_dict(result.action),
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
                act_fn=robot.send_action,
                frame=coerce_frame(frame),
                act_src_fn=lambda _frame, _request, recorded_action=action: (
                    coerce_action(recorded_action)
                ),
            )
            replayed += 1

    print("episode_path:", episode_path)
    print("replayed_actions:", replayed)
    print("last_native_action:", robot.last_native_action)
    print("example 4 passed.")


if __name__ == "__main__":
    main()
