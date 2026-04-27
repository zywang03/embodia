"""Example 1: sync inference with a full runtime preset.

Run with:

    PYTHONPATH=src python examples/01_sync_inference.py
"""

from __future__ import annotations

import inferaxis as infra
import numpy as np
from inferaxis.core.transform import action_to_dict


class YourRobot:
    """Plain local executor used by the runtime loop."""

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


class YourPolicy:
    """Plain policy object exposing one action source for sync execution."""

    def __init__(self) -> None:
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> infra.Action:
        del request
        qpos0 = float(obs.state["YOUR_OWN_arm"][0])
        gripper_pos = float(obs.state["YOUR_OWN_gripper"][0])
        step = float(self.step_index)
        self.step_index += 1
        offset = step * 2.0
        return infra.Action(
            commands={
                "YOUR_OWN_arm": infra.Command(
                    command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                    value=np.array(
                        [qpos0 * 0.01 + offset, 0.0, 0.0, 0.0, 0.0, 0.0],
                        dtype=np.float64,
                    ),
                ),
                "YOUR_OWN_gripper": infra.Command(
                    command=infra.BuiltinCommandKind.GRIPPER_POSITION,
                    value=np.array(
                        [max(0.0, min(1.0, 1.0 - gripper_pos))],
                        dtype=np.float64,
                    ),
                ),
            }
        )


def main() -> None:
    robot = YourRobot()
    policy = YourPolicy()

    runtime = infra.InferenceRuntime(
        mode=infra.InferenceMode.SYNC,
        control_hz=20.0,
    )

    for step_index in range(5):
        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        print(
            "step:",
            step_index,
            "action:",
            action_to_dict(result.action),
            "plan_refreshed:",
            result.plan_refreshed,
            "wait:",
            f"{result.control_wait_s:.4f}",
        )

    print("native_robot_received:", robot.last_native_action)
    print("example 1 passed.")


if __name__ == "__main__":
    main()
