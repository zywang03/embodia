"""Example 2: the same ``run_step(...)`` path with a full async runtime preset.

Run with:

    PYTHONPATH=src python examples/02_async_inference.py
"""

from __future__ import annotations

import inferaxis as infra
import numpy as np


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
    """Plain async-capable source object using the same ``infer`` entrypoint."""

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        gripper_pos = float(obs.state["YOUR_OWN_gripper"][0])

        plan: list[infra.Action] = []
        next_base = float(request.request_step * 3.0)

        while len(plan) < 2:
            plan.append(
                infra.Action(
                    commands={
                        "YOUR_OWN_arm": infra.Command(
                            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                            value=np.full(6, next_base, dtype=np.float64),
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
            )
            next_base += 3.0
        return plan


def main() -> None:
    robot = YourRobot()
    policy = YourPolicy()
    runtime = infra.InferenceRuntime(
        mode=infra.InferenceMode.ASYNC,
        profile=True,
        control_hz=50.0,
        steps_before_request=0,
        warmup_requests=0,
        profile_delay_requests=0,
        interpolation_steps=0,
        ensemble_weight=None,
        enable_rtc=False,
        latency_steps_offset=0,
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
            infra.action_to_dict(result.action),
            "plan_refreshed:",
            result.plan_refreshed,
            "wait:",
            f"{result.control_wait_s:.4f}",
        )

    runtime.close()
    print("native_robot_received:", robot.last_native_action)
    print("runtime_profile_dir:", runtime.profile_output_dir)
    print("example 2 passed.")


if __name__ == "__main__":
    main()
