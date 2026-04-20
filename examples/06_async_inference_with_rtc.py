"""Example 6: async inference with RTC-aware request hints.

Run with:

    PYTHONPATH=src python examples/06_async_inference_with_rtc.py
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


class YourRtcPolicy:
    """Async-capable source that reads RTC hints directly from ``request``."""

    def __init__(self) -> None:
        self.last_rtc_summary: dict[str, int | bool | None] | None = None

    def reset(self) -> None:
        self.last_rtc_summary = None

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        gripper_pos = float(obs.state["YOUR_OWN_gripper"][0])
        prev_action_chunk = request.prev_action_chunk
        inference_delay = request.inference_delay
        execute_horizon = request.execute_horizon
        self.last_rtc_summary = {
            "has_rtc_args": request.rtc_args is not None,
            "prev_action_chunk_len": 0 if prev_action_chunk is None else len(prev_action_chunk),
            "inference_delay": inference_delay,
            "execute_horizon": execute_horizon,
        }

        plan: list[infra.Action] = []
        if prev_action_chunk:
            last_arm = prev_action_chunk[-1].get_command("YOUR_OWN_arm")
            assert last_arm is not None
            next_base = float(last_arm.value[0] + 3.0)
        else:
            # The very first bootstrap request still has no RTC hints, so
            # derive the initial chunk from request metadata rather than
            # mutable call-count state. Later warmup/profile requests already
            # exercise prev_action_chunk.
            next_base = float(request.request_step * 3.0)

        while len(plan) < 4:
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
    policy = YourRtcPolicy()
    runtime = infra.InferenceRuntime(
        mode=infra.InferenceMode.ASYNC,
        overlap_ratio=0.1,
        warmup_requests=1,
        profile_delay_requests=3,
        interpolation_steps=2,
        enable_mismatch_bridge=True,
        # These are execution-only smoothing controls. RTC hints still stay in
        # raw chunk units and are passed to the policy unchanged.
        # Async startup first warms up a few requests, then profiles delay on a
        # few more requests before the first action is sent to the robot. The
        # first run_step() call triggers that bootstrap automatically.
        enable_rtc=True,
        realtime_controller=infra.RealtimeController(hz=50.0),
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
            "rtc:",
            policy.last_rtc_summary,
        )

    print("native_robot_received:", robot.last_native_action)
    print("example 6 passed.")


if __name__ == "__main__":
    main()
