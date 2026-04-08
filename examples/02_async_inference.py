"""Example 2: the same ``run_step(...)`` path with a full async runtime preset.

Run with:

    PYTHONPATH=src python examples/02_async_inference.py
"""

from __future__ import annotations

import embodia as em
import numpy as np


class YourRobot(em.RobotMixin):
    """Pretend this is your original outer robot class after one small edit."""

    def __init__(self) -> None:
        self.last_native_action: object | None = None

    def YOUR_OWN_get_obs(self) -> dict[str, object]:
        return {
            "images": {"YOUR_OWN_front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            "state": {
                "YOUR_OWN_arm": np.full(6, 0.25, dtype=np.float64),
                "YOUR_OWN_gripper": np.array([0.5], dtype=np.float64),
            },
        }

    def YOUR_OWN_send_action(self, action: object) -> object:
        """Pretend the robot controller returns the final accepted action."""

        accepted = em.coerce_action(action)
        self.last_native_action = accepted
        return accepted

    def YOUR_OWN_reset(self) -> dict[str, object]:
        return self.YOUR_OWN_get_obs()


class YourPolicy(em.PolicyMixin):
    """Pretend this is your original outer policy class after one small edit."""

    def __init__(self) -> None:
        self.step_index = 0

    def YOUR_OWN_clear_state(self) -> None:
        self.step_index = 0

    def YOUR_OWN_infer(self, frame: em.Frame) -> dict[str, object]:
        targets = [0.0, 3.0, 6.0, 9.0, 12.0]
        base = targets[self.step_index % len(targets)]
        self.step_index += 1
        gripper_pos = float(frame.state["YOUR_OWN_gripper"][0])
        return {
            "YOUR_OWN_arm": {
                "command": "cartesian_pose_delta",
                "value": np.full(6, base, dtype=np.float64),
            },
            "YOUR_OWN_gripper": {
                "command": "gripper_position",
                "value": np.array(
                    [max(0.0, min(1.0, 1.0 - gripper_pos))],
                    dtype=np.float64,
                ),
            },
        }


def main() -> None:
    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    policy = YourPolicy.from_yaml("examples/basic_runtime.yml")
    runtime = em.InferenceRuntime(
        mode=em.InferenceMode.ASYNC,
        overlap_ratio=0.2,
        action_optimizers=[
            em.ActionEnsembler(current_weight=0.5),
            # This is still one runtime step per run_step() call. The
            # interpolator only changes the action emitted on that call.
            em.ActionInterpolator(steps=1),
        ],
        realtime_controller=em.RealtimeController(hz=50.0),
    )

    for step_index in range(5):
        result = em.run_step(robot, source=policy, runtime=runtime)
        print(
            "step:",
            step_index,
            "action:",
            em.action_to_dict(result.action),
            "plan_refreshed:",
            result.plan_refreshed,
            "wait:",
            f"{result.control_wait_s:.4f}",
        )

    print("native_robot_received:", robot.last_native_action)
    print("example 2 passed.")


if __name__ == "__main__":
    main()
