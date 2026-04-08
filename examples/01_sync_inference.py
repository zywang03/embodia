"""Example 1: sync inference with a full runtime preset.

Run with:

    PYTHONPATH=src python examples/01_sync_inference.py
"""

from __future__ import annotations

import embodia as em


class YourRobot(em.RobotMixin):
    """Pretend this is your original outer robot class after one small edit."""

    def __init__(self) -> None:
        self.last_native_action: object | None = None

    def capture(self) -> dict[str, object]:
        return {
            "images": {"front_rgb": None},
            "state": {
                "joint_positions": [0.25] * 6,
                "position": 0.5,
            },
        }

    def send_command(self, action: object) -> object:
        """Pretend the robot controller returns the final accepted action."""

        accepted = em.coerce_action(action)
        self.last_native_action = accepted
        return accepted

    def home(self) -> dict[str, object]:
        return self.capture()


class YourPolicy(em.PolicyMixin):
    """Pretend this is your original outer policy class after one small edit."""

    def clear_state(self) -> None:
        return None

    def infer(self, frame: em.Frame) -> dict[str, object]:
        qpos0 = float(frame.state["joint_positions"][0])
        gripper_pos = float(frame.state["position"])
        # embodia fills sequence_id automatically when the robot does not.
        step = float(frame.sequence_id or 0)
        offset = step * 2.0
        return {
            "arm": {
                "kind": "cartesian_pose_delta",
                "value": [qpos0 * 0.01 + offset, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "gripper": {
                "kind": "gripper_position",
                "value": [max(0.0, min(1.0, 1.0 - gripper_pos))],
            },
        }


def arm_value0(action: em.Action) -> float:
    """Return the first arm dimension for compact demo printing."""

    arm = action.get_command("arm")
    assert arm is not None
    return float(arm.value[0])


def main() -> None:
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    policy = YourPolicy.from_yaml("examples/basic_runtime.yml")
    runtime = em.InferenceRuntime(
        mode=em.InferenceMode.SYNC,
        overlap_ratio=0.2,
        action_optimizers=[
            em.ActionEnsembler(current_weight=0.5),
            # This is still one runtime step per run_step() call. The
            # interpolator only changes the action emitted on that call.
            em.ActionInterpolator(steps=1),
        ],
        realtime_controller=em.RealtimeController(hz=20.0),
    )

    for step_index in range(5):
        result = em.run_step(robot, source=policy, runtime=runtime)
        print(
            f"step={step_index} "
            f"raw0={arm_value0(result.raw_action):.2f} "
            f"action0={arm_value0(result.action):.2f} "
            f"plan_refreshed={result.plan_refreshed} "
            f"wait={result.control_wait_s:.4f}"
        )

    print("native_robot_received:", robot.last_native_action)
    print("example 1 passed.")


if __name__ == "__main__":
    main()
