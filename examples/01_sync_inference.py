"""Example 1: sync inference with a full runtime preset.

Run with:

    PYTHONPATH=src python examples/01_sync_inference.py
"""

from __future__ import annotations

import time

import embodia as em


class YourRobot(em.RobotMixin):
    """Pretend this is your original outer robot class after one small edit."""

    def __init__(self) -> None:
        self.last_native_action: object | None = None
        self.step_index = 0

    def capture(self) -> dict[str, object]:
        self.step_index += 1
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"front_rgb": None},
            "state": {
                "joint_positions": [float(self.step_index)] * 6,
                "position": min(self.step_index * 0.1, 1.0),
            },
        }

    def send_command(self, action: object) -> None:
        self.last_native_action = action

    def home(self) -> dict[str, object]:
        self.step_index = 0
        return self.capture()


class YourModel(em.ModelMixin):
    """Pretend this is your original outer model class after one small edit."""

    def __init__(self) -> None:
        self.step_index = 0

    def clear_state(self) -> None:
        self.step_index = 0

    def infer(self, frame: object) -> dict[str, object]:
        qpos0 = float(frame.state["joint_positions"][0])
        gripper_pos = float(frame.state["position"])
        offset = float(self.step_index * 2)
        self.step_index += 1
        return {
            "commands": [
                {
                    "target": "arm",
                    "kind": "cartesian_pose_delta",
                    "value": [qpos0 * 0.01 + offset, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                {
                    "target": "gripper",
                    "kind": "gripper_position",
                    "value": [max(0.0, min(1.0, 1.0 - gripper_pos))],
                },
            ],
            "dt": 0.05,
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
    model = YourModel.from_yaml("examples/basic_runtime.yml")
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
        result = em.run_step(robot, model, runtime=runtime)
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
