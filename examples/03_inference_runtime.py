"""Example 3: inference-time runtime features on top of edited outer classes.

Original project structure sketch:

    class YourRobot:
        def capture(self): ...
        def send_command(self, action): ...
        def home(self): ...

    class YourModel:
        def clear_state(self): ...
        def infer(self, frame): ...

Run with:

    PYTHONPATH=src python examples/03_inference_runtime.py
"""

from __future__ import annotations

import time

import embodia as em


class YourRobot(em.RobotMixin):
    """Pretend this is your original outer robot class after one small edit."""

    def __init__(self) -> None:
        self.step_index = 0
        self.last_action: em.Action | None = None

    def capture(self) -> dict[str, object]:
        self.step_index += 1
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"rgb_front": None},
            "state": {"qpos": [float(self.step_index)] * 6},
            "meta": {"step_index": self.step_index},
        }

    def send_command(self, action: em.Action) -> None:
        self.last_action = action

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
        base = float(frame.state["qpos"][0] + self.step_index)
        self.step_index += 1
        return {"mode": "cartesian_delta", "value": [base] * 6, "dt": 0.05}


def main() -> None:
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    model = YourModel.from_yaml("examples/basic_runtime.yml")

    em.check_pair(robot, model, sample_frame=robot.reset())

    runtime = em.InferenceRuntime(
        mode=em.InferenceMode.SYNC,
        action_optimizers=[em.ActionEnsembler(window_size=2)],
        realtime_controller=em.RealtimeController(hz=50.0),
    )

    for step_index in range(4):
        result = em.run_step(robot, model, runtime=runtime)
        print(
            f"step={step_index} "
            f"plan_refreshed={result.plan_refreshed} "
            f"raw={result.raw_action.value[0]:.2f} "
            f"optimized={result.action.value[0]:.2f} "
            f"wait={result.control_wait_s:.3f}"
        )

    print("last_robot_action:", em.action_to_dict(robot.last_action))
    print("example 3 passed.")


if __name__ == "__main__":
    main()
