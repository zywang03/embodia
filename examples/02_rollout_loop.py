"""Example 2: run a short standardized rollout from edited outer classes.

Original project structure sketch:

    class YourRobot:
        def capture(self): ...
        def send_command(self, action): ...
        def home(self): ...

    class YourModel:
        def clear_state(self): ...
        def infer(self, frame): ...

Run with:

    PYTHONPATH=src python examples/02_rollout_loop.py
"""

from __future__ import annotations

import time

import embodia as em


class YourRobot(em.RobotMixin):
    """Pretend this is your existing outer robot class after one small edit."""

    def __init__(self) -> None:
        self.step_count = 0
        self.last_native_action: object | None = None

    def capture(self) -> dict[str, object]:
        self.step_count += 1
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"rgb_front": None},
            "state": {"qpos": [float(self.step_count)] * 6},
            "meta": {"step_count": self.step_count},
        }

    def send_command(self, action: object) -> None:
        self.last_native_action = action

    def home(self) -> dict[str, object]:
        self.step_count = 0
        return self.capture()


class YourModel(em.ModelMixin):
    """Pretend this is your existing outer model class after one small edit."""

    def clear_state(self) -> None:
        return None

    def infer(self, frame: object) -> dict[str, object]:
        qpos = frame.state["qpos"]
        return {
            "mode": "cartesian_delta",
            "value": [qpos[0] * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dt": 0.1,
        }


def main() -> None:
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    model = YourModel.from_yaml("examples/basic_runtime.yml")
    em.check_pair(robot, model, sample_frame=robot.reset())

    records: list[dict[str, object]] = []
    for _ in range(3):
        result = em.run_step(robot, model)
        records.append(
            {
                "frame": em.frame_to_dict(result.frame),
                "action": em.action_to_dict(result.action),
            }
        )

    print("record_count:", len(records))
    print("first_record:", records[0])
    print("last_native_action:", robot.last_native_action)
    print("example 2 passed.")


if __name__ == "__main__":
    main()
