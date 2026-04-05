"""Example 2: adapt existing vendor classes through embodia mixins.

Original project structure sketch:

    class YourRobot:
        def capture(self): ...
        def send_command(self, action): ...
        def home(self): ...

    class YourModel:
        def clear_state(self): ...
        def infer(self, frame): ...

The actual code below shows the final result after editing those same outer
classes in place. The goal is to make the migration easy to read:

- top docstring: what the original code roughly looked like
- actual code: only the final embodia-enabled classes
- no extra ``CompatibleRobot`` / ``CompatibleModel`` shell classes

Run with:

    PYTHONPATH=src python examples/02_wrap_existing_classes.py
"""

from __future__ import annotations

import time

import embodia as em


class YourRobot(em.RobotMixin):
    """Pretend this is the existing outer robot class edited in place."""

    ROBOT_SPEC = {
        "name": "vendor_robot",
        "action_modes": ["cartesian_delta"],
        "image_keys": ["rgb_front"],
        "state_keys": ["qpos", "tcp_pose"],
    }
    METHOD_ALIASES = {
        "observe": "capture",
        "act": "send_command",
        "reset": "home",
    }
    IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
    STATE_KEY_MAP = {"qpos": "joint_positions", "tcp_pose": "ee_pose"}
    ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

    def __init__(self) -> None:
        self.last_native_action: object | None = None

    def capture(self) -> dict[str, object]:
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"rgb_front": None},
            "state": {
                "qpos": [0.0] * 6,
                "tcp_pose": [0.0] * 7,
            },
            "meta": {"source": "vendor_sdk"},
        }

    def send_command(self, action: object) -> None:
        self.last_native_action = action
        print(f"[YourRobot] native action -> {action}")

    def home(self) -> dict[str, object]:
        return self.capture()


class YourModel(em.ModelMixin):
    """Pretend this is the existing outer model class edited in place."""

    MODEL_SPEC = {
        "name": "vendor_model",
        "required_image_keys": ["rgb_front"],
        "required_state_keys": ["qpos"],
        "output_action_mode": "cartesian_delta",
    }
    METHOD_ALIASES = {
        "reset": "clear_state",
        "step": "infer",
    }
    IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
    STATE_KEY_MAP = {"qpos": "joint_positions"}
    ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}

    def clear_state(self) -> None:
        return None

    def infer(self, frame: object) -> dict[str, object]:
        assert "rgb_front" in frame.images
        assert "qpos" in frame.state
        return {
            "mode": "cartesian_delta",
            "value": [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
            "dt": 0.1,
        }


def main() -> None:
    robot = YourRobot()
    model = YourModel()

    frame = robot.observe()
    print("standardized_observation:", em.frame_to_dict(frame))

    em.check_robot(robot)
    em.check_model(model, sample_frame=frame)
    em.check_pair(robot, model)

    result = em.run_step(robot, model)
    print("standardized_action:", em.action_to_dict(result.action))
    print("native_robot_received:", robot.last_native_action)
    print("example 2 passed.")


if __name__ == "__main__":
    main()
