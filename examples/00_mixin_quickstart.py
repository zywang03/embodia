"""Example 0: the recommended low-intrusion embodia integration path.

Original project structure sketch:

    class YourRobot:
        def capture(self): ...
        def send_command(self, action): ...
        def home(self): ...

    class YourModel:
        def clear_state(self): ...
        def infer(self, frame): ...

This file only keeps the final, embodia-enabled version of those same outer
classes. In other words, imagine the original classes above were edited in
place by:

1. adding ``em.RobotMixin`` / ``em.ModelMixin`` to the class header
2. adding declarative embodia class attributes
3. keeping the original native methods like ``capture()`` and ``infer()``

Run with:

    PYTHONPATH=src python examples/00_mixin_quickstart.py
"""

from __future__ import annotations

import time

import embodia as em


class YourRobot(em.RobotMixin):
    """Pretend this is the original outer robot class after one small edit."""

    ROBOT_SPEC = {
        "name": "vendor_robot",
        "action_modes": ["cartesian_delta"],
        "image_keys": ["rgb_front"],
        "state_keys": ["qpos"],
    }
    METHOD_ALIASES = {
        "observe": "capture",
        "act": "send_command",
        "reset": "home",
    }
    MODALITY_MAPS = {
        em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
        em.STATE_KEYS: {"qpos": "joint_positions"},
        em.ACTION_MODES: {"cartesian_delta": "ee_delta"},
    }

    def __init__(self) -> None:
        self.last_native_action = None

    def capture(self) -> dict[str, object]:
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"rgb_front": None},
            "state": {"qpos": [0.0] * 6},
        }

    def send_command(self, action: object) -> None:
        self.last_native_action = action
        print(f"[YourRobot] native action -> {action}")

    def home(self) -> dict[str, object]:
        return self.capture()


class YourModel(em.ModelMixin):
    """Pretend this is the original outer model class after one small edit."""

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
    MODALITY_MAPS = {
        em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
        em.STATE_KEYS: {"qpos": "joint_positions"},
        em.ACTION_MODES: {"cartesian_delta": "ee_delta"},
    }

    def clear_state(self) -> None:
        return None

    def infer(self, frame: object) -> dict[str, object]:
        assert "rgb_front" in frame.images
        assert "qpos" in frame.state
        return {"mode": "cartesian_delta", "value": [0.0] * 6}


def main() -> None:
    robot = YourRobot()
    model = YourModel()

    em.check_pair(robot, model, sample_frame=robot.reset())
    result = em.run_step(robot, model)

    print("standardized_frame:", em.frame_to_dict(result.frame))
    print("standardized_action:", em.action_to_dict(result.action))
    print("native_robot_received:", robot.last_native_action)
    print("example 0 passed.")


if __name__ == "__main__":
    main()
