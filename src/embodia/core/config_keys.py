"""Public declarative config keys for embodia mixin configuration.

These keys are optional convenience helpers. Plain strings remain fully
supported, but these ``StrEnum`` values make the intent of declarative mixin
configuration easier to read in user code.
"""

from __future__ import annotations

from enum import StrEnum


class RobotSpecKey(StrEnum):
    """Keys used inside ``RobotMixin.ROBOT_SPEC`` mappings."""

    NAME = "name"
    ACTION_MODES = "action_modes"
    IMAGE_KEYS = "image_keys"
    STATE_KEYS = "state_keys"


class ModelSpecKey(StrEnum):
    """Keys used inside ``ModelMixin.MODEL_SPEC`` mappings."""

    NAME = "name"
    REQUIRED_IMAGE_KEYS = "required_image_keys"
    REQUIRED_STATE_KEYS = "required_state_keys"
    OUTPUT_ACTION_MODE = "output_action_mode"


class MethodAliasKey(StrEnum):
    """Keys used inside ``METHOD_ALIASES`` mappings."""

    GET_SPEC = "get_spec"
    OBSERVE = "observe"
    ACT = "act"
    RESET = "reset"
    STEP = "step"


__all__ = ["MethodAliasKey", "ModelSpecKey", "RobotSpecKey"]
