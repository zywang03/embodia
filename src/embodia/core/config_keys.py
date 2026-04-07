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
    IMAGE_KEYS = "image_keys"
    COMPONENTS = "components"
    META = "meta"


class PolicySpecKey(StrEnum):
    """Keys used inside ``PolicyMixin.POLICY_SPEC`` mappings."""

    NAME = "name"
    REQUIRED_IMAGE_KEYS = "required_image_keys"
    REQUIRED_STATE_KEYS = "required_state_keys"
    REQUIRED_TASK_KEYS = "required_task_keys"
    OUTPUTS = "outputs"
    META = "meta"


class MethodAliasKey(StrEnum):
    """Keys used inside ``METHOD_ALIASES`` mappings."""

    GET_SPEC = "get_spec"
    OBSERVE = "observe"
    ACT = "act"
    RESET = "reset"
    INFER = "infer"
    INFER_CHUNK = "infer_chunk"
    PLAN = "plan"


__all__ = ["MethodAliasKey", "PolicySpecKey", "RobotSpecKey"]
