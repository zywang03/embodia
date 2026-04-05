"""Public package exports for embodia."""

from .core.base import ModelBase, RobotBase
from .core.protocols import ModelProtocol, RobotProtocol
from .core.schema import Action, ActionMode, Frame, ModelSpec, RobotSpec
from .runtime.checks import (
    InterfaceValidationError,
    check_model,
    check_pair,
    check_robot,
    validate_action,
    validate_frame,
    validate_model_spec,
    validate_robot_spec,
)

__all__ = [
    "Action",
    "ActionMode",
    "Frame",
    "InterfaceValidationError",
    "ModelBase",
    "ModelProtocol",
    "ModelSpec",
    "RobotBase",
    "RobotProtocol",
    "RobotSpec",
    "check_model",
    "check_pair",
    "check_robot",
    "validate_action",
    "validate_frame",
    "validate_model_spec",
    "validate_robot_spec",
]
