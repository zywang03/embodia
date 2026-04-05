"""Core interface building blocks for embodia."""

from .errors import InterfaceValidationError
from .mixins import ModelMixin, RobotMixin
from .protocols import ModelProtocol, RobotProtocol
from .schema import Action, ActionMode, Frame, ModelSpec, RobotSpec
from .transform import (
    action_to_dict,
    coerce_action,
    coerce_frame,
    coerce_model_spec,
    coerce_robot_spec,
    frame_to_dict,
    invert_mapping,
    model_spec_to_dict,
    remap_action,
    remap_frame,
    remap_mapping_keys,
    remap_model_spec,
    remap_robot_spec,
    robot_spec_to_dict,
)

__all__ = [
    "Action",
    "ActionMode",
    "Frame",
    "InterfaceValidationError",
    "ModelMixin",
    "ModelProtocol",
    "ModelSpec",
    "RobotMixin",
    "RobotProtocol",
    "RobotSpec",
    "action_to_dict",
    "coerce_action",
    "coerce_frame",
    "coerce_model_spec",
    "coerce_robot_spec",
    "frame_to_dict",
    "invert_mapping",
    "model_spec_to_dict",
    "remap_action",
    "remap_frame",
    "remap_mapping_keys",
    "remap_model_spec",
    "remap_robot_spec",
    "robot_spec_to_dict",
]
