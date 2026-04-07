"""Core interface building blocks for embodia."""

from .config_keys import MethodAliasKey, ModelSpecKey, RobotSpecKey
from .config_io import load_component_yaml_config
from .errors import InterfaceValidationError
from .mixins import ModelMixin, RobotMixin
from .modalities import ACTION_MODES, IMAGE_KEYS, STATE_KEYS, ModalityToken
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
    "ACTION_MODES",
    "Frame",
    "IMAGE_KEYS",
    "InterfaceValidationError",
    "MethodAliasKey",
    "ModelMixin",
    "ModelProtocol",
    "ModelSpecKey",
    "ModelSpec",
    "RobotMixin",
    "RobotProtocol",
    "RobotSpecKey",
    "RobotSpec",
    "ModalityToken",
    "STATE_KEYS",
    "action_to_dict",
    "coerce_action",
    "coerce_frame",
    "coerce_model_spec",
    "coerce_robot_spec",
    "frame_to_dict",
    "invert_mapping",
    "load_component_yaml_config",
    "model_spec_to_dict",
    "remap_action",
    "remap_frame",
    "remap_mapping_keys",
    "remap_model_spec",
    "remap_robot_spec",
    "robot_spec_to_dict",
]
