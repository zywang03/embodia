"""Helpers for normalizing loose structures into embodia dataclasses."""

from .coerce import (
    coerce_action,
    coerce_command,
    coerce_component_spec,
    coerce_frame,
    coerce_model_output_spec,
    coerce_model_spec,
    coerce_robot_spec,
)
from .export import (
    action_to_dict,
    component_spec_to_dict,
    command_to_dict,
    frame_to_dict,
    model_output_spec_to_dict,
    model_spec_to_dict,
    robot_spec_to_dict,
)
from .remap import (
    invert_mapping,
    remap_action,
    remap_command,
    remap_component_spec,
    remap_frame,
    remap_mapping_keys,
    remap_model_output_spec,
    remap_model_spec,
    remap_robot_spec,
)

__all__ = [
    "action_to_dict",
    "component_spec_to_dict",
    "coerce_action",
    "coerce_command",
    "coerce_component_spec",
    "coerce_frame",
    "coerce_model_output_spec",
    "coerce_model_spec",
    "coerce_robot_spec",
    "command_to_dict",
    "frame_to_dict",
    "invert_mapping",
    "model_output_spec_to_dict",
    "model_spec_to_dict",
    "remap_action",
    "remap_command",
    "remap_component_spec",
    "remap_frame",
    "remap_mapping_keys",
    "remap_model_output_spec",
    "remap_model_spec",
    "remap_robot_spec",
    "robot_spec_to_dict",
]
