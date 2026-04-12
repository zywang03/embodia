"""Helpers for normalizing loose structures into inferaxis dataclasses."""

from .coerce import (
    coerce_action,
    coerce_command,
    coerce_component_spec,
    coerce_frame,
    coerce_policy_output_spec,
    coerce_policy_spec,
    coerce_robot_spec,
)
from .export import (
    action_to_dict,
    component_spec_to_dict,
    command_to_dict,
    frame_to_dict,
    policy_output_spec_to_dict,
    policy_spec_to_dict,
    robot_spec_to_dict,
)

__all__ = [
    "action_to_dict",
    "component_spec_to_dict",
    "coerce_action",
    "coerce_command",
    "coerce_component_spec",
    "coerce_frame",
    "coerce_policy_output_spec",
    "coerce_policy_spec",
    "coerce_robot_spec",
    "command_to_dict",
    "frame_to_dict",
    "policy_output_spec_to_dict",
    "policy_spec_to_dict",
    "robot_spec_to_dict",
]
