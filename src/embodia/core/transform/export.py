"""Export helpers for embodia schema objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema import (
    Action,
    Command,
    ControlGroupSpec,
    Frame,
    ModelOutputSpec,
    ModelSpec,
    RobotSpec,
)
from .coerce import (
    coerce_action,
    coerce_command,
    coerce_control_group_spec,
    coerce_frame,
    coerce_model_output_spec,
    coerce_model_spec,
    coerce_robot_spec,
)


def frame_to_dict(frame: Frame | Mapping[str, Any]) -> dict[str, Any]:
    """Export a frame-like object into a plain dictionary."""

    normalized = coerce_frame(frame)
    return {
        "timestamp_ns": normalized.timestamp_ns,
        "images": dict(normalized.images),
        "state": dict(normalized.state),
        "task": dict(normalized.task),
        "meta": dict(normalized.meta),
        "sequence_id": normalized.sequence_id,
    }


def command_to_dict(command: Command | Mapping[str, Any]) -> dict[str, Any]:
    """Export a command-like object into a plain dictionary."""

    normalized = coerce_command(command)
    return {
        "target": normalized.target,
        "mode": normalized.mode,
        "value": list(normalized.value),
        "ref_frame": normalized.ref_frame,
        "meta": dict(normalized.meta),
    }


def action_to_dict(action: Action | Mapping[str, Any]) -> dict[str, Any]:
    """Export an action-like object into a plain dictionary."""

    normalized = coerce_action(action)
    return {
        "commands": [command_to_dict(command) for command in normalized.commands],
        "dt": normalized.dt,
        "meta": dict(normalized.meta),
    }


def control_group_spec_to_dict(
    spec: ControlGroupSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Export a control-group spec-like object into a plain dictionary."""

    normalized = coerce_control_group_spec(spec)
    return {
        "name": normalized.name,
        "kind": normalized.kind,
        "dof": normalized.dof,
        "action_modes": list(normalized.action_modes),
        "state_keys": list(normalized.state_keys),
        "meta": dict(normalized.meta),
    }


def robot_spec_to_dict(spec: RobotSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a robot spec-like object into a plain dictionary."""

    normalized = coerce_robot_spec(spec)
    return {
        "name": normalized.name,
        "image_keys": list(normalized.image_keys),
        "groups": [
            control_group_spec_to_dict(group) for group in normalized.groups
        ],
        "task_keys": list(normalized.task_keys),
        "meta": dict(normalized.meta),
    }


def model_output_spec_to_dict(
    spec: ModelOutputSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Export a model-output spec-like object into a plain dictionary."""

    normalized = coerce_model_output_spec(spec)
    return {
        "target": normalized.target,
        "mode": normalized.mode,
        "dim": normalized.dim,
        "meta": dict(normalized.meta),
    }


def model_spec_to_dict(spec: ModelSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a model spec-like object into a plain dictionary."""

    normalized = coerce_model_spec(spec)
    return {
        "name": normalized.name,
        "required_image_keys": list(normalized.required_image_keys),
        "required_state_keys": list(normalized.required_state_keys),
        "required_task_keys": list(normalized.required_task_keys),
        "outputs": [model_output_spec_to_dict(output) for output in normalized.outputs],
        "meta": dict(normalized.meta),
    }


__all__ = [
    "action_to_dict",
    "command_to_dict",
    "control_group_spec_to_dict",
    "frame_to_dict",
    "model_output_spec_to_dict",
    "model_spec_to_dict",
    "robot_spec_to_dict",
]
