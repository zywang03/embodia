"""Export helpers for embodia schema objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema import (
    Action,
    ComponentSpec,
    Command,
    Frame,
    PolicyOutputSpec,
    PolicySpec,
    RobotSpec,
)
from .coerce import (
    coerce_action,
    coerce_command,
    coerce_component_spec,
    coerce_frame,
    coerce_policy_output_spec,
    coerce_policy_spec,
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
        "kind": normalized.kind,
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


def component_spec_to_dict(
    spec: ComponentSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Export a component spec-like object into a plain dictionary."""

    normalized = coerce_component_spec(spec)
    return {
        "name": normalized.name,
        "kind": normalized.kind,
        "dof": normalized.dof,
        "supported_command_kinds": list(normalized.supported_command_kinds),
        "state_keys": list(normalized.state_keys),
        "meta": dict(normalized.meta),
    }


def robot_spec_to_dict(spec: RobotSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a robot spec-like object into a plain dictionary."""

    normalized = coerce_robot_spec(spec)
    return {
        "name": normalized.name,
        "image_keys": list(normalized.image_keys),
        "components": [
            component_spec_to_dict(component)
            for component in normalized.components
        ],
        "meta": dict(normalized.meta),
    }


def policy_output_spec_to_dict(
    spec: PolicyOutputSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Export a policy-output spec-like object into a plain dictionary."""

    normalized = coerce_policy_output_spec(spec)
    return {
        "target": normalized.target,
        "command_kind": normalized.command_kind,
        "dim": normalized.dim,
        "meta": dict(normalized.meta),
    }


def policy_spec_to_dict(spec: PolicySpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a policy spec-like object into a plain dictionary."""

    normalized = coerce_policy_spec(spec)
    return {
        "name": normalized.name,
        "required_image_keys": list(normalized.required_image_keys),
        "required_state_keys": list(normalized.required_state_keys),
        "required_task_keys": list(normalized.required_task_keys),
        "outputs": [policy_output_spec_to_dict(output) for output in normalized.outputs],
        "meta": dict(normalized.meta),
    }


__all__ = [
    "action_to_dict",
    "component_spec_to_dict",
    "command_to_dict",
    "frame_to_dict",
    "policy_output_spec_to_dict",
    "policy_spec_to_dict",
    "robot_spec_to_dict",
]
