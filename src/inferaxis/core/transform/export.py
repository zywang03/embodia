"""Export helpers for inferaxis schema objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..arraylike import to_python_value
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
        "images": {
            key: to_python_value(value) for key, value in normalized.images.items()
        },
        "state": {
            key: to_python_value(value) for key, value in normalized.state.items()
        },
        "task": to_python_value(normalized.task),
        "meta": to_python_value(normalized.meta),
        "sequence_id": normalized.sequence_id,
    }


def command_to_dict(
    command: Command | Mapping[str, Any],
    *,
    compact: bool = True,
) -> dict[str, Any]:
    """Export a command-like object into a plain dictionary.

    The owning target is intentionally not included here. In inferaxis's current
    schema, the target lives on ``Action.commands`` as the dictionary key.

    When ``compact=True`` (the default), optional empty/default fields are
    omitted so action JSON stays easier to read:

    - omit ``ref_frame`` when it is ``None``
    - omit ``meta`` when it is empty
    """

    normalized = coerce_command(command)
    exported: dict[str, Any] = {
        "command": str(normalized.command),
        "value": normalized.value.tolist(),
    }
    if not compact or normalized.ref_frame is not None:
        exported["ref_frame"] = normalized.ref_frame
    if not compact or normalized.meta:
        exported["meta"] = to_python_value(normalized.meta)
    return exported


def action_to_dict(
    action: Action | Mapping[str, Any],
    *,
    compact: bool = True,
    commands_as_mapping: bool = True,
) -> dict[str, Any]:
    """Export an action-like object into a plain dictionary.

    When ``compact=True`` (the default), optional empty/default fields are
    omitted:

    - omit ``meta`` when it is empty
    - apply the same compact export rules to each command

    When ``commands_as_mapping=True`` (the default), compact actions without
    action-level metadata export directly as:

    ``{"arm": {"command": "...", "value": [...]}}``

    If action-level metadata is present, the wrapped form is used instead:

    ``{"commands": {"arm": {...}}, "meta": {...}}``

    This keeps JSON smaller by using the component name as the key instead of
    repeating ``target`` inside each command.
    """

    normalized = coerce_action(action)
    if commands_as_mapping:
        commands_export: dict[str, Any] = {
            target: command_to_dict(
                command,
                compact=compact,
            )
            for target, command in normalized.commands.items()
        }
        if compact and not normalized.meta:
            return commands_export
    else:
        commands_export = [
            {
                "target": target,
                **command_to_dict(
                    command,
                    compact=compact,
                ),
            }
            for target, command in normalized.commands.items()
        ]

    exported = {"commands": commands_export}
    if not compact or normalized.meta:
        exported["meta"] = to_python_value(normalized.meta)
    return exported


def component_spec_to_dict(
    spec: ComponentSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Export a component spec-like object into a plain dictionary."""

    normalized = coerce_component_spec(spec)
    return {
        "name": normalized.name,
        "type": normalized.type,
        "dof": normalized.dof,
        "command": [str(command) for command in normalized.command],
        "meta": dict(normalized.meta),
    }


def robot_spec_to_dict(spec: RobotSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a robot spec-like object into a plain dictionary."""

    normalized = coerce_robot_spec(spec)
    return {
        "name": normalized.name,
        "image_keys": list(normalized.image_keys),
        "components": [
            component_spec_to_dict(component) for component in normalized.components
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
        "command": str(normalized.command),
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
        "outputs": [
            policy_output_spec_to_dict(output) for output in normalized.outputs
        ],
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
