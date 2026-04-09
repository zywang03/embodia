"""Coercion helpers for inferaxis schema objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np

from ..arraylike import to_numpy_array
from ..errors import InterfaceValidationError
from ..schema import (
    Action,
    ComponentSpec,
    Command,
    Frame,
    PolicyOutputSpec,
    PolicySpec,
    RobotSpec,
)


def _copy_string_key_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> dict[str, Any]:
    """Convert a mapping into a plain ``dict[str, Any]``."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"{field_name} must be a mapping with string keys, "
            f"got {type(value).__name__}."
        )

    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise InterfaceValidationError(
                f"{field_name} keys must be strings, got {key!r}."
            )
        result[key] = item
    return result


def _copy_sequence(value: object, field_name: str) -> list[Any]:
    """Convert a generic sequence into a plain list."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InterfaceValidationError(
            f"{field_name} must be a sequence, got {type(value).__name__}."
        )
    return list(value)


def _copy_array_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
    *,
    wrap_scalar: bool,
) -> dict[str, np.ndarray]:
    """Convert a mapping into ``dict[str, numpy.ndarray]``."""

    mapping = _copy_string_key_mapping(value, field_name)
    return {
        key: to_numpy_array(
            item,
            field_name=f"{field_name}[{key!r}]",
            wrap_scalar=wrap_scalar,
            numeric_only=True,
            allow_bool=True,
            copy=True,
        )
        for key, item in mapping.items()
    }


def _coerce_action_commands(
    value: object,
    field_name: str,
) -> dict[str, Command]:
    """Normalize command payloads from a compact or wrapped mapping.

    Supported mapping form:

    ``{"arm": {"command": "...", "value": [...]}, "gripper": {...}}``

    The mapping key is the command target. If a nested mapping also provides
    ``target``, it must match the outer key.
    """

    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"{field_name} must be a mapping from target name to command payload, "
            f"got {type(value).__name__}."
        )

    commands: dict[str, Command] = {}
    for target, item in value.items():
        if not isinstance(target, str):
            raise InterfaceValidationError(
                f"{field_name} mapping keys must be strings, got {target!r}."
            )
        if isinstance(item, Command):
            commands[target] = coerce_command(item)
            continue
        if not isinstance(item, Mapping):
            raise InterfaceValidationError(
                f"{field_name}[{target!r}] must be a command mapping, got "
                f"{type(item).__name__}."
            )
        if "target" in item and item["target"] != target:
            raise InterfaceValidationError(
                f"{field_name}[{target!r}] target mismatch: outer key is "
                f"{target!r} but nested target is {item['target']!r}."
            )
        commands[target] = coerce_command(
            {
                "command": item.get("command"),
                "value": item.get("value"),
                "ref_frame": item.get("ref_frame"),
                "meta": item.get("meta"),
            }
        )
    return commands


def _coerce_command_value(value: object, field_name: str) -> np.ndarray:
    """Convert one command payload into a 1D float64 ndarray."""

    array = to_numpy_array(
        value,
        field_name=field_name,
        wrap_scalar=True,
        numeric_only=True,
        allow_bool=False,
        copy=True,
        dtype=np.float64,
    )
    if array.ndim != 1:
        raise InterfaceValidationError(
            f"{field_name} must be a 1D numeric vector, got ndim={array.ndim}."
        )
    return array


def coerce_frame(value: Frame | Mapping[str, Any]) -> Frame:
    """Normalize a ``Frame`` or mapping into a standard :class:`Frame`.

    For frame-like mappings, ``images`` and ``state`` are the only required
    payload fields. inferaxis ignores user-provided ``timestamp_ns`` and
    ``sequence_id`` at the mapping boundary and manages them internally.
    """

    if isinstance(value, Frame):
        copied = Frame(
            images={key: item.copy() for key, item in value.images.items()},
            state={key: item.copy() for key, item in value.state.items()},
            task=dict(value.task),
            meta=dict(value.meta),
        )
        copied.timestamp_ns = value.timestamp_ns
        copied.sequence_id = value.sequence_id
        return copied
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"frame must be a Frame or mapping, got {type(value).__name__}."
        )

    try:
        images = value["images"]
        state = value["state"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"frame mapping is missing required field {exc.args[0]!r}."
        ) from exc
    return Frame(
        images=_copy_array_mapping(images, "frame.images", wrap_scalar=False),
        state=_copy_array_mapping(state, "frame.state", wrap_scalar=True),
        task=_copy_string_key_mapping(value.get("task"), "frame.task"),
        meta=_copy_string_key_mapping(value.get("meta"), "frame.meta"),
    )


def coerce_command(value: Command | Mapping[str, Any]) -> Command:
    """Normalize a ``Command`` or mapping into a standard :class:`Command`.

    A standalone command payload does not carry its owning target. The target
    belongs to ``Action.commands`` when the command is part of one action.
    """

    if isinstance(value, Command):
        return Command(
            command=value.command,
            value=value.value.copy(),
            ref_frame=value.ref_frame,
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"command must be a Command or mapping, got {type(value).__name__}."
        )

    try:
        command_value = value["value"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"command mapping is missing required field {exc.args[0]!r}."
        ) from exc

    command = value.get("command")
    if command is None:
        raise InterfaceValidationError(
            "command mapping is missing required field 'command'."
        )

    return Command(
        command=command,
        value=_coerce_command_value(command_value, "command.value"),
        ref_frame=value.get("ref_frame"),
        meta=_copy_string_key_mapping(value.get("meta"), "command.meta"),
    )


def coerce_action(value: Action | Mapping[str, Any]) -> Action:
    """Normalize an ``Action`` or mapping into a standard :class:`Action`.

    Supported mapping forms are:

    1. wrapped:
       ``{"commands": {"arm": {"command": "...", "value": [...]}}}``
    2. compact:
       ``{"arm": {"command": "...", "value": [...]}}``

    The compact form keeps JSON smaller for the common case where action-level
    metadata is empty.
    """

    if isinstance(value, Action):
        if not isinstance(value.commands, Mapping):
            raise InterfaceValidationError(
                "Action.commands must be a mapping from target to Command, got "
                f"{type(value.commands).__name__}."
            )
        return Action(
            commands={
                target: coerce_command(command)
                for target, command in value.commands.items()
            },
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"action must be an Action or mapping, got {type(value).__name__}."
        )

    if "commands" in value:
        return Action(
            commands=_coerce_action_commands(value["commands"], "action.commands"),
            meta=_copy_string_key_mapping(value.get("meta"), "action.meta"),
        )

    if "meta" in value:
        raise InterfaceValidationError(
            "action mapping with top-level 'meta' must use the wrapped form "
            "{'commands': ..., 'meta': ...}."
        )

    return Action(
        commands=_coerce_action_commands(value, "action"),
        meta={},
    )


def coerce_component_spec(
    value: ComponentSpec | Mapping[str, Any],
) -> ComponentSpec:
    """Normalize a component spec-like value."""

    if isinstance(value, ComponentSpec):
        return ComponentSpec(
            name=value.name,
            type=value.type,
            dof=value.dof,
            command=list(value.command),
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            "component spec must be ComponentSpec or mapping, got "
            f"{type(value).__name__}."
        )

    try:
        name = value["name"]
        component_type = value["type"]
        dof = value["dof"]
        command = value.get("command")
    except KeyError as exc:
        raise InterfaceValidationError(
            f"component spec mapping is missing required field {exc.args[0]!r}."
        ) from exc
    if command is None:
        raise InterfaceValidationError(
            "component spec mapping is missing required field 'command'."
        )

    return ComponentSpec(
        name=name,
        type=component_type,
        dof=dof,
        command=_copy_sequence(
            command,
            "component_spec.command",
        ),
        meta=_copy_string_key_mapping(value.get("meta"), "component_spec.meta"),
    )


def coerce_robot_spec(value: RobotSpec | Mapping[str, Any]) -> RobotSpec:
    """Normalize a ``RobotSpec`` or mapping into a standard :class:`RobotSpec`."""

    if isinstance(value, RobotSpec):
        return RobotSpec(
            name=value.name,
            image_keys=list(value.image_keys),
            components=[
                coerce_component_spec(component) for component in value.components
            ],
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"robot spec must be a RobotSpec or mapping, got {type(value).__name__}."
        )

    try:
        name = value["name"]
        image_keys = value["image_keys"]
        components = value["components"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"robot spec mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return RobotSpec(
        name=name,
        image_keys=_copy_sequence(image_keys, "robot_spec.image_keys"),
        components=[
            coerce_component_spec(item)
            for item in _copy_sequence(components, "robot_spec.components")
        ],
        meta=_copy_string_key_mapping(value.get("meta"), "robot_spec.meta"),
    )


def coerce_policy_output_spec(
    value: PolicyOutputSpec | Mapping[str, Any],
) -> PolicyOutputSpec:
    """Normalize a policy-output spec-like value."""

    if isinstance(value, PolicyOutputSpec):
        return PolicyOutputSpec(
            target=value.target,
            command=value.command,
            dim=value.dim,
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            "policy output spec must be PolicyOutputSpec or mapping, got "
            f"{type(value).__name__}."
        )

    try:
        target = value["target"]
        dim = value["dim"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"policy output spec mapping is missing required field {exc.args[0]!r}."
        ) from exc
    command = value.get("command")
    if command is None:
        raise InterfaceValidationError(
            "policy output spec mapping is missing required field 'command'."
        )

    return PolicyOutputSpec(
        target=target,
        command=command,
        dim=dim,
        meta=_copy_string_key_mapping(value.get("meta"), "policy_output_spec.meta"),
    )


def coerce_policy_spec(value: PolicySpec | Mapping[str, Any]) -> PolicySpec:
    """Normalize a ``PolicySpec`` or mapping into a standard :class:`PolicySpec`."""

    if isinstance(value, PolicySpec):
        return PolicySpec(
            name=value.name,
            required_image_keys=list(value.required_image_keys),
            required_state_keys=list(value.required_state_keys),
            required_task_keys=list(value.required_task_keys),
            outputs=[coerce_policy_output_spec(output) for output in value.outputs],
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"policy spec must be a PolicySpec or mapping, got {type(value).__name__}."
        )

    try:
        name = value["name"]
        required_image_keys = value["required_image_keys"]
        required_state_keys = value["required_state_keys"]
        outputs = value["outputs"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"policy spec mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return PolicySpec(
        name=name,
        required_image_keys=_copy_sequence(
            required_image_keys,
            "policy_spec.required_image_keys",
        ),
        required_state_keys=_copy_sequence(
            required_state_keys,
            "policy_spec.required_state_keys",
        ),
        required_task_keys=_copy_sequence(
            value.get("required_task_keys", []),
            "policy_spec.required_task_keys",
        ),
        outputs=[coerce_policy_output_spec(item) for item in _copy_sequence(outputs, "policy_spec.outputs")],
        meta=_copy_string_key_mapping(value.get("meta"), "policy_spec.meta"),
    )


__all__ = [
    "coerce_action",
    "coerce_command",
    "coerce_component_spec",
    "coerce_frame",
    "coerce_policy_output_spec",
    "coerce_policy_spec",
    "coerce_robot_spec",
]
