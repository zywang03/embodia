"""Coercion helpers for embodia schema objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..arraylike import optional_array_to_list
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


def _copy_float_vector(value: object, field_name: str) -> list[Any]:
    """Convert list-like action payloads into plain Python lists."""

    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)

    converted = optional_array_to_list(value, field_name=field_name)
    if converted is not None:
        return converted

    return _copy_sequence(value, field_name)


def coerce_frame(value: Frame | Mapping[str, Any]) -> Frame:
    """Normalize a ``Frame`` or mapping into a standard :class:`Frame`."""

    if isinstance(value, Frame):
        return Frame(
            timestamp_ns=value.timestamp_ns,
            images=dict(value.images),
            state=dict(value.state),
            task=dict(value.task),
            meta=dict(value.meta),
            sequence_id=value.sequence_id,
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"frame must be a Frame or mapping, got {type(value).__name__}."
        )

    try:
        timestamp_ns = value["timestamp_ns"]
        images = value["images"]
        state = value["state"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"frame mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return Frame(
        timestamp_ns=timestamp_ns,
        images=_copy_string_key_mapping(images, "frame.images"),
        state=_copy_string_key_mapping(state, "frame.state"),
        task=_copy_string_key_mapping(value.get("task"), "frame.task"),
        meta=_copy_string_key_mapping(value.get("meta"), "frame.meta"),
        sequence_id=value.get("sequence_id"),
    )


def coerce_command(value: Command | Mapping[str, Any]) -> Command:
    """Normalize a ``Command`` or mapping into a standard :class:`Command`."""

    if isinstance(value, Command):
        return Command(
            target=value.target,
            kind=value.kind,
            value=list(value.value),
            ref_frame=value.ref_frame,
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"command must be a Command or mapping, got {type(value).__name__}."
        )

    try:
        target = value["target"]
        command_value = value["value"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"command mapping is missing required field {exc.args[0]!r}."
        ) from exc

    kind = value.get("kind")
    if kind is None:
        raise InterfaceValidationError(
            "command mapping is missing required field 'kind'."
        )

    return Command(
        target=target,
        kind=kind,
        value=_copy_float_vector(command_value, "command.value"),
        ref_frame=value.get("ref_frame"),
        meta=_copy_string_key_mapping(value.get("meta"), "command.meta"),
    )


def coerce_action(value: Action | Mapping[str, Any]) -> Action:
    """Normalize an ``Action`` or mapping into a standard :class:`Action`."""

    if isinstance(value, Action):
        return Action(
            commands=[coerce_command(command) for command in value.commands],
            dt=value.dt,
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"action must be an Action or mapping, got {type(value).__name__}."
        )

    try:
        commands = value["commands"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"action mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return Action(
        commands=[coerce_command(item) for item in _copy_sequence(commands, "action.commands")],
        dt=value.get("dt", 0.1),
        meta=_copy_string_key_mapping(value.get("meta"), "action.meta"),
    )


def coerce_component_spec(
    value: ComponentSpec | Mapping[str, Any],
) -> ComponentSpec:
    """Normalize a component spec-like value."""

    if isinstance(value, ComponentSpec):
        return ComponentSpec(
            name=value.name,
            kind=value.kind,
            dof=value.dof,
            supported_command_kinds=list(value.supported_command_kinds),
            state_keys=list(value.state_keys),
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            "component spec must be ComponentSpec or mapping, got "
            f"{type(value).__name__}."
        )

    try:
        name = value["name"]
        kind = value["kind"]
        dof = value["dof"]
        supported_command_kinds = value.get("supported_command_kinds")
    except KeyError as exc:
        raise InterfaceValidationError(
            f"component spec mapping is missing required field {exc.args[0]!r}."
        ) from exc
    if supported_command_kinds is None:
        raise InterfaceValidationError(
            "component spec mapping is missing required field "
            "'supported_command_kinds'."
        )

    return ComponentSpec(
        name=name,
        kind=kind,
        dof=dof,
        supported_command_kinds=_copy_sequence(
            supported_command_kinds,
            "component_spec.supported_command_kinds",
        ),
        state_keys=_copy_sequence(
            value.get("state_keys", []),
            "component_spec.state_keys",
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
            command_kind=value.command_kind,
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
    command_kind = value.get("command_kind")
    if command_kind is None:
        raise InterfaceValidationError(
            "policy output spec mapping is missing required field 'command_kind'."
        )

    return PolicyOutputSpec(
        target=target,
        command_kind=command_kind,
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
