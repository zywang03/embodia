"""Coercion helpers for embodia schema objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..arraylike import optional_array_to_list
from ..errors import InterfaceValidationError
from ..schema import (
    Action,
    Command,
    ControlGroupSpec,
    Frame,
    ModelOutputSpec,
    ModelSpec,
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
            mode=value.mode,
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
        mode = value["mode"]
        command_value = value["value"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"command mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return Command(
        target=target,
        mode=mode,
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


def coerce_control_group_spec(
    value: ControlGroupSpec | Mapping[str, Any],
) -> ControlGroupSpec:
    """Normalize a control-group spec-like value."""

    if isinstance(value, ControlGroupSpec):
        return ControlGroupSpec(
            name=value.name,
            kind=value.kind,
            dof=value.dof,
            action_modes=list(value.action_modes),
            state_keys=list(value.state_keys),
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            "control group spec must be ControlGroupSpec or mapping, got "
            f"{type(value).__name__}."
        )

    try:
        name = value["name"]
        kind = value["kind"]
        dof = value["dof"]
        action_modes = value["action_modes"]
        state_keys = value["state_keys"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"control group spec mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return ControlGroupSpec(
        name=name,
        kind=kind,
        dof=dof,
        action_modes=_copy_sequence(action_modes, "control_group_spec.action_modes"),
        state_keys=_copy_sequence(state_keys, "control_group_spec.state_keys"),
        meta=_copy_string_key_mapping(value.get("meta"), "control_group_spec.meta"),
    )


def coerce_robot_spec(value: RobotSpec | Mapping[str, Any]) -> RobotSpec:
    """Normalize a ``RobotSpec`` or mapping into a standard :class:`RobotSpec`."""

    if isinstance(value, RobotSpec):
        return RobotSpec(
            name=value.name,
            image_keys=list(value.image_keys),
            groups=[coerce_control_group_spec(group) for group in value.groups],
            task_keys=list(value.task_keys),
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"robot spec must be a RobotSpec or mapping, got {type(value).__name__}."
        )

    try:
        name = value["name"]
        image_keys = value["image_keys"]
        groups = value["groups"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"robot spec mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return RobotSpec(
        name=name,
        image_keys=_copy_sequence(image_keys, "robot_spec.image_keys"),
        groups=[coerce_control_group_spec(item) for item in _copy_sequence(groups, "robot_spec.groups")],
        task_keys=_copy_sequence(value.get("task_keys", []), "robot_spec.task_keys"),
        meta=_copy_string_key_mapping(value.get("meta"), "robot_spec.meta"),
    )


def coerce_model_output_spec(
    value: ModelOutputSpec | Mapping[str, Any],
) -> ModelOutputSpec:
    """Normalize a model-output spec-like value."""

    if isinstance(value, ModelOutputSpec):
        return ModelOutputSpec(
            target=value.target,
            mode=value.mode,
            dim=value.dim,
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            "model output spec must be ModelOutputSpec or mapping, got "
            f"{type(value).__name__}."
        )

    try:
        target = value["target"]
        mode = value["mode"]
        dim = value["dim"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"model output spec mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return ModelOutputSpec(
        target=target,
        mode=mode,
        dim=dim,
        meta=_copy_string_key_mapping(value.get("meta"), "model_output_spec.meta"),
    )


def coerce_model_spec(value: ModelSpec | Mapping[str, Any]) -> ModelSpec:
    """Normalize a ``ModelSpec`` or mapping into a standard :class:`ModelSpec`."""

    if isinstance(value, ModelSpec):
        return ModelSpec(
            name=value.name,
            required_image_keys=list(value.required_image_keys),
            required_state_keys=list(value.required_state_keys),
            required_task_keys=list(value.required_task_keys),
            outputs=[coerce_model_output_spec(output) for output in value.outputs],
            meta=dict(value.meta),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"model spec must be a ModelSpec or mapping, got {type(value).__name__}."
        )

    try:
        name = value["name"]
        required_image_keys = value["required_image_keys"]
        required_state_keys = value["required_state_keys"]
        outputs = value["outputs"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"model spec mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return ModelSpec(
        name=name,
        required_image_keys=_copy_sequence(
            required_image_keys,
            "model_spec.required_image_keys",
        ),
        required_state_keys=_copy_sequence(
            required_state_keys,
            "model_spec.required_state_keys",
        ),
        required_task_keys=_copy_sequence(
            value.get("required_task_keys", []),
            "model_spec.required_task_keys",
        ),
        outputs=[coerce_model_output_spec(item) for item in _copy_sequence(outputs, "model_spec.outputs")],
        meta=_copy_string_key_mapping(value.get("meta"), "model_spec.meta"),
    )


__all__ = [
    "coerce_action",
    "coerce_command",
    "coerce_control_group_spec",
    "coerce_frame",
    "coerce_model_output_spec",
    "coerce_model_spec",
    "coerce_robot_spec",
]
