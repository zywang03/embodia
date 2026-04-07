"""Coercion helpers for embodia schema objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..arraylike import optional_array_to_list
from ..errors import InterfaceValidationError
from ..schema import Action, Frame, ModelSpec, RobotSpec


def _copy_string_key_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> dict[str, Any] | None:
    """Convert a mapping into a plain ``dict[str, Any]``."""

    if value is None:
        return None
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


def _copy_action_value(value: object, field_name: str) -> list[Any]:
    """Convert one action vector into a plain list."""

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
            task=None if value.task is None else dict(value.task),
            meta=None if value.meta is None else dict(value.meta),
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
        images=_copy_string_key_mapping(images, "frame.images") or {},
        state=_copy_string_key_mapping(state, "frame.state") or {},
        task=_copy_string_key_mapping(value.get("task"), "frame.task"),
        meta=_copy_string_key_mapping(value.get("meta"), "frame.meta"),
    )


def coerce_action(value: Action | Mapping[str, Any]) -> Action:
    """Normalize an ``Action`` or mapping into a standard :class:`Action`."""

    if isinstance(value, Action):
        return Action(
            mode=value.mode,
            value=list(value.value),
            gripper=value.gripper,
            ref_frame=value.ref_frame,
            dt=value.dt,
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"action must be an Action or mapping, got {type(value).__name__}."
        )

    try:
        mode = value["mode"]
        action_value = value["value"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"action mapping is missing required field {exc.args[0]!r}."
        ) from exc

    if "ref_frame" in value and "frame" in value:
        raise InterfaceValidationError(
            "action mapping must not contain both 'ref_frame' and legacy 'frame'."
        )

    return Action(
        mode=mode,
        value=_copy_action_value(action_value, "action.value"),
        gripper=value.get("gripper"),
        ref_frame=value.get("ref_frame", value.get("frame")),
        dt=value.get("dt", 0.1),
    )


def coerce_robot_spec(value: RobotSpec | Mapping[str, Any]) -> RobotSpec:
    """Normalize a ``RobotSpec`` or mapping into a standard :class:`RobotSpec`."""

    if isinstance(value, RobotSpec):
        return RobotSpec(
            name=value.name,
            action_modes=list(value.action_modes),
            image_keys=list(value.image_keys),
            state_keys=list(value.state_keys),
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"robot spec must be a RobotSpec or mapping, got {type(value).__name__}."
        )

    try:
        name = value["name"]
        action_modes = value["action_modes"]
        image_keys = value["image_keys"]
        state_keys = value["state_keys"]
    except KeyError as exc:
        raise InterfaceValidationError(
            f"robot spec mapping is missing required field {exc.args[0]!r}."
        ) from exc

    return RobotSpec(
        name=name,
        action_modes=_copy_sequence(action_modes, "robot_spec.action_modes"),
        image_keys=_copy_sequence(image_keys, "robot_spec.image_keys"),
        state_keys=_copy_sequence(state_keys, "robot_spec.state_keys"),
    )


def coerce_model_spec(value: ModelSpec | Mapping[str, Any]) -> ModelSpec:
    """Normalize a ``ModelSpec`` or mapping into a standard :class:`ModelSpec`."""

    if isinstance(value, ModelSpec):
        return ModelSpec(
            name=value.name,
            required_image_keys=list(value.required_image_keys),
            required_state_keys=list(value.required_state_keys),
            output_action_mode=value.output_action_mode,
        )
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"model spec must be a ModelSpec or mapping, got {type(value).__name__}."
        )

    try:
        name = value["name"]
        required_image_keys = value["required_image_keys"]
        required_state_keys = value["required_state_keys"]
        output_action_mode = value["output_action_mode"]
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
        output_action_mode=output_action_mode,
    )


__all__ = [
    "coerce_action",
    "coerce_frame",
    "coerce_model_spec",
    "coerce_robot_spec",
]
