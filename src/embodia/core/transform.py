"""Helpers for normalizing loose structures into embodia dataclasses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import InterfaceValidationError
from .schema import Action, Frame, ModelSpec, RobotSpec


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


def _remap_name(value: str, key_map: Mapping[str, str]) -> str:
    """Remap a single string key while preserving unknown names."""

    return key_map.get(value, value)


def _remap_name_list(
    values: Sequence[str],
    key_map: Mapping[str, str],
    field_name: str,
) -> list[str]:
    """Remap a list of names and detect duplicates introduced by mapping."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        mapped = _remap_name(value, key_map)
        if mapped in seen:
            raise InterfaceValidationError(
                f"{field_name} remapping creates duplicate entry {mapped!r}."
            )
        seen.add(mapped)
        result.append(mapped)
    return result


def remap_mapping_keys(
    value: Mapping[str, Any],
    key_map: Mapping[str, str],
    field_name: str,
) -> dict[str, Any]:
    """Rename dictionary keys according to ``key_map``."""

    result: dict[str, Any] = {}
    for key, item in value.items():
        mapped_key = _remap_name(key, key_map)
        if mapped_key in result and mapped_key != key:
            raise InterfaceValidationError(
                f"{field_name} remapping creates duplicate key {mapped_key!r}."
            )
        result[mapped_key] = item
    return result


def invert_mapping(
    key_map: Mapping[str, str],
    field_name: str = "mapping",
) -> dict[str, str]:
    """Invert a name mapping while detecting collisions."""

    inverse: dict[str, str] = {}
    for source, target in key_map.items():
        if target in inverse:
            raise InterfaceValidationError(
                f"{field_name} cannot be inverted because target {target!r} "
                "appears more than once."
            )
        inverse[target] = source
    return inverse


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
            frame=value.frame,
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

    return Action(
        mode=mode,
        value=_copy_sequence(action_value, "action.value"),
        gripper=value.get("gripper"),
        frame=value.get("frame"),
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


def remap_frame(
    frame: Frame | Mapping[str, Any],
    *,
    image_key_map: Mapping[str, str] | None = None,
    state_key_map: Mapping[str, str] | None = None,
    task_key_map: Mapping[str, str] | None = None,
    meta_key_map: Mapping[str, str] | None = None,
) -> Frame:
    """Rename frame sub-dictionary keys according to mapping tables."""

    normalized = coerce_frame(frame)
    return Frame(
        timestamp_ns=normalized.timestamp_ns,
        images=remap_mapping_keys(
            normalized.images,
            image_key_map or {},
            "frame.images",
        ),
        state=remap_mapping_keys(
            normalized.state,
            state_key_map or {},
            "frame.state",
        ),
        task=None
        if normalized.task is None
        else remap_mapping_keys(normalized.task, task_key_map or {}, "frame.task"),
        meta=None
        if normalized.meta is None
        else remap_mapping_keys(normalized.meta, meta_key_map or {}, "frame.meta"),
    )


def remap_action(
    action: Action | Mapping[str, Any],
    *,
    mode_map: Mapping[str, str] | None = None,
    frame_map: Mapping[str, str] | None = None,
) -> Action:
    """Rename action fields according to mapping tables."""

    normalized = coerce_action(action)
    mapped_mode = _remap_name(normalized.mode, mode_map or {})
    mapped_frame = normalized.frame
    if mapped_frame is not None:
        mapped_frame = _remap_name(mapped_frame, frame_map or {})

    return Action(
        mode=mapped_mode,
        value=list(normalized.value),
        gripper=normalized.gripper,
        frame=mapped_frame,
        dt=normalized.dt,
    )


def remap_robot_spec(
    spec: RobotSpec | Mapping[str, Any],
    *,
    image_key_map: Mapping[str, str] | None = None,
    state_key_map: Mapping[str, str] | None = None,
    action_mode_map: Mapping[str, str] | None = None,
) -> RobotSpec:
    """Rename robot spec keys and action modes according to mapping tables."""

    normalized = coerce_robot_spec(spec)
    return RobotSpec(
        name=normalized.name,
        action_modes=_remap_name_list(
            normalized.action_modes,
            action_mode_map or {},
            "robot_spec.action_modes",
        ),
        image_keys=_remap_name_list(
            normalized.image_keys,
            image_key_map or {},
            "robot_spec.image_keys",
        ),
        state_keys=_remap_name_list(
            normalized.state_keys,
            state_key_map or {},
            "robot_spec.state_keys",
        ),
    )


def remap_model_spec(
    spec: ModelSpec | Mapping[str, Any],
    *,
    image_key_map: Mapping[str, str] | None = None,
    state_key_map: Mapping[str, str] | None = None,
    action_mode_map: Mapping[str, str] | None = None,
) -> ModelSpec:
    """Rename model spec keys and output mode according to mapping tables."""

    normalized = coerce_model_spec(spec)
    return ModelSpec(
        name=normalized.name,
        required_image_keys=_remap_name_list(
            normalized.required_image_keys,
            image_key_map or {},
            "model_spec.required_image_keys",
        ),
        required_state_keys=_remap_name_list(
            normalized.required_state_keys,
            state_key_map or {},
            "model_spec.required_state_keys",
        ),
        output_action_mode=_remap_name(
            normalized.output_action_mode,
            action_mode_map or {},
        ),
    )


def frame_to_dict(frame: Frame | Mapping[str, Any]) -> dict[str, Any]:
    """Export a frame-like object into a shallow plain dictionary."""

    normalized = coerce_frame(frame)
    return {
        "timestamp_ns": normalized.timestamp_ns,
        "images": dict(normalized.images),
        "state": dict(normalized.state),
        "task": None if normalized.task is None else dict(normalized.task),
        "meta": None if normalized.meta is None else dict(normalized.meta),
    }


def action_to_dict(action: Action | Mapping[str, Any]) -> dict[str, Any]:
    """Export an action-like object into a shallow plain dictionary."""

    normalized = coerce_action(action)
    return {
        "mode": normalized.mode,
        "value": list(normalized.value),
        "gripper": normalized.gripper,
        "frame": normalized.frame,
        "dt": normalized.dt,
    }


def robot_spec_to_dict(spec: RobotSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a robot spec-like object into a shallow plain dictionary."""

    normalized = coerce_robot_spec(spec)
    return {
        "name": normalized.name,
        "action_modes": list(normalized.action_modes),
        "image_keys": list(normalized.image_keys),
        "state_keys": list(normalized.state_keys),
    }


def model_spec_to_dict(spec: ModelSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a model spec-like object into a shallow plain dictionary."""

    normalized = coerce_model_spec(spec)
    return {
        "name": normalized.name,
        "required_image_keys": list(normalized.required_image_keys),
        "required_state_keys": list(normalized.required_state_keys),
        "output_action_mode": normalized.output_action_mode,
    }


__all__ = [
    "action_to_dict",
    "coerce_action",
    "coerce_frame",
    "coerce_model_spec",
    "coerce_robot_spec",
    "frame_to_dict",
    "invert_mapping",
    "model_spec_to_dict",
    "remap_action",
    "remap_frame",
    "remap_mapping_keys",
    "remap_model_spec",
    "remap_robot_spec",
    "robot_spec_to_dict",
]
