"""Field remapping helpers for embodia schema objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import InterfaceValidationError
from ..schema import Action, Frame, ModelSpec, RobotSpec
from .coerce import coerce_action, coerce_frame, coerce_model_spec, coerce_robot_spec


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
    ref_frame_map: Mapping[str, str] | None = None,
    frame_map: Mapping[str, str] | None = None,
) -> Action:
    """Rename action fields according to mapping tables."""

    normalized = coerce_action(action)
    mapped_mode = _remap_name(normalized.mode, mode_map or {})
    active_ref_frame_map = ref_frame_map or frame_map or {}
    mapped_ref_frame = normalized.ref_frame
    if mapped_ref_frame is not None:
        mapped_ref_frame = _remap_name(mapped_ref_frame, active_ref_frame_map)

    return Action(
        mode=mapped_mode,
        value=list(normalized.value),
        gripper=normalized.gripper,
        ref_frame=mapped_ref_frame,
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


__all__ = [
    "invert_mapping",
    "remap_action",
    "remap_frame",
    "remap_mapping_keys",
    "remap_model_spec",
    "remap_robot_spec",
]
