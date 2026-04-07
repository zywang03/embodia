"""Field remapping helpers for embodia schema objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
from .coerce import (
    coerce_action,
    coerce_command,
    coerce_component_spec,
    coerce_frame,
    coerce_policy_output_spec,
    coerce_policy_spec,
    coerce_robot_spec,
)


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
        task=remap_mapping_keys(
            normalized.task,
            task_key_map or {},
            "frame.task",
        ),
        meta=remap_mapping_keys(
            normalized.meta,
            meta_key_map or {},
            "frame.meta",
        ),
        sequence_id=normalized.sequence_id,
    )


def remap_command(
    command: Command | Mapping[str, Any],
    *,
    target_map: Mapping[str, str] | None = None,
    kind_map: Mapping[str, str] | None = None,
    ref_frame_map: Mapping[str, str] | None = None,
) -> Command:
    """Rename one command's target/kind/reference frame."""

    normalized = coerce_command(command)
    mapped_ref_frame = normalized.ref_frame
    if mapped_ref_frame is not None:
        mapped_ref_frame = _remap_name(mapped_ref_frame, ref_frame_map or {})

    return Command(
        target=_remap_name(normalized.target, target_map or {}),
        kind=_remap_name(normalized.kind, kind_map or {}),
        value=list(normalized.value),
        ref_frame=mapped_ref_frame,
        meta=dict(normalized.meta),
    )


def remap_action(
    action: Action | Mapping[str, Any],
    *,
    target_map: Mapping[str, str] | None = None,
    kind_map: Mapping[str, str] | None = None,
    ref_frame_map: Mapping[str, str] | None = None,
) -> Action:
    """Rename action commands according to mapping tables."""

    normalized = coerce_action(action)
    return Action(
        commands=[
            remap_command(
                command,
                target_map=target_map,
                kind_map=kind_map,
                ref_frame_map=ref_frame_map,
            )
            for command in normalized.commands
        ],
        dt=normalized.dt,
        meta=dict(normalized.meta),
    )


def remap_component_spec(
    spec: ComponentSpec | Mapping[str, Any],
    *,
    target_map: Mapping[str, str] | None = None,
    state_key_map: Mapping[str, str] | None = None,
    command_kind_map: Mapping[str, str] | None = None,
) -> ComponentSpec:
    """Rename a component spec."""

    normalized = coerce_component_spec(spec)
    return ComponentSpec(
        name=_remap_name(normalized.name, target_map or {}),
        kind=normalized.kind,
        dof=normalized.dof,
        supported_command_kinds=_remap_name_list(
            normalized.supported_command_kinds,
            command_kind_map or {},
            "component_spec.supported_command_kinds",
        ),
        state_keys=_remap_name_list(
            normalized.state_keys,
            state_key_map or {},
            "component_spec.state_keys",
        ),
        meta=dict(normalized.meta),
    )


def remap_robot_spec(
    spec: RobotSpec | Mapping[str, Any],
    *,
    image_key_map: Mapping[str, str] | None = None,
    state_key_map: Mapping[str, str] | None = None,
    command_kind_map: Mapping[str, str] | None = None,
    target_map: Mapping[str, str] | None = None,
) -> RobotSpec:
    """Rename robot spec keys and component names according to mappings."""

    normalized = coerce_robot_spec(spec)
    return RobotSpec(
        name=normalized.name,
        image_keys=_remap_name_list(
            normalized.image_keys,
            image_key_map or {},
            "robot_spec.image_keys",
        ),
        components=[
            remap_component_spec(
                component,
                target_map=target_map,
                state_key_map=state_key_map,
                command_kind_map=command_kind_map,
            )
            for component in normalized.components
        ],
        meta=dict(normalized.meta),
    )


def remap_policy_output_spec(
    spec: PolicyOutputSpec | Mapping[str, Any],
    *,
    target_map: Mapping[str, str] | None = None,
    command_kind_map: Mapping[str, str] | None = None,
) -> PolicyOutputSpec:
    """Rename a policy-output spec."""

    normalized = coerce_policy_output_spec(spec)
    return PolicyOutputSpec(
        target=_remap_name(normalized.target, target_map or {}),
        command_kind=_remap_name(normalized.command_kind, command_kind_map or {}),
        dim=normalized.dim,
        meta=dict(normalized.meta),
    )


def remap_policy_spec(
    spec: PolicySpec | Mapping[str, Any],
    *,
    image_key_map: Mapping[str, str] | None = None,
    state_key_map: Mapping[str, str] | None = None,
    task_key_map: Mapping[str, str] | None = None,
    command_kind_map: Mapping[str, str] | None = None,
    target_map: Mapping[str, str] | None = None,
) -> PolicySpec:
    """Rename policy spec keys and output definitions according to mappings."""

    normalized = coerce_policy_spec(spec)
    return PolicySpec(
        name=normalized.name,
        required_image_keys=_remap_name_list(
            normalized.required_image_keys,
            image_key_map or {},
            "policy_spec.required_image_keys",
        ),
        required_state_keys=_remap_name_list(
            normalized.required_state_keys,
            state_key_map or {},
            "policy_spec.required_state_keys",
        ),
        required_task_keys=_remap_name_list(
            normalized.required_task_keys,
            task_key_map or {},
            "policy_spec.required_task_keys",
        ),
        outputs=[
            remap_policy_output_spec(
                output,
                target_map=target_map,
                command_kind_map=command_kind_map,
            )
            for output in normalized.outputs
        ],
        meta=dict(normalized.meta),
    )


__all__ = [
    "invert_mapping",
    "remap_action",
    "remap_command",
    "remap_component_spec",
    "remap_frame",
    "remap_mapping_keys",
    "remap_policy_output_spec",
    "remap_policy_spec",
    "remap_robot_spec",
]
