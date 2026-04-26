"""Command kind registry for inferaxis schema objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .schema_validation import (
    _ensure_non_empty_string,
    _ensure_positive_int,
    _ensure_string_key_dict,
    _ensure_string_list,
)

CUSTOM_COMMAND_KIND_PREFIX = "custom:"
_USES_COMPONENT_DOF_META_KEY = "uses_component_dof"


class BuiltinCommandKind(StrEnum):
    """Canonical built-in command-kind names."""

    JOINT_POSITION = "joint_position"
    JOINT_POSITION_DELTA = "joint_position_delta"
    JOINT_VELOCITY = "joint_velocity"
    CARTESIAN_POSE = "cartesian_pose"
    CARTESIAN_POSE_DELTA = "cartesian_pose_delta"
    CARTESIAN_TWIST = "cartesian_twist"
    GRIPPER_POSITION = "gripper_position"
    GRIPPER_POSITION_DELTA = "gripper_position_delta"
    GRIPPER_VELOCITY = "gripper_velocity"
    GRIPPER_OPEN_CLOSE = "gripper_open_close"
    HAND_JOINT_POSITION = "hand_joint_position"
    HAND_JOINT_POSITION_DELTA = "hand_joint_position_delta"
    EEF_ACTIVATION = "eef_activation"


@dataclass(slots=True)
class CommandKindSpec:
    """Metadata and lightweight validation rules for one command kind."""

    name: str
    description: str = ""
    requires_ref_frame: bool = False
    default_dim: int | None = None
    allowed_component_types: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


COMMAND_KIND_REGISTRY: dict[str, CommandKindSpec] = {}


def register_command_kind(spec: CommandKindSpec) -> None:
    """Register one command-kind specification."""

    if not isinstance(spec, CommandKindSpec):
        raise TypeError(
            "register_command_kind() expects CommandKindSpec, got "
            f"{type(spec).__name__}."
        )

    name = _ensure_non_empty_string(spec.name, "command_kind_spec.name")
    if name in COMMAND_KIND_REGISTRY:
        raise ValueError(f"Command kind {name!r} is already registered.")

    if spec.default_dim is not None:
        _ensure_positive_int(spec.default_dim, "command_kind_spec.default_dim")
    spec.allowed_component_types = _ensure_string_list(
        spec.allowed_component_types,
        "command_kind_spec.allowed_component_types",
        allow_empty=True,
    )
    _ensure_string_key_dict(spec.meta, "command_kind_spec.meta")
    spec.name = name
    COMMAND_KIND_REGISTRY[name] = spec


def get_command_kind_spec(name: str) -> CommandKindSpec:
    """Return the registered specification for ``name``.

    Raises ``KeyError`` when the kind has not been registered.
    """

    try:
        return COMMAND_KIND_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown command kind {name!r}.") from exc


def is_known_command_kind(name: str) -> bool:
    """Return whether ``name`` is registered."""

    return name in COMMAND_KIND_REGISTRY


def is_custom_command_kind_name(name: str) -> bool:
    """Return whether ``name`` uses the ``custom:...`` namespace."""

    return name.startswith(CUSTOM_COMMAND_KIND_PREFIX) and bool(
        name[len(CUSTOM_COMMAND_KIND_PREFIX) :].strip()
    )


def _builtin_command_kind_specs() -> tuple[CommandKindSpec, ...]:
    """Return the built-in canonical command kinds."""

    return (
        CommandKindSpec(
            name=BuiltinCommandKind.JOINT_POSITION,
            description="Absolute joint position command.",
            allowed_component_types=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.JOINT_POSITION_DELTA,
            description="Joint position delta command.",
            allowed_component_types=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.JOINT_VELOCITY,
            description="Joint velocity command.",
            allowed_component_types=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.CARTESIAN_POSE,
            description="End-effector cartesian pose command.",
            allowed_component_types=["arm", "custom"],
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            description="End-effector cartesian pose delta command.",
            allowed_component_types=["arm", "custom"],
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.CARTESIAN_TWIST,
            description="End-effector cartesian twist command.",
            default_dim=6,
            allowed_component_types=["arm", "custom"],
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.GRIPPER_POSITION,
            description="Absolute gripper position command.",
            default_dim=1,
            allowed_component_types=["gripper", "custom"],
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.GRIPPER_POSITION_DELTA,
            description="Gripper position delta command.",
            default_dim=1,
            allowed_component_types=["gripper", "custom"],
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.GRIPPER_VELOCITY,
            description="Gripper velocity command.",
            default_dim=1,
            allowed_component_types=["gripper", "custom"],
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.GRIPPER_OPEN_CLOSE,
            description="Binary or scalar open/close gripper command.",
            default_dim=1,
            allowed_component_types=["gripper", "custom"],
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.HAND_JOINT_POSITION,
            description="Absolute dexterous-hand joint position command.",
            allowed_component_types=["hand", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.HAND_JOINT_POSITION_DELTA,
            description="Dexterous-hand joint position delta command.",
            allowed_component_types=["hand", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name=BuiltinCommandKind.EEF_ACTIVATION,
            description="Generic end-effector activation command.",
            default_dim=1,
            allowed_component_types=["gripper", "hand", "suction", "custom"],
        ),
    )


def _register_builtin_command_kinds() -> None:
    """Populate the built-in command kinds once at import time."""

    for spec in _builtin_command_kind_specs():
        register_command_kind(spec)


_register_builtin_command_kinds()


__all__ = [
    "BuiltinCommandKind",
    "COMMAND_KIND_REGISTRY",
    "CUSTOM_COMMAND_KIND_PREFIX",
    "CommandKindSpec",
    "_USES_COMPONENT_DOF_META_KEY",
    "get_command_kind_spec",
    "is_custom_command_kind_name",
    "is_known_command_kind",
    "register_command_kind",
]
