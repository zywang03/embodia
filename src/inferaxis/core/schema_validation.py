"""Validation helpers for inferaxis schema objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from .arraylike import to_numpy_array
from .errors import InterfaceValidationError


def _ensure_non_empty_string(value: object, field_name: str) -> str:
    """Validate one non-empty string value."""

    if not isinstance(value, str):
        raise InterfaceValidationError(
            f"{field_name} must be a string, got {type(value).__name__}."
        )
    text = value.strip()
    if not text:
        raise InterfaceValidationError(f"{field_name} must not be empty.")
    return text


def _ensure_string_key_dict(value: object, field_name: str) -> dict[str, Any]:
    """Validate one ``dict[str, Any]`` field."""

    if not isinstance(value, dict):
        raise InterfaceValidationError(
            f"{field_name} must be a dict[str, Any], got {type(value).__name__}."
        )
    for key in value:
        if not isinstance(key, str):
            raise InterfaceValidationError(
                f"{field_name} keys must be strings, got {key!r}."
            )
    return value


def _ensure_string_list(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> list[str]:
    """Validate one ``list[str]`` field."""

    if not isinstance(value, list):
        raise InterfaceValidationError(
            f"{field_name} must be a list[str], got {type(value).__name__}."
        )
    if not allow_empty and not value:
        raise InterfaceValidationError(f"{field_name} must not be empty.")

    seen: set[str] = set()
    result: list[str] = []
    for index, item in enumerate(value):
        text = _ensure_non_empty_string(item, f"{field_name}[{index}]")
        if text in seen:
            raise InterfaceValidationError(
                f"{field_name} contains duplicate entry {text!r}."
            )
        seen.add(text)
        result.append(text)
    return result


def _ensure_ndarray(
    value: object,
    field_name: str,
    *,
    allow_bool: bool,
    require_1d: bool,
    finite: bool,
) -> np.ndarray:
    """Validate one ndarray payload."""

    if not isinstance(value, np.ndarray):
        raise InterfaceValidationError(
            f"{field_name} must be a numpy.ndarray, got {type(value).__name__}."
        )
    if value.dtype == np.dtype("O"):
        raise InterfaceValidationError(f"{field_name} must not use object dtype.")
    if require_1d and value.ndim != 1:
        raise InterfaceValidationError(
            f"{field_name} must be 1D, got ndim={value.ndim}."
        )
    if np.issubdtype(value.dtype, np.bool_) and not allow_bool:
        raise InterfaceValidationError(
            f"{field_name} must use a real numeric dtype, got bool."
        )
    if not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
        or (allow_bool and np.issubdtype(value.dtype, np.bool_))
    ):
        raise InterfaceValidationError(
            f"{field_name} must use a numeric dtype, got {value.dtype}."
        )
    if finite and np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise InterfaceValidationError(f"{field_name} must be finite.")
    return value


def _ensure_positive_int(value: object, field_name: str) -> int:
    """Validate one positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise InterfaceValidationError(
            f"{field_name} must be an int, got {type(value).__name__}."
        )
    if value <= 0:
        raise InterfaceValidationError(f"{field_name} must be > 0, got {value!r}.")
    return value


def _coerce_numpy_mapping(
    value: object,
    field_name: str,
    *,
    wrap_scalar: bool,
    copy: bool,
) -> dict[str, np.ndarray]:
    """Normalize one string-keyed mapping into ndarray values."""

    mapping = _ensure_string_key_dict(value, field_name)
    return {
        key: to_numpy_array(
            item,
            field_name=f"{field_name}[{key!r}]",
            wrap_scalar=wrap_scalar,
            numeric_only=True,
            allow_bool=True,
            copy=copy,
        )
        for key, item in mapping.items()
    }


def _ensure_bool(value: object, field_name: str) -> bool:
    """Validate one boolean setting."""

    if not isinstance(value, bool):
        raise InterfaceValidationError(
            f"{field_name} must be a bool, got {type(value).__name__}."
        )
    return value


def _validate_command_kind_name(
    name: object,
    field_name: str,
    *,
    allow_unregistered_custom: bool,
) -> str:
    """Validate one command-kind name and return the normalized string."""

    from .command_kinds import (
        CUSTOM_COMMAND_KIND_PREFIX,
        is_custom_command_kind_name,
        is_known_command_kind,
    )

    text = _ensure_non_empty_string(name, field_name)
    if is_known_command_kind(text):
        return text
    if allow_unregistered_custom and is_custom_command_kind_name(text):
        return text
    raise InterfaceValidationError(
        f"{field_name} must reference a registered command kind or an "
        f"unregistered {CUSTOM_COMMAND_KIND_PREFIX}... extension, got {text!r}."
    )


def _kind_uses_component_dof(spec: Any) -> bool:
    """Return whether one kind should match the owning component dof."""

    from .command_kinds import _USES_COMPONENT_DOF_META_KEY

    return bool(spec.meta.get(_USES_COMPONENT_DOF_META_KEY, False))


def validate_frame(frame: object) -> None:
    """Validate a :class:`Frame` instance."""

    from .schema_models import Frame

    if not isinstance(frame, Frame):
        raise InterfaceValidationError(
            f"frame must be a Frame instance, got {type(frame).__name__}."
        )
    if isinstance(frame.timestamp_ns, bool) or not isinstance(frame.timestamp_ns, int):
        raise InterfaceValidationError(
            "frame.timestamp_ns must be an int representing nanoseconds."
        )
    if frame.timestamp_ns < 0:
        raise InterfaceValidationError("frame.timestamp_ns must be >= 0.")

    _ensure_string_key_dict(frame.images, "frame.images")
    _ensure_string_key_dict(frame.state, "frame.state")
    _ensure_string_key_dict(frame.task, "frame.task")
    _ensure_string_key_dict(frame.meta, "frame.meta")
    for key, value in frame.images.items():
        _ensure_ndarray(
            value,
            f"frame.images[{key!r}]",
            allow_bool=True,
            require_1d=False,
            finite=False,
        )
    for key, value in frame.state.items():
        _ensure_ndarray(
            value,
            f"frame.state[{key!r}]",
            allow_bool=True,
            require_1d=False,
            finite=False,
        )
    if frame.sequence_id is not None:
        if isinstance(frame.sequence_id, bool) or not isinstance(frame.sequence_id, int):
            raise InterfaceValidationError(
                "frame.sequence_id must be an int when provided."
            )


def validate_command(cmd: object) -> None:
    """Validate one command structurally and against registry metadata."""

    from .command_kinds import get_command_kind_spec, is_known_command_kind
    from .schema_models import Command

    if not isinstance(cmd, Command):
        raise InterfaceValidationError(
            f"command must be a Command instance, got {type(cmd).__name__}."
        )

    command = _validate_command_kind_name(
        cmd.command,
        "command.command",
        allow_unregistered_custom=True,
    )
    _ensure_ndarray(
        cmd.value,
        "command.value",
        allow_bool=False,
        require_1d=True,
        finite=True,
    )
    if cmd.ref_frame is not None:
        _ensure_non_empty_string(cmd.ref_frame, "command.ref_frame")
    _ensure_string_key_dict(cmd.meta, "command.meta")

    if not is_known_command_kind(command):
        return

    spec = get_command_kind_spec(command)
    if spec.requires_ref_frame and cmd.ref_frame is None:
        raise InterfaceValidationError(
            f"command.command {command!r} requires command.ref_frame."
        )
    if spec.default_dim is not None and len(cmd.value) != spec.default_dim:
        raise InterfaceValidationError(
            f"command.command {command!r} expects dim={spec.default_dim}, got "
            f"{len(cmd.value)}."
        )


def validate_action(action: object) -> None:
    """Validate one action."""

    from .schema_models import Action

    if not isinstance(action, Action):
        raise InterfaceValidationError(
            f"action must be an Action instance, got {type(action).__name__}."
        )
    if not isinstance(action.commands, dict):
        raise InterfaceValidationError(
            f"action.commands must be a dict[str, Command], got "
            f"{type(action.commands).__name__}."
        )
    if not action.commands:
        raise InterfaceValidationError("action.commands must not be empty.")

    for target, command in action.commands.items():
        if not isinstance(target, str):
            raise InterfaceValidationError(
                f"action.commands keys must be strings, got {target!r}."
            )
        _ensure_non_empty_string(target, f"action.commands[{target!r}] key")
        try:
            validate_command(command)
        except InterfaceValidationError as exc:
            raise InterfaceValidationError(
                f"invalid action.commands[{target!r}]: {exc}"
            ) from exc

    _ensure_string_key_dict(action.meta, "action.meta")


def validate_component_spec(spec: object) -> None:
    """Validate one robot component spec."""

    from .command_kinds import get_command_kind_spec, is_known_command_kind
    from .schema_models import ComponentSpec

    if not isinstance(spec, ComponentSpec):
        raise InterfaceValidationError(
            "spec must be a ComponentSpec instance, got "
            f"{type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.name, "component_spec.name")
    _ensure_non_empty_string(spec.type, "component_spec.type")
    _ensure_positive_int(spec.dof, "component_spec.dof")
    supported = _ensure_string_list(
        spec.command,
        "component_spec.command",
        allow_empty=False,
    )
    _ensure_string_key_dict(spec.meta, "component_spec.meta")

    for index, command in enumerate(supported):
        command_name = _validate_command_kind_name(
            command,
            f"component_spec.command[{index}]",
            allow_unregistered_custom=True,
        )
        if not is_known_command_kind(command_name):
            continue
        command_spec = get_command_kind_spec(command_name)
        if (
            command_spec.allowed_component_types
            and spec.type not in command_spec.allowed_component_types
        ):
            raise InterfaceValidationError(
                f"component_spec {spec.name!r} has type {spec.type!r}, which "
                f"is incompatible with command {command_name!r}; allowed component "
                f"types: {command_spec.allowed_component_types!r}."
            )


def validate_robot_spec(spec: object) -> None:
    """Validate one robot embodiment spec."""

    from .schema_models import RobotSpec

    if not isinstance(spec, RobotSpec):
        raise InterfaceValidationError(
            f"spec must be a RobotSpec instance, got {type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.name, "robot_spec.name")
    _ensure_string_list(spec.image_keys, "robot_spec.image_keys", allow_empty=True)
    _ensure_string_key_dict(spec.meta, "robot_spec.meta")
    if not isinstance(spec.components, list):
        raise InterfaceValidationError(
            f"robot_spec.components must be a list, got "
            f"{type(spec.components).__name__}."
        )
    if not spec.components:
        raise InterfaceValidationError("robot_spec.components must not be empty.")

    seen_names: set[str] = set()
    for index, component in enumerate(spec.components):
        try:
            validate_component_spec(component)
        except InterfaceValidationError as exc:
            raise InterfaceValidationError(
                f"invalid robot_spec.components[{index}]: {exc}"
            ) from exc
        if component.name in seen_names:
            raise InterfaceValidationError(
                f"robot_spec.components contains duplicate component "
                f"{component.name!r}."
            )
        seen_names.add(component.name)


def validate_policy_output_spec(spec: object) -> None:
    """Validate one policy-output description."""

    from .command_kinds import get_command_kind_spec, is_known_command_kind
    from .schema_models import PolicyOutputSpec

    if not isinstance(spec, PolicyOutputSpec):
        raise InterfaceValidationError(
            "spec must be a PolicyOutputSpec instance, got "
            f"{type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.target, "policy_output_spec.target")
    command = _validate_command_kind_name(
        spec.command,
        "policy_output_spec.command",
        allow_unregistered_custom=True,
    )
    _ensure_positive_int(spec.dim, "policy_output_spec.dim")
    _ensure_string_key_dict(spec.meta, "policy_output_spec.meta")

    if not is_known_command_kind(command):
        return

    command_spec = get_command_kind_spec(command)
    if command_spec.default_dim is not None and spec.dim != command_spec.default_dim:
        raise InterfaceValidationError(
            f"policy_output_spec.command {command!r} expects dim="
            f"{command_spec.default_dim}, got {spec.dim}."
        )


def validate_policy_spec(spec: object) -> None:
    """Validate one policy interface spec."""

    from .schema_models import PolicySpec

    if not isinstance(spec, PolicySpec):
        raise InterfaceValidationError(
            f"spec must be a PolicySpec instance, got {type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.name, "policy_spec.name")
    _ensure_string_list(
        spec.required_image_keys,
        "policy_spec.required_image_keys",
        allow_empty=True,
    )
    _ensure_string_list(
        spec.required_state_keys,
        "policy_spec.required_state_keys",
        allow_empty=True,
    )
    _ensure_string_list(
        spec.required_task_keys,
        "policy_spec.required_task_keys",
        allow_empty=True,
    )
    _ensure_string_key_dict(spec.meta, "policy_spec.meta")
    if not isinstance(spec.outputs, list):
        raise InterfaceValidationError(
            f"policy_spec.outputs must be a list, got {type(spec.outputs).__name__}."
        )
    if not spec.outputs:
        raise InterfaceValidationError("policy_spec.outputs must not be empty.")

    seen_targets: set[str] = set()
    for index, output in enumerate(spec.outputs):
        try:
            validate_policy_output_spec(output)
        except InterfaceValidationError as exc:
            raise InterfaceValidationError(
                f"invalid policy_spec.outputs[{index}]: {exc}"
            ) from exc
        if output.target in seen_targets:
            raise InterfaceValidationError(
                f"policy_spec.outputs contains duplicate target {output.target!r}."
            )
        seen_targets.add(output.target)


__all__ = [
    "_coerce_numpy_mapping",
    "_ensure_bool",
    "_ensure_ndarray",
    "_ensure_non_empty_string",
    "_ensure_positive_int",
    "_ensure_string_key_dict",
    "_ensure_string_list",
    "_kind_uses_component_dof",
    "_validate_command_kind_name",
    "validate_action",
    "validate_command",
    "validate_component_spec",
    "validate_frame",
    "validate_policy_output_spec",
    "validate_policy_spec",
    "validate_robot_spec",
]
