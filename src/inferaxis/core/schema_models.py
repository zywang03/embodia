"""Schema model objects for inferaxis runtime interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import time
from typing import Any

import numpy as np

from .arraylike import to_numpy_array
from .errors import InterfaceValidationError
from .schema_validation import (
    _coerce_numpy_mapping,
    _ensure_bool,
    _ensure_non_empty_string,
    _ensure_string_key_dict,
)

KNOWN_COMPONENT_TYPES: tuple[str, ...] = (
    "arm",
    "gripper",
    "hand",
    "suction",
    "base",
    "custom",
)


def _materialize_command(
    *,
    command: str,
    value: np.ndarray,
    ref_frame: str | None,
    meta: dict[str, Any],
) -> Command:
    """Build one already-normalized command without copying arrays."""

    materialized = object.__new__(Command)
    materialized.command = command
    materialized.value = value
    materialized.ref_frame = ref_frame
    materialized.meta = meta
    return materialized


@dataclass(slots=True)
class Frame:
    """One standardized observation frame."""

    images: dict[str, np.ndarray] = field(default_factory=dict)
    state: dict[str, np.ndarray] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=time.time_ns, init=False)
    task: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    sequence_id: int | None = field(default=None, init=False)
    copy: InitVar[bool] = False

    def __post_init__(self, copy: bool) -> None:
        """Normalize frame arrays at construction time."""

        copy = _ensure_bool(copy, "frame.copy")
        self.images = _coerce_numpy_mapping(
            self.images,
            "frame.images",
            wrap_scalar=False,
            copy=copy,
        )
        self.state = _coerce_numpy_mapping(
            self.state,
            "frame.state",
            wrap_scalar=True,
            copy=copy,
        )
        self.task = _ensure_string_key_dict(self.task, "frame.task")
        self.meta = _ensure_string_key_dict(self.meta, "frame.meta")


@dataclass(slots=True)
class Command:
    """One command payload for one robot component.

    The owning component name lives on ``Action.commands`` as the dictionary
    key, for example ``"left_arm"`` or ``"right_hand"``.
    """

    command: str
    value: np.ndarray
    ref_frame: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    copy: InitVar[bool] = False

    def __post_init__(self, copy: bool) -> None:
        """Normalize command vectors at construction time."""

        copy = _ensure_bool(copy, "command.copy")
        self.command = _ensure_non_empty_string(self.command, "command.command")
        self.value = to_numpy_array(
            self.value,
            field_name="command.value",
            wrap_scalar=True,
            numeric_only=True,
            allow_bool=False,
            copy=copy,
            dtype=np.float64,
        )
        if self.value.ndim != 1:
            raise InterfaceValidationError(
                f"command.value must be a 1D numeric vector, got ndim={self.value.ndim}."
            )
        self.meta = _ensure_string_key_dict(self.meta, "command.meta")


@dataclass(slots=True)
class Action:
    """One control step containing one or more component-keyed commands."""

    commands: dict[str, Command]
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def single(
        cls,
        *,
        target: str,
        command: str,
        value: object,
        ref_frame: str | None = None,
        command_meta: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        copy: bool = False,
    ) -> Action:
        """Build an action containing exactly one command."""

        copy = _ensure_bool(copy, "action.copy")
        normalized_command_meta = (
            {}
            if command_meta is None
            else _ensure_string_key_dict(dict(command_meta), "command.meta")
        )
        normalized_value = to_numpy_array(
            value,
            field_name="action.value",
            wrap_scalar=True,
            numeric_only=True,
            allow_bool=False,
            copy=copy,
            dtype=np.float64,
        )
        if normalized_value.ndim != 1:
            normalized_value = normalized_value.reshape(-1)

        return cls.from_commands(
            {
                target: _materialize_command(
                    command=_ensure_non_empty_string(command, "command.command"),
                    value=normalized_value,
                    ref_frame=ref_frame,
                    meta=normalized_command_meta,
                )
            },
            meta={} if meta is None else dict(meta),
            trusted=True,
        )

    @classmethod
    def from_commands(
        cls,
        commands: Mapping[str, Command | Mapping[str, Any]],
        *,
        meta: dict[str, Any] | None = None,
        trusted: bool = False,
        copy: bool = False,
    ) -> Action:
        """Build an action from existing command objects or command mappings."""

        trusted = _ensure_bool(trusted, "action.trusted")
        copy = _ensure_bool(copy, "action.copy")
        action_meta = {} if meta is None else dict(meta)

        if trusted:
            if not isinstance(commands, dict):
                raise InterfaceValidationError(
                    "trusted action.commands must be a dict[str, Command]."
                )
            for target, command in commands.items():
                _ensure_non_empty_string(target, f"action.commands[{target!r}] key")
                if not isinstance(command, Command):
                    raise InterfaceValidationError(
                        "trusted action.commands values must be Command instances, "
                        f"got {type(command).__name__} for target {target!r}."
                    )
            return cls(commands=commands, meta=action_meta)

        if not isinstance(commands, Mapping):
            raise InterfaceValidationError(
                f"action.commands must be a mapping, got {type(commands).__name__}."
            )

        normalized: dict[str, Command] = {}
        for target, item in commands.items():
            target_name = _ensure_non_empty_string(
                target,
                f"action.commands[{target!r}] key",
            )
            if isinstance(item, Command):
                normalized[target_name] = Command(
                    command=item.command,
                    value=item.value,
                    ref_frame=item.ref_frame,
                    meta=dict(item.meta),
                    copy=copy,
                )
                continue
            if not isinstance(item, Mapping):
                raise InterfaceValidationError(
                    f"action.commands[{target_name!r}] must be a Command or "
                    f"command mapping, got {type(item).__name__}."
                )
            if "target" in item and item["target"] != target_name:
                raise InterfaceValidationError(
                    f"action.commands[{target_name!r}] target mismatch: outer "
                    f"key is {target_name!r} but nested target is {item['target']!r}."
                )
            if "command" not in item:
                raise InterfaceValidationError(
                    f"action.commands[{target_name!r}] is missing required field "
                    "'command'."
                )
            if "value" not in item:
                raise InterfaceValidationError(
                    f"action.commands[{target_name!r}] is missing required field "
                    "'value'."
                )
            normalized[target_name] = Command(
                command=item["command"],
                value=item["value"],
                ref_frame=item.get("ref_frame"),
                meta={} if item.get("meta") is None else dict(item["meta"]),
                copy=copy,
            )

        return cls(
            commands=normalized,
            meta=action_meta,
        )

    def get_command(self, target: str) -> Command | None:
        """Return the command for ``target`` when present."""

        return self.commands.get(target)


@dataclass(slots=True)
class ComponentSpec:
    """Description of one controllable robot component."""

    name: str
    type: str
    dof: int
    command: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def supports_command(self, command: str) -> bool:
        """Return whether the component accepts ``command``."""

        return command in self.command


@dataclass(slots=True)
class RobotSpec:
    """Description of one robot embodiment."""

    name: str
    image_keys: list[str]
    components: list[ComponentSpec]
    meta: dict[str, Any] = field(default_factory=dict)

    def get_component(self, name: str) -> ComponentSpec | None:
        """Return one component by name when present."""

        for component in self.components:
            if component.name == name:
                return component
        return None

    def all_supported_commands(self) -> list[str]:
        """Return unique commands supported across all components."""

        seen: set[str] = set()
        result: list[str] = []
        for component in self.components:
            for command in component.command:
                if command not in seen:
                    seen.add(command)
                    result.append(command)
        return result

    def all_state_keys(self) -> list[str]:
        """Return the standardized state keys exposed across all components.

        inferaxis keeps the main observation and action flows aligned by using
        the component name itself as the canonical state key.
        """

        return [component.name for component in self.components]


@dataclass(slots=True)
class PolicyOutputSpec:
    """Description of one policy output command slot."""

    target: str
    command: str
    dim: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicySpec:
    """Description of one policy's required inputs and emitted commands."""

    name: str
    required_image_keys: list[str]
    required_state_keys: list[str]
    required_task_keys: list[str] = field(default_factory=list)
    outputs: list[PolicyOutputSpec] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def get_output(self, target: str) -> PolicyOutputSpec | None:
        """Return one output spec by target when present."""

        for output in self.outputs:
            if output.target == target:
                return output
        return None


__all__ = [
    "Action",
    "Command",
    "ComponentSpec",
    "Frame",
    "KNOWN_COMPONENT_TYPES",
    "PolicyOutputSpec",
    "PolicySpec",
    "RobotSpec",
]
