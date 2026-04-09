"""Lightweight shared schema for robot/policy runtime interaction.

This module is intentionally small. It standardizes the data objects that sit
between robot adapters and policy adapters without turning inferaxis into a full
robotics middleware stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
import time
from typing import Any

import numpy as np

from .arraylike import to_numpy_array
from .errors import InterfaceValidationError

KNOWN_COMPONENT_TYPES: tuple[str, ...] = (
    "arm",
    "gripper",
    "hand",
    "suction",
    "base",
    "custom",
)
CUSTOM_COMMAND_KIND_PREFIX = "custom:"
_USES_COMPONENT_DOF_META_KEY = "uses_component_dof"


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


def _ensure_real_number(value: object, field_name: str) -> float:
    """Validate one finite real number while rejecting ``bool``."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise InterfaceValidationError(
            f"{field_name} must be a real number, got {type(value).__name__}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise InterfaceValidationError(f"{field_name} must be finite, got {value!r}.")
    return number


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
        raise InterfaceValidationError(
            f"{field_name} must not use object dtype."
        )
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
            copy=True,
        )
        for key, item in mapping.items()
    }


def is_custom_command_kind_name(name: str) -> bool:
    """Return whether ``name`` uses the ``custom:...`` namespace."""

    return name.startswith(CUSTOM_COMMAND_KIND_PREFIX) and bool(
        name[len(CUSTOM_COMMAND_KIND_PREFIX) :].strip()
    )


def _validate_command_kind_name(
    name: object,
    field_name: str,
    *,
    allow_unregistered_custom: bool,
) -> str:
    """Validate one command-kind name and return the normalized string."""

    text = _ensure_non_empty_string(name, field_name)
    if is_known_command_kind(text):
        return text
    if allow_unregistered_custom and is_custom_command_kind_name(text):
        return text
    raise InterfaceValidationError(
        f"{field_name} must reference a registered command kind or an "
        f"unregistered {CUSTOM_COMMAND_KIND_PREFIX}... extension, got {text!r}."
    )


def _kind_uses_component_dof(spec: CommandKindSpec) -> bool:
    """Return whether one kind should match the owning component dof."""

    return bool(spec.meta.get(_USES_COMPONENT_DOF_META_KEY, False))


@dataclass(slots=True)
class Frame:
    """One standardized observation frame."""

    images: dict[str, np.ndarray] = field(default_factory=dict)
    state: dict[str, np.ndarray] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=time.time_ns, init=False)
    task: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    sequence_id: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Normalize frame arrays at construction time."""

        self.images = _coerce_numpy_mapping(
            self.images,
            "frame.images",
            wrap_scalar=False,
        )
        self.state = _coerce_numpy_mapping(
            self.state,
            "frame.state",
            wrap_scalar=True,
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

    def __post_init__(self) -> None:
        """Normalize command vectors at construction time."""

        self.value = to_numpy_array(
            self.value,
            field_name="command.value",
            wrap_scalar=True,
            numeric_only=True,
            allow_bool=False,
            copy=True,
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
    ) -> Action:
        """Build an action containing exactly one command."""

        return cls(
            commands={
                target: Command(
                    command=command,
                    value=to_numpy_array(
                        value,
                        field_name="action.value",
                        wrap_scalar=True,
                        numeric_only=True,
                        allow_bool=False,
                        copy=True,
                        dtype=np.float64,
                    ).reshape(-1),
                    ref_frame=ref_frame,
                    meta={} if command_meta is None else dict(command_meta),
                )
            },
            meta={} if meta is None else dict(meta),
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


def validate_frame(frame: Frame) -> None:
    """Validate a :class:`Frame` instance."""

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


def validate_command(cmd: Command) -> None:
    """Validate one command structurally and against registry metadata."""

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


def validate_action(action: Action) -> None:
    """Validate one action."""

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


def validate_component_spec(spec: ComponentSpec) -> None:
    """Validate one robot component spec."""

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


def validate_robot_spec(spec: RobotSpec) -> None:
    """Validate one robot embodiment spec."""

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


def validate_policy_output_spec(spec: PolicyOutputSpec) -> None:
    """Validate one policy-output description."""

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


def validate_policy_spec(spec: PolicySpec) -> None:
    """Validate one policy interface spec."""

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


def ensure_action_supported_by_robot(action: Action, spec: RobotSpec) -> None:
    """Ensure an action is compatible with one robot spec."""

    validate_action(action)
    validate_robot_spec(spec)

    for target, command in action.commands.items():
        component = spec.get_component(target)
        if component is None:
            raise InterfaceValidationError(
                f"robot {spec.name!r} does not define component "
                f"{target!r}."
            )
        if command.command not in component.command:
            raise InterfaceValidationError(
                f"robot {spec.name!r} component {component.name!r} does not support "
                f"command {command.command!r}; supported commands: "
                f"{component.command!r}."
            )
        if not is_known_command_kind(command.command):
            continue
        command_spec = get_command_kind_spec(command.command)
        if (
            command_spec.allowed_component_types
            and component.type not in command_spec.allowed_component_types
        ):
            raise InterfaceValidationError(
                f"command {command.command!r} is not allowed for robot component "
                f"{component.name!r} of type {component.type!r}; allowed component "
                f"types: {command_spec.allowed_component_types!r}."
            )
        if _kind_uses_component_dof(command_spec) and len(command.value) != component.dof:
            raise InterfaceValidationError(
                f"command {command.command!r} for robot component "
                f"{component.name!r} must match component dof={component.dof}, got "
                f"dim={len(command.value)}."
            )


def ensure_action_matches_policy_spec(action: Action, spec: PolicySpec) -> None:
    """Ensure an action matches one policy's declared outputs."""

    validate_action(action)
    validate_policy_spec(spec)

    command_targets = set(action.commands)
    output_targets = {output.target for output in spec.outputs}
    if command_targets != output_targets:
        missing = sorted(output_targets - command_targets)
        extra = sorted(command_targets - output_targets)
        details: list[str] = []
        if missing:
            details.append(f"missing targets: {missing!r}")
        if extra:
            details.append(f"unexpected targets: {extra!r}")
        raise InterfaceValidationError(
            f"policy {spec.name!r} produced commands that do not match its outputs; "
            + ", ".join(details)
            + "."
        )

    for output in spec.outputs:
        command = action.commands[output.target]
        if command.command != output.command:
            raise InterfaceValidationError(
                f"policy {spec.name!r} output {output.target!r} declared command "
                f"{output.command!r}, but produced {command.command!r}."
            )
        if len(command.value) != output.dim:
            raise InterfaceValidationError(
                f"policy {spec.name!r} output {output.target!r} declared dim "
                f"{output.dim}, but produced dim {len(command.value)}."
            )


def _builtin_command_kind_specs() -> tuple[CommandKindSpec, ...]:
    """Return the built-in canonical command kinds."""

    return (
        CommandKindSpec(
            name="joint_position",
            description="Absolute joint position command.",
            allowed_component_types=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="joint_position_delta",
            description="Joint position delta command.",
            allowed_component_types=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="joint_velocity",
            description="Joint velocity command.",
            allowed_component_types=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="cartesian_pose",
            description="End-effector cartesian pose command.",
            allowed_component_types=["arm", "custom"],
        ),
        CommandKindSpec(
            name="cartesian_pose_delta",
            description="End-effector cartesian pose delta command.",
            allowed_component_types=["arm", "custom"],
        ),
        CommandKindSpec(
            name="cartesian_twist",
            description="End-effector cartesian twist command.",
            default_dim=6,
            allowed_component_types=["arm", "custom"],
        ),
        CommandKindSpec(
            name="gripper_position",
            description="Absolute gripper position command.",
            default_dim=1,
            allowed_component_types=["gripper", "custom"],
        ),
        CommandKindSpec(
            name="gripper_position_delta",
            description="Gripper position delta command.",
            default_dim=1,
            allowed_component_types=["gripper", "custom"],
        ),
        CommandKindSpec(
            name="gripper_velocity",
            description="Gripper velocity command.",
            default_dim=1,
            allowed_component_types=["gripper", "custom"],
        ),
        CommandKindSpec(
            name="gripper_open_close",
            description="Binary or scalar open/close gripper command.",
            default_dim=1,
            allowed_component_types=["gripper", "custom"],
        ),
        CommandKindSpec(
            name="hand_joint_position",
            description="Absolute dexterous-hand joint position command.",
            allowed_component_types=["hand", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="hand_joint_position_delta",
            description="Dexterous-hand joint position delta command.",
            allowed_component_types=["hand", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="eef_activation",
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
    "Action",
    "COMMAND_KIND_REGISTRY",
    "CUSTOM_COMMAND_KIND_PREFIX",
    "ComponentSpec",
    "Command",
    "CommandKindSpec",
    "Frame",
    "KNOWN_COMPONENT_TYPES",
    "PolicyOutputSpec",
    "PolicySpec",
    "RobotSpec",
    "ensure_action_matches_policy_spec",
    "ensure_action_supported_by_robot",
    "get_command_kind_spec",
    "is_custom_command_kind_name",
    "is_known_command_kind",
    "register_command_kind",
    "validate_action",
    "validate_command",
    "validate_component_spec",
    "validate_frame",
    "validate_policy_output_spec",
    "validate_policy_spec",
    "validate_robot_spec",
]
