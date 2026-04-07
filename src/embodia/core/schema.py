"""Lightweight shared schema for robot/model runtime interaction.

This module is intentionally small. It standardizes the data objects that sit
between robot adapters and model adapters without turning embodia into a full
robotics middleware stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

from .errors import InterfaceValidationError

KNOWN_COMPONENT_KINDS: tuple[str, ...] = (
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


def _ensure_positive_int(value: object, field_name: str) -> int:
    """Validate one positive integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise InterfaceValidationError(
            f"{field_name} must be an int, got {type(value).__name__}."
        )
    if value <= 0:
        raise InterfaceValidationError(f"{field_name} must be > 0, got {value!r}.")
    return value


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

    timestamp_ns: int
    images: dict[str, Any]
    state: dict[str, Any]
    task: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    sequence_id: int | None = None


@dataclass(slots=True)
class Command:
    """One command for one robot component.

    ``target`` names the component, for example ``"left_arm"`` or
    ``"right_hand"``. ``kind`` is the main semantic identifier and is the
    primary hook into the command-kind registry.
    """

    target: str
    kind: str
    value: list[float]
    ref_frame: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Action:
    """One control step containing one or more commands."""

    commands: list[Command]
    dt: float = 0.1
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def single(
        cls,
        *,
        target: str,
        kind: str,
        value: list[float],
        dt: float = 0.1,
        ref_frame: str | None = None,
        command_meta: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Action:
        """Build an action containing exactly one command."""

        return cls(
            commands=[
                Command(
                    target=target,
                    kind=kind,
                    value=list(value),
                    ref_frame=ref_frame,
                    meta={} if command_meta is None else dict(command_meta),
                )
            ],
            dt=dt,
            meta={} if meta is None else dict(meta),
        )

    def get_command(self, target: str) -> Command | None:
        """Return the command for ``target`` when present."""

        for command in self.commands:
            if command.target == target:
                return command
        return None


@dataclass(slots=True)
class ComponentSpec:
    """Description of one controllable robot component."""

    name: str
    kind: str
    dof: int
    supported_command_kinds: list[str]
    state_keys: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def supports_command_kind(self, command_kind: str) -> bool:
        """Return whether the component accepts ``command_kind``."""

        return command_kind in self.supported_command_kinds


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

    def all_supported_command_kinds(self) -> list[str]:
        """Return unique command kinds supported across all components."""

        seen: set[str] = set()
        result: list[str] = []
        for component in self.components:
            for command_kind in component.supported_command_kinds:
                if command_kind not in seen:
                    seen.add(command_kind)
                    result.append(command_kind)
        return result

    def all_state_keys(self) -> list[str]:
        """Return the unique state keys exposed across all components."""

        seen: set[str] = set()
        result: list[str] = []
        for component in self.components:
            for key in component.state_keys:
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        return result


@dataclass(slots=True)
class ModelOutputSpec:
    """Description of one model output command slot."""

    target: str
    command_kind: str
    dim: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelSpec:
    """Description of one model's required inputs and emitted commands."""

    name: str
    required_image_keys: list[str]
    required_state_keys: list[str]
    required_task_keys: list[str] = field(default_factory=list)
    outputs: list[ModelOutputSpec] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def get_output(self, target: str) -> ModelOutputSpec | None:
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
    allowed_component_kinds: list[str] = field(default_factory=list)
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
    spec.allowed_component_kinds = _ensure_string_list(
        spec.allowed_component_kinds,
        "command_kind_spec.allowed_component_kinds",
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

    _ensure_non_empty_string(cmd.target, "command.target")
    kind = _validate_command_kind_name(
        cmd.kind,
        "command.kind",
        allow_unregistered_custom=True,
    )
    if not isinstance(cmd.value, list):
        raise InterfaceValidationError(
            f"command.value must be a list[float], got {type(cmd.value).__name__}."
        )
    for index, number in enumerate(cmd.value):
        _ensure_real_number(number, f"command.value[{index}]")
    if cmd.ref_frame is not None:
        _ensure_non_empty_string(cmd.ref_frame, "command.ref_frame")
    _ensure_string_key_dict(cmd.meta, "command.meta")

    if not is_known_command_kind(kind):
        return

    spec = get_command_kind_spec(kind)
    if spec.requires_ref_frame and cmd.ref_frame is None:
        raise InterfaceValidationError(
            f"command.kind {kind!r} requires command.ref_frame."
        )
    if spec.default_dim is not None and len(cmd.value) != spec.default_dim:
        raise InterfaceValidationError(
            f"command.kind {kind!r} expects dim={spec.default_dim}, got "
            f"{len(cmd.value)}."
        )


def validate_action(action: Action) -> None:
    """Validate one action."""

    if not isinstance(action, Action):
        raise InterfaceValidationError(
            f"action must be an Action instance, got {type(action).__name__}."
        )
    if not isinstance(action.commands, list):
        raise InterfaceValidationError(
            f"action.commands must be a list[Command], got "
            f"{type(action.commands).__name__}."
        )
    if not action.commands:
        raise InterfaceValidationError("action.commands must not be empty.")

    seen_targets: set[str] = set()
    for index, command in enumerate(action.commands):
        try:
            validate_command(command)
        except InterfaceValidationError as exc:
            raise InterfaceValidationError(
                f"invalid action.commands[{index}]: {exc}"
            ) from exc
        if command.target in seen_targets:
            raise InterfaceValidationError(
                f"action.commands contains duplicate target {command.target!r}."
            )
        seen_targets.add(command.target)

    dt = _ensure_real_number(action.dt, "action.dt")
    if dt <= 0.0:
        raise InterfaceValidationError("action.dt must be > 0.")
    _ensure_string_key_dict(action.meta, "action.meta")


def validate_component_spec(spec: ComponentSpec) -> None:
    """Validate one robot component spec."""

    if not isinstance(spec, ComponentSpec):
        raise InterfaceValidationError(
            "spec must be a ComponentSpec instance, got "
            f"{type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.name, "component_spec.name")
    _ensure_non_empty_string(spec.kind, "component_spec.kind")
    _ensure_positive_int(spec.dof, "component_spec.dof")
    supported = _ensure_string_list(
        spec.supported_command_kinds,
        "component_spec.supported_command_kinds",
        allow_empty=False,
    )
    _ensure_string_list(
        spec.state_keys,
        "component_spec.state_keys",
        allow_empty=True,
    )
    _ensure_string_key_dict(spec.meta, "component_spec.meta")

    for index, command_kind in enumerate(supported):
        kind_name = _validate_command_kind_name(
            command_kind,
            f"component_spec.supported_command_kinds[{index}]",
            allow_unregistered_custom=True,
        )
        if not is_known_command_kind(kind_name):
            continue
        kind_spec = get_command_kind_spec(kind_name)
        if (
            kind_spec.allowed_component_kinds
            and spec.kind not in kind_spec.allowed_component_kinds
        ):
            raise InterfaceValidationError(
                f"component_spec {spec.name!r} has kind {spec.kind!r}, which "
                f"is incompatible with command kind {kind_name!r}; allowed component "
                f"kinds: {kind_spec.allowed_component_kinds!r}."
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


def validate_model_output_spec(spec: ModelOutputSpec) -> None:
    """Validate one model-output description."""

    if not isinstance(spec, ModelOutputSpec):
        raise InterfaceValidationError(
            "spec must be a ModelOutputSpec instance, got "
            f"{type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.target, "model_output_spec.target")
    command_kind = _validate_command_kind_name(
        spec.command_kind,
        "model_output_spec.command_kind",
        allow_unregistered_custom=True,
    )
    _ensure_positive_int(spec.dim, "model_output_spec.dim")
    _ensure_string_key_dict(spec.meta, "model_output_spec.meta")

    if not is_known_command_kind(command_kind):
        return

    kind_spec = get_command_kind_spec(command_kind)
    if kind_spec.default_dim is not None and spec.dim != kind_spec.default_dim:
        raise InterfaceValidationError(
            f"model_output_spec.command_kind {command_kind!r} expects dim="
            f"{kind_spec.default_dim}, got {spec.dim}."
        )


def validate_model_spec(spec: ModelSpec) -> None:
    """Validate one model interface spec."""

    if not isinstance(spec, ModelSpec):
        raise InterfaceValidationError(
            f"spec must be a ModelSpec instance, got {type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.name, "model_spec.name")
    _ensure_string_list(
        spec.required_image_keys,
        "model_spec.required_image_keys",
        allow_empty=True,
    )
    _ensure_string_list(
        spec.required_state_keys,
        "model_spec.required_state_keys",
        allow_empty=True,
    )
    _ensure_string_list(
        spec.required_task_keys,
        "model_spec.required_task_keys",
        allow_empty=True,
    )
    _ensure_string_key_dict(spec.meta, "model_spec.meta")
    if not isinstance(spec.outputs, list):
        raise InterfaceValidationError(
            f"model_spec.outputs must be a list, got {type(spec.outputs).__name__}."
        )
    if not spec.outputs:
        raise InterfaceValidationError("model_spec.outputs must not be empty.")

    seen_targets: set[str] = set()
    for index, output in enumerate(spec.outputs):
        try:
            validate_model_output_spec(output)
        except InterfaceValidationError as exc:
            raise InterfaceValidationError(
                f"invalid model_spec.outputs[{index}]: {exc}"
            ) from exc
        if output.target in seen_targets:
            raise InterfaceValidationError(
                f"model_spec.outputs contains duplicate target {output.target!r}."
            )
        seen_targets.add(output.target)


def ensure_action_supported_by_robot(action: Action, spec: RobotSpec) -> None:
    """Ensure an action is compatible with one robot spec."""

    validate_action(action)
    validate_robot_spec(spec)

    for command in action.commands:
        component = spec.get_component(command.target)
        if component is None:
            raise InterfaceValidationError(
                f"robot {spec.name!r} does not define component "
                f"{command.target!r}."
            )
        if command.kind not in component.supported_command_kinds:
            raise InterfaceValidationError(
                f"robot {spec.name!r} component {component.name!r} does not support "
                f"command kind {command.kind!r}; supported kinds: "
                f"{component.supported_command_kinds!r}."
            )
        if not is_known_command_kind(command.kind):
            continue
        kind_spec = get_command_kind_spec(command.kind)
        if (
            kind_spec.allowed_component_kinds
            and component.kind not in kind_spec.allowed_component_kinds
        ):
            raise InterfaceValidationError(
                f"command kind {command.kind!r} is not allowed for robot component "
                f"{component.name!r} of kind {component.kind!r}; allowed component "
                f"kinds: {kind_spec.allowed_component_kinds!r}."
            )
        if _kind_uses_component_dof(kind_spec) and len(command.value) != component.dof:
            raise InterfaceValidationError(
                f"command kind {command.kind!r} for robot component "
                f"{component.name!r} must match component dof={component.dof}, got "
                f"dim={len(command.value)}."
            )


def ensure_action_matches_model_spec(action: Action, spec: ModelSpec) -> None:
    """Ensure an action matches one model's declared outputs."""

    validate_action(action)
    validate_model_spec(spec)

    command_targets = {command.target for command in action.commands}
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
            f"model {spec.name!r} produced commands that do not match its outputs; "
            + ", ".join(details)
            + "."
        )

    for output in spec.outputs:
        command = next(
            candidate for candidate in action.commands if candidate.target == output.target
        )
        if command.kind != output.command_kind:
            raise InterfaceValidationError(
                f"model {spec.name!r} output {output.target!r} declared command "
                f"kind {output.command_kind!r}, but produced {command.kind!r}."
            )
        if len(command.value) != output.dim:
            raise InterfaceValidationError(
                f"model {spec.name!r} output {output.target!r} declared dim "
                f"{output.dim}, but produced dim {len(command.value)}."
            )


def _builtin_command_kind_specs() -> tuple[CommandKindSpec, ...]:
    """Return the built-in canonical command kinds."""

    return (
        CommandKindSpec(
            name="joint_position",
            description="Absolute joint position command.",
            allowed_component_kinds=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="joint_position_delta",
            description="Joint position delta command.",
            allowed_component_kinds=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="joint_velocity",
            description="Joint velocity command.",
            allowed_component_kinds=["arm", "hand", "base", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="cartesian_pose",
            description="End-effector cartesian pose command.",
            allowed_component_kinds=["arm", "custom"],
        ),
        CommandKindSpec(
            name="cartesian_pose_delta",
            description="End-effector cartesian pose delta command.",
            allowed_component_kinds=["arm", "custom"],
        ),
        CommandKindSpec(
            name="cartesian_twist",
            description="End-effector cartesian twist command.",
            default_dim=6,
            allowed_component_kinds=["arm", "custom"],
        ),
        CommandKindSpec(
            name="gripper_position",
            description="Absolute gripper position command.",
            default_dim=1,
            allowed_component_kinds=["gripper", "custom"],
        ),
        CommandKindSpec(
            name="gripper_position_delta",
            description="Gripper position delta command.",
            default_dim=1,
            allowed_component_kinds=["gripper", "custom"],
        ),
        CommandKindSpec(
            name="gripper_velocity",
            description="Gripper velocity command.",
            default_dim=1,
            allowed_component_kinds=["gripper", "custom"],
        ),
        CommandKindSpec(
            name="gripper_open_close",
            description="Binary or scalar open/close gripper command.",
            default_dim=1,
            allowed_component_kinds=["gripper", "custom"],
        ),
        CommandKindSpec(
            name="hand_joint_position",
            description="Absolute dexterous-hand joint position command.",
            allowed_component_kinds=["hand", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="hand_joint_position_delta",
            description="Dexterous-hand joint position delta command.",
            allowed_component_kinds=["hand", "custom"],
            meta={_USES_COMPONENT_DOF_META_KEY: True},
        ),
        CommandKindSpec(
            name="eef_activation",
            description="Generic end-effector activation command.",
            default_dim=1,
            allowed_component_kinds=["gripper", "hand", "suction", "custom"],
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
    "KNOWN_COMPONENT_KINDS",
    "ModelOutputSpec",
    "ModelSpec",
    "RobotSpec",
    "ensure_action_matches_model_spec",
    "ensure_action_supported_by_robot",
    "get_command_kind_spec",
    "is_custom_command_kind_name",
    "is_known_command_kind",
    "register_command_kind",
    "validate_action",
    "validate_command",
    "validate_component_spec",
    "validate_frame",
    "validate_model_output_spec",
    "validate_model_spec",
    "validate_robot_spec",
]
