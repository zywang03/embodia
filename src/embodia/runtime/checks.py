"""Runtime validation helpers and acceptance checks."""

from __future__ import annotations

import inspect
import math
from numbers import Real
from typing import Any

from ..core.modalities import images, state, task
from ..core.errors import InterfaceValidationError
from ..core.schema import (
    Action,
    Command,
    ControlGroupSpec,
    Frame,
    ModelOutputSpec,
    ModelSpec,
    RobotSpec,
)


def _object_label(obj: object) -> str:
    """Return a helpful label for error messages."""

    return f"{type(obj).__name__} instance"


def _ensure_non_empty_string(value: object, field_name: str) -> str:
    """Validate that ``value`` is a non-empty string."""

    if not isinstance(value, str):
        raise InterfaceValidationError(
            f"{field_name} must be a string, got {type(value).__name__}."
        )
    if not value.strip():
        raise InterfaceValidationError(f"{field_name} must not be empty.")
    return value


def _ensure_string_list(
    value: object,
    field_name: str,
    *,
    allow_empty: bool,
) -> list[str]:
    """Validate ``list[str]`` with duplicate detection."""

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


def _ensure_string_key_dict(value: object, field_name: str) -> dict[str, Any]:
    """Validate a ``dict[str, Any]`` field."""

    if not isinstance(value, dict):
        raise InterfaceValidationError(
            f"{field_name} must be a dict[str, Any], got {type(value).__name__}."
        )
    for key in value:
        if not isinstance(key, str):
            raise InterfaceValidationError(
                f"{field_name} keys must be strings, got key {key!r} "
                f"of type {type(key).__name__}."
            )
    return value


def _ensure_real_number(value: object, field_name: str) -> float:
    """Validate a finite real number while rejecting ``bool``."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise InterfaceValidationError(
            f"{field_name} must be a real number, got {type(value).__name__}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise InterfaceValidationError(f"{field_name} must be finite, got {value!r}.")
    return number


def _ensure_positive_int(value: object, field_name: str) -> int:
    """Validate a positive integer while rejecting ``bool``."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise InterfaceValidationError(
            f"{field_name} must be an int, got {type(value).__name__}."
        )
    if value <= 0:
        raise InterfaceValidationError(f"{field_name} must be > 0, got {value!r}.")
    return value


def _require_method(obj: object, method_name: str) -> Any:
    """Fetch and validate a required callable attribute."""

    if not hasattr(obj, method_name):
        raise InterfaceValidationError(
            f"{_object_label(obj)} is missing required method {method_name!r}."
        )

    method = getattr(obj, method_name)
    if not callable(method):
        raise InterfaceValidationError(
            f"{_object_label(obj)} attribute {method_name!r} exists but is not callable."
        )
    return method


def _call_method(method: Any, obj: object, method_name: str, *args: object) -> Any:
    """Call a checked method and wrap runtime errors consistently."""

    try:
        return method(*args)
    except Exception as exc:
        raise InterfaceValidationError(
            f"{_object_label(obj)} {method_name}() raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _ensure_signature_accepts(method: Any, method_name: str, *args: object) -> None:
    """Check that a method can be called with the expected runtime arguments."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        raise InterfaceValidationError(
            f"Could not inspect signature of method {method_name!r}: {exc}."
        ) from exc

    try:
        signature.bind(*args)
    except TypeError as exc:
        raise InterfaceValidationError(
            f"Method {method_name!r} has incompatible signature {signature}; "
            f"it must accept {len(args)} runtime argument(s)."
        ) from exc


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


def validate_command(command: Command) -> None:
    """Validate a :class:`Command` instance."""

    if not isinstance(command, Command):
        raise InterfaceValidationError(
            f"command must be a Command instance, got {type(command).__name__}."
        )

    _ensure_non_empty_string(command.target, "command.target")
    _ensure_non_empty_string(command.mode, "command.mode")

    if not isinstance(command.value, list):
        raise InterfaceValidationError(
            f"command.value must be a list[float], got {type(command.value).__name__}."
        )
    for index, number in enumerate(command.value):
        _ensure_real_number(number, f"command.value[{index}]")

    if command.ref_frame is not None:
        _ensure_non_empty_string(command.ref_frame, "command.ref_frame")
    _ensure_string_key_dict(command.meta, "command.meta")


def validate_action(action: Action) -> None:
    """Validate an :class:`Action` instance."""

    if not isinstance(action, Action):
        raise InterfaceValidationError(
            f"action must be an Action instance, got {type(action).__name__}."
        )
    if not isinstance(action.commands, list):
        raise InterfaceValidationError(
            f"action.commands must be a list[Command], got {type(action.commands).__name__}."
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


def validate_control_group_spec(spec: ControlGroupSpec) -> None:
    """Validate a :class:`ControlGroupSpec` instance."""

    if not isinstance(spec, ControlGroupSpec):
        raise InterfaceValidationError(
            "spec must be a ControlGroupSpec instance, got "
            f"{type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.name, "control_group_spec.name")
    _ensure_non_empty_string(spec.kind, "control_group_spec.kind")
    _ensure_positive_int(spec.dof, "control_group_spec.dof")
    _ensure_string_list(
        spec.action_modes,
        "control_group_spec.action_modes",
        allow_empty=False,
    )
    _ensure_string_list(
        spec.state_keys,
        "control_group_spec.state_keys",
        allow_empty=True,
    )
    _ensure_string_key_dict(spec.meta, "control_group_spec.meta")


def validate_robot_spec(spec: RobotSpec) -> None:
    """Validate a :class:`RobotSpec` instance."""

    if not isinstance(spec, RobotSpec):
        raise InterfaceValidationError(
            f"spec must be a RobotSpec instance, got {type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.name, "robot_spec.name")
    _ensure_string_list(spec.image_keys, "robot_spec.image_keys", allow_empty=True)
    _ensure_string_list(spec.task_keys, "robot_spec.task_keys", allow_empty=True)
    _ensure_string_key_dict(spec.meta, "robot_spec.meta")

    if not isinstance(spec.groups, list):
        raise InterfaceValidationError(
            f"robot_spec.groups must be a list, got {type(spec.groups).__name__}."
        )
    if not spec.groups:
        raise InterfaceValidationError("robot_spec.groups must not be empty.")

    seen_names: set[str] = set()
    for index, group in enumerate(spec.groups):
        try:
            validate_control_group_spec(group)
        except InterfaceValidationError as exc:
            raise InterfaceValidationError(
                f"invalid robot_spec.groups[{index}]: {exc}"
            ) from exc
        if group.name in seen_names:
            raise InterfaceValidationError(
                f"robot_spec.groups contains duplicate group {group.name!r}."
            )
        seen_names.add(group.name)


def validate_model_output_spec(spec: ModelOutputSpec) -> None:
    """Validate a :class:`ModelOutputSpec` instance."""

    if not isinstance(spec, ModelOutputSpec):
        raise InterfaceValidationError(
            "spec must be a ModelOutputSpec instance, got "
            f"{type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.target, "model_output_spec.target")
    _ensure_non_empty_string(spec.mode, "model_output_spec.mode")
    _ensure_positive_int(spec.dim, "model_output_spec.dim")
    _ensure_string_key_dict(spec.meta, "model_output_spec.meta")


def validate_model_spec(spec: ModelSpec) -> None:
    """Validate a :class:`ModelSpec` instance."""

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
        group = spec.get_group(command.target)
        if group is None:
            raise InterfaceValidationError(
                f"robot {spec.name!r} does not define control group "
                f"{command.target!r}."
            )
        if command.mode not in group.action_modes:
            raise InterfaceValidationError(
                f"robot {spec.name!r} group {group.name!r} does not support mode "
                f"{command.mode!r}; supported modes: {group.action_modes!r}."
            )
        if len(command.value) != group.dof:
            raise InterfaceValidationError(
                f"robot {spec.name!r} group {group.name!r} expects dof={group.dof}, "
                f"but action command has dim={len(command.value)}."
            )


def ensure_action_matches_model_spec(action: Action, spec: ModelSpec) -> None:
    """Ensure an action matches one model's declared output structure."""

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
        if command.mode != output.mode:
            raise InterfaceValidationError(
                f"model {spec.name!r} output {output.target!r} declared mode "
                f"{output.mode!r}, but produced {command.mode!r}."
            )
        if len(command.value) != output.dim:
            raise InterfaceValidationError(
                f"model {spec.name!r} output {output.target!r} declared dim "
                f"{output.dim}, but produced dim {len(command.value)}."
            )


def _pair_problems(robot_spec: RobotSpec, model_spec: ModelSpec) -> list[str]:
    """Return compatibility problems between a robot spec and a model spec."""

    problems: list[str] = []

    image_problem = images.pair_problem(
        available_keys=robot_spec.image_keys,
        required_keys=model_spec.required_image_keys,
    )
    if image_problem is not None:
        problems.append(image_problem)

    state_problem = state.pair_problem(
        available_keys=robot_spec.all_state_keys(),
        required_keys=model_spec.required_state_keys,
    )
    if state_problem is not None:
        problems.append(state_problem)

    for output in model_spec.outputs:
        group = robot_spec.get_group(output.target)
        if group is None:
            problems.append(
                f"robot is missing required control group {output.target!r}."
            )
            continue
        if output.mode not in group.action_modes:
            problems.append(
                f"control group {output.target!r} does not support model output "
                f"mode {output.mode!r}; supported modes: {group.action_modes!r}."
            )
        if output.dim != group.dof:
            problems.append(
                f"control group {output.target!r} has dof={group.dof}, but model "
                f"declares dim={output.dim}."
            )

    return problems


def check_robot(robot: object, *, call_observe: bool = True) -> None:
    """Runtime-check whether an object is a compatible robot implementation."""

    get_spec = _require_method(robot, "get_spec")
    observe = _require_method(robot, "observe")
    act = _require_method(robot, "act")
    reset = _require_method(robot, "reset")

    _ensure_signature_accepts(get_spec, "get_spec")
    _ensure_signature_accepts(observe, "observe")
    _ensure_signature_accepts(act, "act", object())
    _ensure_signature_accepts(reset, "reset")

    spec = _call_method(get_spec, robot, "get_spec")
    if not isinstance(spec, RobotSpec):
        raise InterfaceValidationError(
            f"{_object_label(robot)} get_spec() must return RobotSpec, "
            f"got {type(spec).__name__}."
        )
    validate_robot_spec(spec)

    if call_observe:
        frame = _call_method(observe, robot, "observe")
        if not isinstance(frame, Frame):
            raise InterfaceValidationError(
                f"{_object_label(robot)} observe() must return Frame, "
                f"got {type(frame).__name__}."
            )
        validate_frame(frame)
        images.ensure_frame_keys(frame, spec.image_keys, owner_label="robot")
        state.ensure_frame_keys(frame, spec.all_state_keys(), owner_label="robot")
        task.ensure_frame_keys(frame, spec.task_keys, owner_label="robot")


def check_model(model: object, *, sample_frame: Frame | None = None) -> None:
    """Runtime-check whether an object is a compatible model implementation."""

    get_spec = _require_method(model, "get_spec")
    reset = _require_method(model, "reset")
    step = _require_method(model, "step")

    _ensure_signature_accepts(get_spec, "get_spec")
    _ensure_signature_accepts(reset, "reset")
    _ensure_signature_accepts(step, "step", object())

    spec = _call_method(get_spec, model, "get_spec")
    if not isinstance(spec, ModelSpec):
        raise InterfaceValidationError(
            f"{_object_label(model)} get_spec() must return ModelSpec, "
            f"got {type(spec).__name__}."
        )
    validate_model_spec(spec)

    reset_result = _call_method(reset, model, "reset")
    if reset_result is not None:
        raise InterfaceValidationError(
            f"{_object_label(model)} reset() must return None, "
            f"got {type(reset_result).__name__}."
        )

    if sample_frame is None:
        return

    validate_frame(sample_frame)
    images.ensure_frame_keys(
        sample_frame,
        spec.required_image_keys,
        owner_label="model",
    )
    state.ensure_frame_keys(
        sample_frame,
        spec.required_state_keys,
        owner_label="model",
    )
    task.ensure_frame_keys(
        sample_frame,
        spec.required_task_keys,
        owner_label="model",
    )

    action = _call_method(step, model, "step", sample_frame)
    if not isinstance(action, Action):
        raise InterfaceValidationError(
            f"{_object_label(model)} step(frame) must return Action, "
            f"got {type(action).__name__}."
        )
    validate_action(action)
    ensure_action_matches_model_spec(action, spec)


def check_pair(
    robot: object,
    model: object,
    *,
    sample_frame: Frame | None = None,
) -> None:
    """Validate that a robot and a model are individually valid and compatible."""

    check_robot(robot, call_observe=sample_frame is None)
    check_model(model, sample_frame=sample_frame)

    robot_spec = robot.get_spec()
    model_spec = model.get_spec()

    validate_robot_spec(robot_spec)
    validate_model_spec(model_spec)

    problems = _pair_problems(robot_spec, model_spec)
    if problems:
        raise InterfaceValidationError(
            "Robot/model pair is incompatible:\n- " + "\n- ".join(problems)
        )


__all__ = [
    "InterfaceValidationError",
    "check_model",
    "check_pair",
    "check_robot",
    "ensure_action_matches_model_spec",
    "ensure_action_supported_by_robot",
    "validate_action",
    "validate_command",
    "validate_control_group_spec",
    "validate_frame",
    "validate_model_output_spec",
    "validate_model_spec",
    "validate_robot_spec",
]
