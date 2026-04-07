"""Runtime validation helpers and acceptance checks."""

from __future__ import annotations

import inspect
import math
from numbers import Real
from typing import Any

from ..core.modalities import action_modes, images, state
from ..core.errors import InterfaceValidationError
from ..core.schema import (
    Action,
    Frame,
    KNOWN_ACTION_MODES,
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
        item_name = f"{field_name}[{index}]"
        text = _ensure_non_empty_string(item, item_name)
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

    if frame.task is not None:
        _ensure_string_key_dict(frame.task, "frame.task")
    if frame.meta is not None:
        _ensure_string_key_dict(frame.meta, "frame.meta")


def validate_action(action: Action) -> None:
    """Validate an :class:`Action` instance."""

    if not isinstance(action, Action):
        raise InterfaceValidationError(
            f"action must be an Action instance, got {type(action).__name__}."
        )
    if action.mode not in KNOWN_ACTION_MODES:
        raise InterfaceValidationError(
            f"action.mode must be one of {KNOWN_ACTION_MODES}, got {action.mode!r}."
        )
    if not isinstance(action.value, list):
        raise InterfaceValidationError(
            f"action.value must be a list[float], got {type(action.value).__name__}."
        )
    for index, number in enumerate(action.value):
        _ensure_real_number(number, f"action.value[{index}]")

    if action.gripper is not None:
        _ensure_real_number(action.gripper, "action.gripper")
    if action.ref_frame is not None:
        _ensure_non_empty_string(action.ref_frame, "action.ref_frame")
    dt = _ensure_real_number(action.dt, "action.dt")
    if dt <= 0.0:
        raise InterfaceValidationError("action.dt must be > 0.")


def validate_robot_spec(spec: RobotSpec) -> None:
    """Validate a :class:`RobotSpec` instance."""

    if not isinstance(spec, RobotSpec):
        raise InterfaceValidationError(
            f"spec must be a RobotSpec instance, got {type(spec).__name__}."
        )

    _ensure_non_empty_string(spec.name, "robot_spec.name")
    _ensure_string_list(spec.action_modes, "robot_spec.action_modes", allow_empty=False)
    _ensure_string_list(spec.image_keys, "robot_spec.image_keys", allow_empty=True)
    _ensure_string_list(spec.state_keys, "robot_spec.state_keys", allow_empty=True)


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
    _ensure_non_empty_string(spec.output_action_mode, "model_spec.output_action_mode")


def _pair_problems(robot_spec: RobotSpec, model_spec: ModelSpec) -> list[str]:
    """Return compatibility problems between a robot spec and a model spec."""

    problems: list[str] = []

    action_mode_problem = action_modes.pair_problem(
        supported_modes=robot_spec.action_modes,
        output_mode=model_spec.output_action_mode,
    )
    if action_mode_problem is not None:
        problems.append(action_mode_problem)

    image_problem = images.pair_problem(
        available_keys=robot_spec.image_keys,
        required_keys=model_spec.required_image_keys,
    )
    if image_problem is not None:
        problems.append(image_problem)

    state_problem = state.pair_problem(
        available_keys=robot_spec.state_keys,
        required_keys=model_spec.required_state_keys,
    )
    if state_problem is not None:
        problems.append(state_problem)

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

    action = _call_method(step, model, "step", sample_frame)
    if not isinstance(action, Action):
        raise InterfaceValidationError(
            f"{_object_label(model)} step(frame) must return Action, "
            f"got {type(action).__name__}."
        )
    validate_action(action)


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
    "validate_action",
    "validate_frame",
    "validate_model_spec",
    "validate_robot_spec",
]
