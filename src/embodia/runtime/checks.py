"""Runtime validation helpers and acceptance checks.

This module intentionally keeps runtime acceptance logic thin. Structural
schema validation lives in :mod:`embodia.core.schema`; the functions here focus
on calling user objects safely and verifying that runtime behavior matches the
declared specs.
"""

from __future__ import annotations

import inspect
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.modalities import images, state, task
from ..core.schema import (
    Action,
    Frame,
    ModelSpec,
    RobotSpec,
    ensure_action_matches_model_spec,
    ensure_action_supported_by_robot,
    validate_action,
    validate_command,
    validate_control_group_spec,
    validate_frame,
    validate_model_output_spec,
    validate_model_spec,
    validate_robot_spec,
)


def _object_label(obj: object) -> str:
    """Return a helpful label for error messages."""

    return f"{type(obj).__name__} instance"


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

    task_problem = task.pair_problem(
        available_keys=robot_spec.task_keys,
        required_keys=model_spec.required_task_keys,
    )
    if task_problem is not None:
        problems.append(task_problem)

    for output in model_spec.outputs:
        group = robot_spec.get_group(output.target)
        if group is None:
            problems.append(
                f"robot is missing required control group {output.target!r}."
            )
            continue
        if output.command_kind not in group.supported_command_kinds:
            problems.append(
                f"control group {output.target!r} does not support model output "
                f"command kind {output.command_kind!r}; supported command kinds: "
                f"{group.supported_command_kinds!r}."
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

    if not call_observe:
        return

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
