"""Runtime validation helpers and acceptance checks.

This module intentionally keeps runtime acceptance logic thin. Structural
schema validation lives in :mod:`embodia.core.schema`; the functions here focus
on calling user objects safely and verifying that runtime behavior matches the
declared specs.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
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
    validate_component_spec,
    validate_command,
    validate_frame,
    validate_model_output_spec,
    validate_model_spec,
    validate_robot_spec,
)
from ._dispatch import (
    MODEL_GET_SPEC_METHODS,
    MODEL_INFER_CHUNK_METHODS,
    MODEL_INFER_METHODS,
    MODEL_RESET_METHODS,
    ROBOT_ACT_METHODS,
    ROBOT_GET_SPEC_METHODS,
    ROBOT_OBSERVE_METHODS,
    ROBOT_RESET_METHODS,
    format_method_options,
    resolve_callable_method,
)


def _object_label(obj: object) -> str:
    """Return a helpful label for error messages."""

    return f"{type(obj).__name__} instance"


def _require_method(
    obj: object,
    method_names: tuple[str, ...],
) -> tuple[Any, str]:
    """Fetch the first required callable method from a priority list."""

    method, resolved_name = resolve_callable_method(obj, method_names)
    if callable(method) and resolved_name is not None:
        return method, resolved_name

    raise InterfaceValidationError(
        f"{_object_label(obj)} is missing required method "
        f"{format_method_options(method_names)}."
    )


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


def _single_step_chunk_request() -> object:
    """Build one minimal request object for chunk-model acceptance checks."""

    return SimpleNamespace(
        request_step=0,
        request_time_s=0.0,
        history_start=0,
        history_end=0,
        active_chunk_length=0,
        remaining_steps=0,
        overlap_steps=0,
        latency_steps=0,
        request_trigger_steps=0,
        plan_start_step=0,
        history_actions=[],
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

    task_problem = task.pair_problem(
        available_keys=robot_spec.task_keys,
        required_keys=model_spec.required_task_keys,
    )
    if task_problem is not None:
        problems.append(task_problem)

    for output in model_spec.outputs:
        component = robot_spec.get_component(output.target)
        if component is None:
            problems.append(
                f"robot is missing required component {output.target!r}."
            )
            continue
        if output.command_kind not in component.supported_command_kinds:
            problems.append(
                f"component {output.target!r} does not support model output "
                f"command kind {output.command_kind!r}; supported command kinds: "
                f"{component.supported_command_kinds!r}."
            )
        if output.dim != component.dof:
            problems.append(
                f"component {output.target!r} has dof={component.dof}, but model "
                f"declares dim={output.dim}."
            )

    return problems


def check_robot(robot: object, *, call_observe: bool = True) -> None:
    """Runtime-check whether an object is a compatible robot implementation."""

    get_spec, get_spec_name = _require_method(robot, ROBOT_GET_SPEC_METHODS)
    observe, observe_name = _require_method(robot, ROBOT_OBSERVE_METHODS)
    act, act_name = _require_method(robot, ROBOT_ACT_METHODS)
    reset, reset_name = _require_method(robot, ROBOT_RESET_METHODS)

    _ensure_signature_accepts(get_spec, get_spec_name)
    _ensure_signature_accepts(observe, observe_name)
    _ensure_signature_accepts(act, act_name, object())
    _ensure_signature_accepts(reset, reset_name)

    spec = _call_method(get_spec, robot, get_spec_name)
    if not isinstance(spec, RobotSpec):
        raise InterfaceValidationError(
            f"{_object_label(robot)} {get_spec_name}() must return RobotSpec, "
            f"got {type(spec).__name__}."
        )
    validate_robot_spec(spec)

    if not call_observe:
        return

    frame = _call_method(observe, robot, observe_name)
    if not isinstance(frame, Frame):
        raise InterfaceValidationError(
            f"{_object_label(robot)} {observe_name}() must return Frame, "
            f"got {type(frame).__name__}."
        )
    validate_frame(frame)
    images.ensure_frame_keys(frame, spec.image_keys, owner_label="robot")
    state.ensure_frame_keys(frame, spec.all_state_keys(), owner_label="robot")
    task.ensure_frame_keys(frame, spec.task_keys, owner_label="robot")


def check_model(model: object, *, sample_frame: Frame | None = None) -> None:
    """Runtime-check whether an object is a compatible model implementation."""

    get_spec, get_spec_name = _require_method(model, MODEL_GET_SPEC_METHODS)
    reset, reset_name = _require_method(model, MODEL_RESET_METHODS)
    infer, infer_name = resolve_callable_method(model, MODEL_INFER_METHODS)
    infer_chunk, infer_chunk_name = resolve_callable_method(
        model,
        MODEL_INFER_CHUNK_METHODS,
    )
    if not callable(infer) and not callable(infer_chunk):
        raise InterfaceValidationError(
            f"{_object_label(model)} must expose "
            f"{format_method_options(MODEL_INFER_METHODS)} or "
            f"{format_method_options(MODEL_INFER_CHUNK_METHODS)}."
        )

    _ensure_signature_accepts(get_spec, get_spec_name)
    _ensure_signature_accepts(reset, reset_name)
    if callable(infer) and infer_name is not None:
        _ensure_signature_accepts(infer, infer_name, object())
    if callable(infer_chunk) and infer_chunk_name is not None:
        _ensure_signature_accepts(infer_chunk, infer_chunk_name, object(), object())

    spec = _call_method(get_spec, model, get_spec_name)
    if not isinstance(spec, ModelSpec):
        raise InterfaceValidationError(
            f"{_object_label(model)} {get_spec_name}() must return ModelSpec, "
            f"got {type(spec).__name__}."
        )
    validate_model_spec(spec)

    reset_result = _call_method(reset, model, reset_name)
    if reset_result is not None:
        raise InterfaceValidationError(
            f"{_object_label(model)} {reset_name}() must return None, "
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

    if callable(infer) and infer_name is not None:
        action = _call_method(infer, model, infer_name, sample_frame)
        if not isinstance(action, Action):
            raise InterfaceValidationError(
                f"{_object_label(model)} {infer_name}(frame) must return Action, "
                f"got {type(action).__name__}."
            )
        validate_action(action)
        ensure_action_matches_model_spec(action, spec)
        return

    assert callable(infer_chunk) and infer_chunk_name is not None
    plan = _call_method(
        infer_chunk,
        model,
        infer_chunk_name,
        sample_frame,
        _single_step_chunk_request(),
    )
    if isinstance(plan, Action):
        actions = [plan]
    elif isinstance(plan, list):
        actions = plan
    else:
        raise InterfaceValidationError(
            f"{_object_label(model)} {infer_chunk_name}(frame, request) must return "
            f"list[Action], got {type(plan).__name__}."
        )
    if not actions:
        raise InterfaceValidationError(
            f"{_object_label(model)} {infer_chunk_name}(frame, request) must not "
            "return an empty chunk."
        )
    for index, action in enumerate(actions):
        if not isinstance(action, Action):
            raise InterfaceValidationError(
                f"{_object_label(model)} {infer_chunk_name}(frame, request) returned "
                f"non-Action item at index {index}: {type(action).__name__}."
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

    robot_get_spec, robot_get_spec_name = _require_method(robot, ROBOT_GET_SPEC_METHODS)
    model_get_spec, model_get_spec_name = _require_method(model, MODEL_GET_SPEC_METHODS)

    robot_spec = _call_method(robot_get_spec, robot, robot_get_spec_name)
    model_spec = _call_method(model_get_spec, model, model_get_spec_name)

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
    "validate_component_spec",
    "validate_command",
    "validate_frame",
    "validate_model_output_spec",
    "validate_model_spec",
    "validate_robot_spec",
]
