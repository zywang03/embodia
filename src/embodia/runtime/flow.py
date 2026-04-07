"""Helpers for the inference side of embodia's unified runtime data flow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from ..core.transform import coerce_action, coerce_frame
from ._dispatch import (
    MODEL_INFER_CHUNK_METHODS,
    MODEL_INFER_METHODS,
    MODEL_RESET_METHODS,
    ROBOT_ACT_METHODS,
    ROBOT_HAS_REMOTE_POLICY_METHODS,
    ROBOT_OBSERVE_METHODS,
    ROBOT_REMOTE_ACTION_METHODS,
    format_method_options,
    resolve_callable_method,
)
from .checks import validate_action, validate_frame

ActionSource = Callable[[Frame], Action | Mapping[str, Any] | None]


@dataclass(slots=True)
class StepResult:
    """Result of one normalized robot -> action source -> robot runtime step."""

    frame: Frame
    action: Action


def _as_frame(value: Frame | Mapping[str, Any]) -> Frame:
    """Return a frame object without copying when already standardized."""

    if isinstance(value, Frame):
        return value
    return coerce_frame(value)


def _as_action(value: Action | Mapping[str, Any]) -> Action:
    """Return an action object without copying when already standardized."""

    if isinstance(value, Action):
        return value
    return coerce_action(value)


def _call_action_fn(action_fn: ActionSource, frame: Frame) -> Action | Mapping[str, Any]:
    """Run one external action source with clear runtime errors."""

    try:
        raw_action = action_fn(frame)
    except Exception as exc:
        raise InterfaceValidationError(
            f"action_fn(frame) raised {type(exc).__name__}: {exc}"
        ) from exc

    if raw_action is None:
        raise InterfaceValidationError(
            "action_fn(frame) must return an action-like value, got None."
        )
    return raw_action


def _single_step_chunk_request() -> object:
    """Build one minimal request object for chunk-to-step fallback."""

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


def _first_action_from_chunk_call(
    infer_chunk: Callable[[Frame, object], object],
    frame: Frame,
) -> Action:
    """Call one chunk-producing source and return the first action."""

    try:
        raw_plan = infer_chunk(frame, _single_step_chunk_request())
    except Exception as exc:
        raise InterfaceValidationError(
            f"infer_chunk(frame, request) raised {type(exc).__name__}: {exc}"
        ) from exc

    if isinstance(raw_plan, (Action, Mapping)):
        action = _as_action(raw_plan)
        validate_action(action)
        return action

    if not isinstance(raw_plan, list) or not raw_plan:
        raise InterfaceValidationError(
            "infer_chunk(frame, request) must return a non-empty action chunk."
        )

    action = _as_action(raw_plan[0])
    validate_action(action)
    return action


def _resolve_action_source(
    source: object | None,
    action_fn: ActionSource | None,
    *,
    robot: object | None = None,
) -> tuple[ActionSource, bool]:
    """Resolve one action source into a callable and whether it supports reset."""

    if source is not None and action_fn is not None:
        raise InterfaceValidationError(
            "run_step() accepts either a model/callable source as the second "
            "argument or action_fn=..., not both."
        )

    if source is None:
        if action_fn is None:
            request_remote, _ = resolve_callable_method(
                robot,
                ROBOT_REMOTE_ACTION_METHODS,
            )
            has_remote, _ = resolve_callable_method(
                robot,
                ROBOT_HAS_REMOTE_POLICY_METHODS,
            )
            if callable(request_remote) and callable(has_remote):
                try:
                    enabled = bool(has_remote())
                except Exception as exc:
                    raise InterfaceValidationError(
                        f"{type(robot).__name__}.has_remote_policy() raised "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if enabled:
                    return request_remote, False
            raise InterfaceValidationError(
                "run_step() requires a model-like source, action_fn=..., or a "
                "robot with configured remote policy."
            )
        return action_fn, False

    reset_method, _ = resolve_callable_method(source, MODEL_RESET_METHODS)
    can_reset = callable(reset_method)

    infer, _ = resolve_callable_method(source, MODEL_INFER_METHODS)
    if callable(infer):
        return infer, can_reset

    infer_chunk, _ = resolve_callable_method(source, MODEL_INFER_CHUNK_METHODS)
    if callable(infer_chunk):
        return (
            lambda frame, _infer_chunk=infer_chunk: _first_action_from_chunk_call(
                _infer_chunk,
                frame,
            ),
            can_reset,
        )

    if callable(source):
        return source, False

    raise InterfaceValidationError(
        "run_step() source must expose "
        f"{format_method_options(MODEL_INFER_METHODS)}, "
        f"{format_method_options(MODEL_INFER_CHUNK_METHODS)}, or be "
        f"callable(frame), got {type(source).__name__}."
    )


def run_step(
    robot: object,
    model: object | None = None,
    *,
    action_fn: ActionSource | None = None,
    frame: Frame | Mapping[str, Any] | None = None,
    execute_action: bool = True,
    reset_model: bool = False,
    runtime: object | None = None,
    pace_control: bool = True,
) -> StepResult:
    """Run one normalized data-flow step.

    The flow is:

    1. observe a frame from the robot, unless a frame is provided
    2. normalize and validate the frame
    3. get one action from either ``source.infer(frame)`` or ``source(frame)``
    4. normalize and validate the action
    5. optionally execute the action on the robot

    When ``runtime`` is provided, embodia routes the same step request through
    the runtime's optimizer / chunk-scheduling / pacing layer while keeping the
    public call site unchanged.
    """

    if runtime is not None:
        run_with_runtime = getattr(runtime, "_run_step_impl", None)
        if not callable(run_with_runtime):
            raise InterfaceValidationError(
                "runtime must expose _run_step_impl(...), for example an "
                "InferenceRuntime instance."
            )
        return run_with_runtime(
            robot,
            model,
            action_fn=action_fn,
            frame=frame,
            execute_action=execute_action,
            reset_model=reset_model,
            pace_control=pace_control,
        )

    action_source, can_reset = _resolve_action_source(
        model,
        action_fn,
        robot=robot,
    )
    if reset_model and not can_reset:
        raise InterfaceValidationError(
            "reset_model=True requires a source object that exposes "
            f"{format_method_options(MODEL_RESET_METHODS)} together with "
            f"{format_method_options(MODEL_INFER_METHODS)} or "
            f"{format_method_options(MODEL_INFER_CHUNK_METHODS)}, not a bare callable."
        )

    if reset_model:
        assert model is not None
        reset, reset_name = resolve_callable_method(model, MODEL_RESET_METHODS)
        if not callable(reset) or reset_name is None:
            raise InterfaceValidationError(
                "reset_model=True requires a model object exposing "
                f"{format_method_options(MODEL_RESET_METHODS)}."
            )
        try:
            reset()
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(model).__name__}.{reset_name}() raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    observe, observe_name = resolve_callable_method(robot, ROBOT_OBSERVE_METHODS)
    if frame is None:
        if not callable(observe) or observe_name is None:
            raise InterfaceValidationError(
                f"{type(robot).__name__} must expose "
                f"{format_method_options(ROBOT_OBSERVE_METHODS)}."
            )
        try:
            raw_frame = observe()
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(robot).__name__}.{observe_name}() raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    else:
        raw_frame = frame
    normalized_frame = _as_frame(raw_frame)
    validate_frame(normalized_frame)

    raw_action = _call_action_fn(action_source, normalized_frame)
    normalized_action = _as_action(raw_action)
    validate_action(normalized_action)

    if execute_action:
        act, act_name = resolve_callable_method(robot, ROBOT_ACT_METHODS)
        if not callable(act) or act_name is None:
            raise InterfaceValidationError(
                f"{type(robot).__name__} must expose "
                f"{format_method_options(ROBOT_ACT_METHODS)}."
            )
        try:
            act(normalized_action)
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(robot).__name__}.{act_name}(action) raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    return StepResult(frame=normalized_frame, action=normalized_action)


__all__ = ["ActionSource", "StepResult", "run_step"]
