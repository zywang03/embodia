"""Helpers for the inference side of embodia's unified runtime data flow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from ..core.transform import coerce_action, coerce_frame
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
            request_remote = getattr(robot, "_request_remote_policy_action", None)
            if not callable(request_remote):
                request_remote = getattr(robot, "request_remote_policy_action", None)
            has_remote = getattr(robot, "has_remote_policy", None)
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

    step = getattr(source, "step", None)
    if callable(step):
        return step, True

    if callable(source):
        return source, False

    raise InterfaceValidationError(
        f"run_step() source must expose step(frame) or be callable(frame), got "
        f"{type(source).__name__}."
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
    3. get one action from either ``source.step(frame)`` or ``source(frame)``
    4. normalize and validate the action
    5. optionally execute the action on the robot

    When ``runtime`` is provided, embodia routes the same step request through
    the runtime's optimizer / async inference / pacing layer while keeping the
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
            "reset_model=True requires a source object with reset()/step(), "
            "not a bare callable."
        )

    if reset_model:
        assert model is not None
        model.reset()

    raw_frame = robot.observe() if frame is None else frame
    normalized_frame = _as_frame(raw_frame)
    validate_frame(normalized_frame)

    raw_action = _call_action_fn(action_source, normalized_frame)
    normalized_action = _as_action(raw_action)
    validate_action(normalized_action)

    if execute_action:
        robot.act(normalized_action)

    return StepResult(frame=normalized_frame, action=normalized_action)


__all__ = ["ActionSource", "StepResult", "run_step"]
