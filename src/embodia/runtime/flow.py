"""Helpers for the inference side of embodia's unified runtime data flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from .shared.action_source import (
    ActionSource,
    call_action_fn as _call_action_fn,
    resolve_action_source as _resolve_action_source,
)
from .shared.coerce import as_action as _as_action
from .shared.coerce import as_frame as _as_frame
from .shared.dispatch import (
    POLICY_INFER_CHUNK_METHODS,
    POLICY_INFER_METHODS,
    POLICY_RESET_METHODS,
    ROBOT_ACT_METHODS,
    ROBOT_OBSERVE_METHODS,
    format_method_options,
    resolve_callable_method,
)
from .checks import validate_action, validate_frame


@dataclass(slots=True)
class StepResult:
    """Result of one normalized robot -> action source -> robot runtime step."""

    frame: Frame
    action: Action


def run_step(
    robot: object,
    policy: object | None = None,
    *,
    action_fn: ActionSource | None = None,
    frame: Frame | Mapping[str, Any] | None = None,
    execute_action: bool = True,
    reset_policy: bool = False,
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
            policy,
            action_fn=action_fn,
            frame=frame,
            execute_action=execute_action,
            reset_policy=reset_policy,
            pace_control=pace_control,
        )

    action_source, can_reset = _resolve_action_source(
        policy,
        action_fn,
        robot=robot,
    )
    if reset_policy and not can_reset:
        raise InterfaceValidationError(
            "reset_policy=True requires a source object that exposes "
            f"{format_method_options(POLICY_RESET_METHODS)} together with "
            f"{format_method_options(POLICY_INFER_METHODS)} or "
            f"{format_method_options(POLICY_INFER_CHUNK_METHODS)}, not a bare callable."
        )

    if reset_policy:
        assert policy is not None
        reset, reset_name = resolve_callable_method(policy, POLICY_RESET_METHODS)
        if not callable(reset) or reset_name is None:
            raise InterfaceValidationError(
                "reset_policy=True requires a policy object exposing "
                f"{format_method_options(POLICY_RESET_METHODS)}."
            )
        try:
            reset()
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(policy).__name__}.{reset_name}() raised "
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
