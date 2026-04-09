"""Helpers for the inference side of inferaxis's unified runtime data flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from .shared.action_source import (
    ActionSource,
    coalesce_source_argument as _coalesce_source_argument,
    call_action_fn as _call_action_fn,
    resolve_action_source as _resolve_action_source,
)
from .shared.coerce import as_action as _as_action
from .shared.coerce import as_frame as _as_frame
from .shared.coerce import maybe_as_action as _maybe_as_action
from .shared.dispatch import (
    POLICY_INFER_CHUNK_METHODS,
    POLICY_INFER_METHODS,
    POLICY_RESET_METHODS,
    ROBOT_ACT_METHODS,
    ROBOT_OBSERVE_METHODS,
    format_method_options,
    resolve_callable_method,
)
from .shared.sequence import attach_runtime_frame_metadata
from .checks import validate_action, validate_frame


@dataclass(slots=True)
class StepResult:
    """Unified result of one inferaxis-controlled closed-loop step.

    ``raw_action`` is the action produced by the policy / action source before
    robot-side execution feedback or runtime-side action optimizers change it.

    ``action`` is the final action inferaxis attributes to this step. When the
    robot's ``act`` / ``send_command`` method returns an action-like value,
    inferaxis uses that returned action. Otherwise it falls back to the
    requested action.

    ``plan_refreshed`` and ``control_wait_s`` are always present so the result
    shape stays stable across plain and runtime-managed calls. In the plain
    non-runtime path they default to ``True`` and ``0.0``.
    """

    frame: Frame
    raw_action: Action
    action: Action
    plan_refreshed: bool = True
    control_wait_s: float = 0.0


def run_step(
    robot: object,
    source: object | None = None,
    *,
    policy: object | None = None,
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
    6. if the robot returns an action-like value, use it as the final action

    ``source=...`` is the preferred public name for the action-producing side.
    ``policy=...`` is kept as a compatibility alias.

    When ``runtime`` is provided, inferaxis routes the same step request through
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
            source,
            policy=policy,
            action_fn=action_fn,
            frame=frame,
            execute_action=execute_action,
            reset_policy=reset_policy,
            pace_control=pace_control,
        )
    source_obj = _coalesce_source_argument(source, policy)

    action_source, can_reset = _resolve_action_source(
        source_obj,
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
        if source_obj is None:
            raise InterfaceValidationError(
                "reset_policy=True requires source=... or policy=... to be set."
            )
        reset, reset_name = resolve_callable_method(source_obj, POLICY_RESET_METHODS)
        if not callable(reset) or reset_name is None:
            raise InterfaceValidationError(
                "reset_policy=True requires a source object exposing "
                f"{format_method_options(POLICY_RESET_METHODS)}."
            )
        try:
            reset()
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(source_obj).__name__}.{reset_name}() raised "
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
    normalized_frame = attach_runtime_frame_metadata(
        _as_frame(raw_frame),
        owner=robot,
    )
    validate_frame(normalized_frame)

    raw_action = _call_action_fn(action_source, normalized_frame)
    normalized_action = _as_action(raw_action)
    validate_action(normalized_action)

    final_action = normalized_action
    if execute_action:
        act, act_name = resolve_callable_method(robot, ROBOT_ACT_METHODS)
        if not callable(act) or act_name is None:
            raise InterfaceValidationError(
                f"{type(robot).__name__} must expose "
                f"{format_method_options(ROBOT_ACT_METHODS)}."
            )
        try:
            raw_executed_action = act(normalized_action)
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(robot).__name__}.{act_name}(action) raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        executed_action = _maybe_as_action(raw_executed_action)
        if executed_action is not None:
            validate_action(executed_action)
            final_action = executed_action

    return StepResult(
        frame=normalized_frame,
        raw_action=normalized_action,
        action=final_action,
        plan_refreshed=True,
        control_wait_s=0.0,
    )


__all__ = ["ActionSource", "StepResult", "run_step"]
