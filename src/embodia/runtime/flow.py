"""Helpers for the inference side of embodia's unified runtime data flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from .shared.action_source import (
    ActionSource,
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
from .checks import validate_action, validate_frame


@dataclass(slots=True)
class StepTiming:
    """Timing summary for one embodia-controlled closed-loop step.

    ``embodia_overhead_s`` estimates the framework's own synchronous overhead on
    the current step. It is computed as:

    ``total_s - observe_call_s - source_call_s - act_call_s - scheduler_wait_s - control_wait_s``

    Notes:

    - ``source_call_s`` covers direct blocking calls into the policy / action
      function / chunk provider on the current thread.
    - ``scheduler_wait_s`` covers time spent waiting for an async chunk request
      to finish when the current step had to block for it.
    - background async inference that does not block the current step is not
      charged to this step's wall time.
    """

    total_s: float
    embodia_overhead_s: float
    observe_call_s: float = 0.0
    source_call_s: float = 0.0
    act_call_s: float = 0.0
    scheduler_wait_s: float = 0.0
    control_wait_s: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Export timing values to a plain dictionary."""

        return {
            "total_s": self.total_s,
            "embodia_overhead_s": self.embodia_overhead_s,
            "observe_call_s": self.observe_call_s,
            "source_call_s": self.source_call_s,
            "act_call_s": self.act_call_s,
            "scheduler_wait_s": self.scheduler_wait_s,
            "control_wait_s": self.control_wait_s,
        }


def build_step_timing(
    total_s: float,
    *,
    observe_call_s: float = 0.0,
    source_call_s: float = 0.0,
    act_call_s: float = 0.0,
    scheduler_wait_s: float = 0.0,
    control_wait_s: float = 0.0,
) -> StepTiming:
    """Build one :class:`StepTiming` while keeping overhead non-negative."""

    external_s = (
        observe_call_s
        + source_call_s
        + act_call_s
        + scheduler_wait_s
        + control_wait_s
    )
    return StepTiming(
        total_s=max(float(total_s), 0.0),
        embodia_overhead_s=max(float(total_s) - external_s, 0.0),
        observe_call_s=max(float(observe_call_s), 0.0),
        source_call_s=max(float(source_call_s), 0.0),
        act_call_s=max(float(act_call_s), 0.0),
        scheduler_wait_s=max(float(scheduler_wait_s), 0.0),
        control_wait_s=max(float(control_wait_s), 0.0),
    )


@dataclass(slots=True)
class StepResult:
    """Unified result of one embodia-controlled closed-loop step.

    ``raw_action`` is the action produced by the policy / action source before
    robot-side execution feedback or runtime-side action optimizers change it.

    ``action`` is the final action embodia attributes to this step. When the
    robot's ``act`` / ``send_command`` method returns an action-like value,
    embodia uses that returned action. Otherwise it falls back to the
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
    timing: StepTiming | None = None


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
    measure_timing: bool = False,
) -> StepResult:
    """Run one normalized data-flow step.

    The flow is:

    1. observe a frame from the robot, unless a frame is provided
    2. normalize and validate the frame
    3. get one action from either ``source.infer(frame)`` or ``source(frame)``
    4. normalize and validate the action
    5. optionally execute the action on the robot
    6. if the robot returns an action-like value, use it as the final action

    When ``runtime`` is provided, embodia routes the same step request through
    the runtime's optimizer / chunk-scheduling / pacing layer while keeping the
    public call site unchanged.

    When ``measure_timing=True``, the returned :class:`StepResult` includes one
    :class:`StepTiming` summary with ``embodia_overhead_s``.
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
            measure_timing=measure_timing,
        )

    observe_call_s = 0.0
    source_call_s = 0.0
    act_call_s = 0.0
    total_start_s = time.perf_counter() if measure_timing else 0.0

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
            observe_started_at_s = time.perf_counter() if measure_timing else 0.0
            raw_frame = observe()
            if measure_timing:
                observe_call_s = max(time.perf_counter() - observe_started_at_s, 0.0)
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(robot).__name__}.{observe_name}() raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    else:
        raw_frame = frame
    normalized_frame = _as_frame(raw_frame)
    validate_frame(normalized_frame)

    source_started_at_s = time.perf_counter() if measure_timing else 0.0
    raw_action = _call_action_fn(action_source, normalized_frame)
    if measure_timing:
        source_call_s = max(time.perf_counter() - source_started_at_s, 0.0)
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
            act_started_at_s = time.perf_counter() if measure_timing else 0.0
            raw_executed_action = act(normalized_action)
            if measure_timing:
                act_call_s = max(time.perf_counter() - act_started_at_s, 0.0)
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(robot).__name__}.{act_name}(action) raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        executed_action = _maybe_as_action(raw_executed_action)
        if executed_action is not None:
            validate_action(executed_action)
            final_action = executed_action

    timing = None
    if measure_timing:
        timing = build_step_timing(
            time.perf_counter() - total_start_s,
            observe_call_s=observe_call_s,
            source_call_s=source_call_s,
            act_call_s=act_call_s,
        )

    return StepResult(
        frame=normalized_frame,
        raw_action=normalized_action,
        action=final_action,
        plan_refreshed=True,
        control_wait_s=0.0,
        timing=timing,
    )


__all__ = ["ActionSource", "StepResult", "StepTiming", "build_step_timing", "run_step"]
