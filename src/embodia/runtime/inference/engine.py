"""Inference runtime engine built on embodia's normalized flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import time
from typing import Any

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ..shared.action_source import (
    ActionSource,
    resolve_action_source as _resolve_action_source,
)
from ..shared.coerce import maybe_as_action as _maybe_as_action
from ..shared.dispatch import (
    POLICY_INFER_CHUNK_METHODS,
    POLICY_INFER_METHODS,
    POLICY_PLAN_METHODS,
    POLICY_RESET_METHODS,
    ROBOT_ACT_METHODS,
    ROBOT_OBSERVE_METHODS,
    format_method_options,
    resolve_callable_method,
)
from ..checks import validate_action, validate_frame
from ..flow import StepResult, build_step_timing, run_step
from .chunk_scheduler import ChunkScheduler
from .common import as_action, as_frame, reset_if_possible
from .control import RealtimeController
from .protocols import ActionOptimizer, ActionOptimizerProtocol


class InferenceMode(StrEnum):
    """Named runtime modes for :class:`InferenceRuntime`.

    ``StrEnum`` keeps the values readable in user code while remaining fully
    compatible with plain string-based configuration.
    """

    SYNC = "sync"
    ASYNC = "async"


@dataclass(slots=True)
class InferenceRuntime:
    """Composable runtime configuration for optimized :func:`run_step` calls."""

    mode: InferenceMode | str
    overlap_ratio: float | None = None
    action_optimizers: Sequence[ActionOptimizerProtocol | ActionOptimizer] = ()
    realtime_controller: RealtimeController | None = None
    _chunk_scheduler: ChunkScheduler | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _chunk_scheduler_source_id: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _chunk_scheduler_action_source_key: object | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate mode-specific runtime configuration."""

        try:
            self.mode = InferenceMode(str(self.mode))
        except ValueError as exc:
            raise InterfaceValidationError(
                "InferenceRuntime.mode must be InferenceMode.SYNC or "
                f"InferenceMode.ASYNC, got {self.mode!r}."
            ) from exc
        if self.overlap_ratio is not None:
            if isinstance(self.overlap_ratio, bool) or not isinstance(
                self.overlap_ratio,
                (int, float),
            ):
                raise InterfaceValidationError(
                    "InferenceRuntime.overlap_ratio must be a real number in "
                    "the range (0, 1)."
                )
            self.overlap_ratio = float(self.overlap_ratio)
            if not 0.0 < self.overlap_ratio < 1.0:
                raise InterfaceValidationError(
                    "InferenceRuntime.overlap_ratio must be in the range (0, 1), "
                    f"got {self.overlap_ratio!r}."
                )

    def reset(self, *, policy: object | None = None) -> None:
        """Reset policy state and any attached runtime components."""

        if policy is not None:
            reset, reset_name = resolve_callable_method(policy, POLICY_RESET_METHODS)
            if not callable(reset) or reset_name is None:
                raise InterfaceValidationError(
                    "InferenceRuntime.reset(policy=...) requires the policy to expose "
                    f"{format_method_options(POLICY_RESET_METHODS)}."
                )
            try:
                reset()
            except Exception as exc:
                raise InterfaceValidationError(
                    f"{type(policy).__name__}.{reset_name}() raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        for optimizer in self.action_optimizers:
            reset_if_possible(optimizer)

        if self._chunk_scheduler is not None:
            self._chunk_scheduler.reset()
        self._chunk_scheduler_source_id = None
        self._chunk_scheduler_action_source_key = None

        if self.realtime_controller is not None:
            self.realtime_controller.reset()

    def optimize_action(self, action: Action, *, frame: Frame) -> Action:
        """Run the configured action optimizers in sequence."""

        current = as_action(action)
        validate_action(current)

        for index, optimizer in enumerate(self.action_optimizers):
            try:
                optimized = optimizer(current, frame)
            except Exception as exc:
                raise InterfaceValidationError(
                    f"action optimizer #{index} raised {type(exc).__name__}: {exc}"
                ) from exc

            current = as_action(optimized)
            validate_action(current)

        return current

    @staticmethod
    def _resolve_runtime_chunk_hooks(
        source: object | None,
        action_source: ActionSource,
    ) -> tuple[
        Any | None,
        Any | None,
    ]:
        """Resolve optional source-level chunk hooks for runtime scheduling.

        Supported optional hooks are:

        - ``embodia_infer_chunk(frame, request)`` / ``infer_chunk(frame, request)``
          for overlap-conditioned chunk generation
        - ``plan(frame)`` for simple chunk generation without request context
        """

        candidates: list[object] = []
        if source is not None:
            candidates.append(source)
        if action_source is not source:
            candidates.append(action_source)

        for candidate in candidates:
            infer_chunk, _ = resolve_callable_method(
                candidate,
                POLICY_INFER_CHUNK_METHODS,
            )
            if callable(infer_chunk):
                return (
                    lambda _source, frame, request, _infer_chunk=infer_chunk: _infer_chunk(
                        frame,
                        request,
                    ),
                    None,
                )

            plan, _ = resolve_callable_method(candidate, POLICY_PLAN_METHODS)
            if callable(plan):
                return (
                    None,
                    lambda _source, frame, _plan=plan: _plan(frame),
                )

        return None, None

    def _should_use_chunk_scheduler(self) -> bool:
        """Return whether this runtime should route steps through chunk scheduling."""

        return self.mode is InferenceMode.ASYNC or self.overlap_ratio is not None

    @staticmethod
    def _action_source_key(action_source: ActionSource) -> object:
        """Return one stable identity key for a callable action source."""

        bound_self = getattr(action_source, "__self__", None)
        bound_func = getattr(action_source, "__func__", None)
        if bound_self is not None and bound_func is not None:
            return (id(bound_self), id(bound_func))
        return id(action_source)

    def _ensure_chunk_scheduler(
        self,
        *,
        source: object | None,
        action_source: ActionSource,
    ) -> ChunkScheduler | None:
        """Create one hidden chunk scheduler lazily when runtime mode needs it."""

        if not self._should_use_chunk_scheduler():
            return None

        source_id = id(source) if source is not None else None
        action_source_key = self._action_source_key(action_source)
        if self._chunk_scheduler is not None:
            if (
                self._chunk_scheduler_source_id != source_id
                or self._chunk_scheduler_action_source_key != action_source_key
            ):
                self._chunk_scheduler = None
            else:
                if self.overlap_ratio is not None:
                    self._chunk_scheduler.overlap_ratio = self.overlap_ratio
                return self._chunk_scheduler

        chunk_provider, plan_provider = self._resolve_runtime_chunk_hooks(
            source,
            action_source,
        )
        scheduler = ChunkScheduler(
            chunk_provider=chunk_provider,
            plan_provider=plan_provider,
            overlap_ratio=(
                self.overlap_ratio if self.overlap_ratio is not None else 0.2
            ),
        )
        self._chunk_scheduler = scheduler
        self._chunk_scheduler_source_id = source_id
        self._chunk_scheduler_action_source_key = action_source_key
        return scheduler

    def _run_step_impl(
        self,
        robot: object,
        policy: object | None = None,
        *,
        action_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, Any] | None = None,
        execute_action: bool | None = None,
        reset_policy: bool = False,
        pace_control: bool = True,
        measure_timing: bool = False,
    ) -> StepResult:
        """Internal implementation for runtime-managed inference."""

        total_start_s = time.perf_counter() if measure_timing else 0.0
        observe_call_s = 0.0
        source_call_s = 0.0
        act_call_s = 0.0

        resolved_action_source, can_reset = _resolve_action_source(
            policy,
            action_fn,
            robot=robot,
        )
        if reset_policy and not can_reset:
            raise InterfaceValidationError(
                "reset_policy=True requires a source object exposing "
                f"{format_method_options(POLICY_RESET_METHODS)} together with "
                f"{format_method_options(POLICY_INFER_METHODS)} or "
                f"{format_method_options(POLICY_INFER_CHUNK_METHODS)}, not a bare callable."
            )

        if reset_policy:
            self.reset(policy=policy)

        chunk_scheduler = self._ensure_chunk_scheduler(
            source=policy,
            action_source=resolved_action_source,
        )

        if chunk_scheduler is None:
            raw_result = run_step(
                robot,
                policy,
                action_fn=action_fn,
                frame=frame,
                execute_action=False,
                reset_policy=False,
                runtime=None,
                measure_timing=measure_timing,
            )
            normalized_frame = raw_result.frame
            raw_action = raw_result.raw_action
            plan_refreshed = True
            if raw_result.timing is not None:
                observe_call_s += raw_result.timing.observe_call_s
                source_call_s += raw_result.timing.source_call_s
        else:
            if frame is None:
                observe, observe_name = resolve_callable_method(
                    robot,
                    ROBOT_OBSERVE_METHODS,
                )
                if not callable(observe) or observe_name is None:
                    raise InterfaceValidationError(
                        f"{type(robot).__name__} must expose "
                        f"{format_method_options(ROBOT_OBSERVE_METHODS)}."
                    )
                try:
                    observe_started_at_s = time.perf_counter() if measure_timing else 0.0
                    raw_frame = observe()
                    if measure_timing:
                        observe_call_s = max(
                            time.perf_counter() - observe_started_at_s,
                            0.0,
                        )
                except Exception as exc:
                    raise InterfaceValidationError(
                        f"{type(robot).__name__}.{observe_name}() raised "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            else:
                raw_frame = frame
            normalized_frame = as_frame(raw_frame)
            validate_frame(normalized_frame)

            if (
                chunk_scheduler.control_hz is None
                and self.realtime_controller is not None
            ):
                chunk_scheduler.bind_control_hz(self.realtime_controller.hz)

            action_context = (
                policy
                if policy is not None
                else (action_fn if action_fn is not None else resolved_action_source)
            )
            scheduler_timing = None
            raw_action, plan_refreshed, scheduler_timing = chunk_scheduler.next_action(
                action_context,
                normalized_frame,
                fallback_action_source=resolved_action_source,
                prefetch_async=self.mode is InferenceMode.ASYNC,
                measure_timing=measure_timing,
            )
            if scheduler_timing is not None:
                source_call_s += scheduler_timing.provider_call_s
                scheduler_wait_s = scheduler_timing.wait_for_ready_s
            else:
                scheduler_wait_s = 0.0
        if chunk_scheduler is None:
            scheduler_wait_s = 0.0

        optimized_action = self.optimize_action(raw_action, frame=normalized_frame)

        should_execute = True if execute_action is None else execute_action
        final_action = optimized_action
        if should_execute:
            act, act_name = resolve_callable_method(robot, ROBOT_ACT_METHODS)
            if not callable(act) or act_name is None:
                raise InterfaceValidationError(
                    f"{type(robot).__name__} must expose "
                    f"{format_method_options(ROBOT_ACT_METHODS)}."
                )
            try:
                act_started_at_s = time.perf_counter() if measure_timing else 0.0
                raw_executed_action = act(optimized_action)
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

        control_wait_s = 0.0
        if pace_control and self.realtime_controller is not None:
            control_wait_s = self.realtime_controller.wait()

        timing = None
        if measure_timing:
            timing = build_step_timing(
                time.perf_counter() - total_start_s,
                observe_call_s=observe_call_s,
                source_call_s=source_call_s,
                act_call_s=act_call_s,
                scheduler_wait_s=scheduler_wait_s,
                control_wait_s=control_wait_s,
            )

        return StepResult(
            frame=normalized_frame,
            raw_action=raw_action,
            action=final_action,
            plan_refreshed=plan_refreshed,
            control_wait_s=control_wait_s,
            timing=timing,
        )

    def step(
        self,
        robot: object,
        policy: object | None = None,
        *,
        action_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, Any] | None = None,
        execute_action: bool | None = None,
        reset_policy: bool = False,
        pace_control: bool = True,
        measure_timing: bool = False,
    ) -> StepResult:
        """Compatibility wrapper around :func:`embodia.run_step`."""

        return run_step(
            robot,
            policy,
            action_fn=action_fn,
            frame=frame,
            execute_action=True if execute_action is None else execute_action,
            reset_policy=reset_policy,
            runtime=self,
            pace_control=pace_control,
            measure_timing=measure_timing,
        )

    def run_step(
        self,
        robot: object,
        policy: object | None = None,
        *,
        action_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, Any] | None = None,
        execute_action: bool | None = None,
        reset_policy: bool = False,
        pace_control: bool = True,
        measure_timing: bool = False,
    ) -> StepResult:
        """Alias for :meth:`step` for users who prefer function-style naming."""

        return self.step(
            robot,
            policy,
            action_fn=action_fn,
            frame=frame,
            execute_action=execute_action,
            reset_policy=reset_policy,
            pace_control=pace_control,
            measure_timing=measure_timing,
        )

__all__ = ["InferenceMode", "InferenceRuntime"]
