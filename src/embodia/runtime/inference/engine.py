"""Inference runtime engine built on embodia's normalized flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from .._dispatch import (
    MODEL_INFER_CHUNK_METHODS,
    MODEL_INFER_METHODS,
    MODEL_PLAN_METHODS,
    MODEL_RESET_METHODS,
    ROBOT_ACT_METHODS,
    ROBOT_OBSERVE_METHODS,
    format_method_options,
    resolve_callable_method,
)
from ..checks import validate_action, validate_frame
from ..flow import ActionSource, _resolve_action_source, run_step
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
class InferenceStepResult:
    """Result of one optimized inference step."""

    frame: Frame
    raw_action: Action
    action: Action
    plan_refreshed: bool = True
    control_wait_s: float = 0.0


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

    def reset(self, *, model: object | None = None) -> None:
        """Reset model state and any attached runtime components."""

        if model is not None:
            reset, reset_name = resolve_callable_method(model, MODEL_RESET_METHODS)
            if not callable(reset) or reset_name is None:
                raise InterfaceValidationError(
                    "InferenceRuntime.reset(model=...) requires the model to expose "
                    f"{format_method_options(MODEL_RESET_METHODS)}."
                )
            try:
                reset()
            except Exception as exc:
                raise InterfaceValidationError(
                    f"{type(model).__name__}.{reset_name}() raised "
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
                MODEL_INFER_CHUNK_METHODS,
            )
            if callable(infer_chunk):
                return (
                    lambda _source, frame, request, _infer_chunk=infer_chunk: _infer_chunk(
                        frame,
                        request,
                    ),
                    None,
                )

            plan, _ = resolve_callable_method(candidate, MODEL_PLAN_METHODS)
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
        model: object | None = None,
        *,
        action_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, Any] | None = None,
        execute_action: bool | None = None,
        reset_model: bool = False,
        pace_control: bool = True,
    ) -> InferenceStepResult:
        """Internal implementation for runtime-managed inference."""

        resolved_action_source, can_reset = _resolve_action_source(
            model,
            action_fn,
            robot=robot,
        )
        if reset_model and not can_reset:
            raise InterfaceValidationError(
                "reset_model=True requires a source object exposing "
                f"{format_method_options(MODEL_RESET_METHODS)} together with "
                f"{format_method_options(MODEL_INFER_METHODS)} or "
                f"{format_method_options(MODEL_INFER_CHUNK_METHODS)}, not a bare callable."
            )

        if reset_model:
            self.reset(model=model)

        chunk_scheduler = self._ensure_chunk_scheduler(
            source=model,
            action_source=resolved_action_source,
        )

        if chunk_scheduler is None:
            raw_result = run_step(
                robot,
                model,
                action_fn=action_fn,
                frame=frame,
                execute_action=False,
                reset_model=False,
                runtime=None,
            )
            normalized_frame = raw_result.frame
            raw_action = raw_result.action
            plan_refreshed = True
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
                    raw_frame = observe()
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
                model
                if model is not None
                else (action_fn if action_fn is not None else resolved_action_source)
            )
            raw_action, plan_refreshed = chunk_scheduler.next_action(
                action_context,
                normalized_frame,
                fallback_action_source=resolved_action_source,
                prefetch_async=self.mode is InferenceMode.ASYNC,
            )

        optimized_action = self.optimize_action(raw_action, frame=normalized_frame)

        should_execute = True if execute_action is None else execute_action
        if should_execute:
            act, act_name = resolve_callable_method(robot, ROBOT_ACT_METHODS)
            if not callable(act) or act_name is None:
                raise InterfaceValidationError(
                    f"{type(robot).__name__} must expose "
                    f"{format_method_options(ROBOT_ACT_METHODS)}."
                )
            try:
                act(optimized_action)
            except Exception as exc:
                raise InterfaceValidationError(
                    f"{type(robot).__name__}.{act_name}(action) raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        control_wait_s = 0.0
        if pace_control and self.realtime_controller is not None:
            control_wait_s = self.realtime_controller.wait()

        return InferenceStepResult(
            frame=normalized_frame,
            raw_action=raw_action,
            action=optimized_action,
            plan_refreshed=plan_refreshed,
            control_wait_s=control_wait_s,
        )

    def step(
        self,
        robot: object,
        model: object | None = None,
        *,
        action_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, Any] | None = None,
        execute_action: bool | None = None,
        reset_model: bool = False,
        pace_control: bool = True,
    ) -> InferenceStepResult:
        """Compatibility wrapper around :func:`embodia.run_step`."""

        return run_step(
            robot,
            model,
            action_fn=action_fn,
            frame=frame,
            execute_action=True if execute_action is None else execute_action,
            reset_model=reset_model,
            runtime=self,
            pace_control=pace_control,
        )

    def run_step(
        self,
        robot: object,
        model: object | None = None,
        *,
        action_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, Any] | None = None,
        execute_action: bool | None = None,
        reset_model: bool = False,
        pace_control: bool = True,
    ) -> InferenceStepResult:
        """Alias for :meth:`step` for users who prefer function-style naming."""

        return self.step(
            robot,
            model,
            action_fn=action_fn,
            frame=frame,
            execute_action=execute_action,
            reset_model=reset_model,
            pace_control=pace_control,
        )

__all__ = ["InferenceMode", "InferenceRuntime", "InferenceStepResult"]
