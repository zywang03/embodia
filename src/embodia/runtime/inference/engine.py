"""Inference runtime engine built on embodia's normalized flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ..checks import validate_action, validate_frame
from ..flow import ActionSource, _resolve_action_source, run_step
from .async_inference import AsyncInference
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
    action_optimizers: Sequence[ActionOptimizerProtocol | ActionOptimizer] = ()
    async_inference: AsyncInference | None = None
    realtime_controller: RealtimeController | None = None
    execute_action: bool = True

    def __post_init__(self) -> None:
        """Validate mode-specific runtime configuration."""

        try:
            self.mode = InferenceMode(str(self.mode))
        except ValueError as exc:
            raise InterfaceValidationError(
                "InferenceRuntime.mode must be InferenceMode.SYNC or "
                f"InferenceMode.ASYNC, got {self.mode!r}."
            ) from exc
        if self.mode is InferenceMode.SYNC and self.async_inference is not None:
            raise InterfaceValidationError(
                "InferenceRuntime(mode='sync') cannot be combined with "
                "async_inference=.... Remove async_inference or set mode='async'."
            )
        if self.mode is InferenceMode.ASYNC and self.async_inference is None:
            raise InterfaceValidationError(
                "InferenceRuntime(mode='async') requires "
                "async_inference=AsyncInference(...)."
            )

    def reset(self, *, model: object | None = None) -> None:
        """Reset model state and any attached runtime components."""

        if model is not None:
            model.reset()

        for optimizer in self.action_optimizers:
            reset_if_possible(optimizer)

        if self.async_inference is not None:
            self.async_inference.reset()

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
                "reset_model=True requires a source object with reset()/step(), "
                "not a bare callable."
            )

        if reset_model:
            self.reset(model=model)

        if self.mode is InferenceMode.SYNC:
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
            assert self.async_inference is not None
            raw_frame = robot.observe() if frame is None else frame
            normalized_frame = as_frame(raw_frame)
            validate_frame(normalized_frame)

            if (
                self.async_inference.control_hz is None
                and self.realtime_controller is not None
            ):
                self.async_inference.bind_control_hz(self.realtime_controller.hz)

            action_context = (
                model
                if model is not None
                else (action_fn if action_fn is not None else resolved_action_source)
            )
            raw_action, plan_refreshed = self.async_inference.next_action(
                action_context,
                normalized_frame,
                fallback_action_source=resolved_action_source,
            )

        optimized_action = self.optimize_action(raw_action, frame=normalized_frame)

        should_execute = (
            self.execute_action if execute_action is None else execute_action
        )
        if should_execute:
            robot.act(optimized_action)

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
            execute_action=(
                self.execute_action if execute_action is None else execute_action
            ),
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
