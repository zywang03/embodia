"""Inference runtime engine built on inferaxis's normalized flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ..flow import (
    StepResult,
    _execute_step_action,
    _resolve_step_frame,
)
from ...shared.action_source import (
    ActionSink,
    ActionSource,
    FrameSource,
    callable_key,
    resolve_runtime_owner,
)
from .chunk_scheduler import ChunkScheduler
from .control import RealtimeController
from .optimizers import BlendWeight
from .protocols import ChunkRequest
from ._engine_config import (
    scheduler_control_period_s,
    scheduler_initial_latency_steps,
    validate_runtime_config,
)
from ._engine_scheduler import (
    bootstrap_chunk_scheduler,
    default_request,
    ensure_chunk_scheduler,
    resolve_raw_action,
)


class InferenceMode(StrEnum):
    """Named runtime modes for :class:`InferenceRuntime`."""

    SYNC = "sync"
    ASYNC = "async"


@dataclass(slots=True)
class InferenceRuntime:
    """Composable runtime configuration for optimized :func:`run_step` calls."""

    mode: InferenceMode | str
    steps_before_request: int = 0
    execution_steps: int | None = None
    latency_steps: float | None = None
    warmup_requests: int = 1
    profile_delay_requests: int = 3
    interpolation_steps: int = 0
    ensemble_weight: BlendWeight | None = None
    control_hz: float | None = None
    enable_rtc: bool = False
    latency_steps_offset: int = 0
    realtime_controller: RealtimeController | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _chunk_scheduler: ChunkScheduler | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _chunk_scheduler_key: object | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _single_step_source_keys: set[object] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate mode-specific runtime configuration."""

        validate_runtime_config(self, mode_enum=InferenceMode)

    def reset(self) -> None:
        """Reset source state and any attached runtime components."""

        if self._chunk_scheduler is not None:
            self._chunk_scheduler.reset()
        self._chunk_scheduler_key = None

        if self.realtime_controller is not None:
            self.realtime_controller.reset()

    def close(self) -> None:
        """Release any background scheduler resources held by this runtime."""

        if self._chunk_scheduler is not None:
            self._chunk_scheduler.close()
            self._chunk_scheduler = None
        self._chunk_scheduler_key = None

    def _scheduler_initial_latency_steps(self) -> float:
        """Return the startup async latency prior used by the scheduler."""

        return scheduler_initial_latency_steps(self)

    def _scheduler_control_period_s(self) -> float | None:
        """Return the control period used for async latency bootstrap."""

        return scheduler_control_period_s(self)

    def _ensure_chunk_scheduler(
        self,
        *,
        act_src_fn: ActionSource | None,
    ) -> ChunkScheduler | None:
        """Create one hidden chunk scheduler lazily when runtime mode needs it."""

        return ensure_chunk_scheduler(self, act_src_fn=act_src_fn)

    def _default_request(self) -> ChunkRequest:
        """Build one direct-call request for non-scheduled action sources."""

        return default_request()

    def _bootstrap_chunk_scheduler(
        self,
        *,
        frame: Frame,
        chunk_scheduler: ChunkScheduler,
    ) -> bool:
        """Run async startup warmup/profile outside of ``next_action()``."""

        return bootstrap_chunk_scheduler(
            self,
            frame=frame,
            chunk_scheduler=chunk_scheduler,
        )

    def _resolve_raw_action(
        self,
        *,
        frame: Frame,
        act_src_fn: ActionSource | None,
        source_key: object | None,
    ) -> tuple[Action, bool, int]:
        """Resolve one raw action plus its plan metadata from the configured source."""

        return resolve_raw_action(
            self,
            frame=frame,
            act_src_fn=act_src_fn,
            source_key=source_key,
        )

    def bootstrap_async(
        self,
        *,
        observe_fn: FrameSource | None = None,
        act_src_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, object] | None = None,
    ) -> bool:
        """Explicitly run async startup warmup/profile before the first step."""

        if self.mode is not InferenceMode.ASYNC:
            raise InterfaceValidationError(
                "InferenceRuntime.bootstrap_async() requires mode=ASYNC."
            )

        metadata_owner = resolve_runtime_owner(
            observe_fn,
            act_src_fn,
        )
        normalized_frame = _resolve_step_frame(
            observe_fn,
            frame,
            owner=metadata_owner,
        )
        chunk_scheduler = self._ensure_chunk_scheduler(
            act_src_fn=act_src_fn,
        )
        if chunk_scheduler is None:
            raise InterfaceValidationError(
                "InferenceRuntime.bootstrap_async() requires async chunk scheduling."
            )

        bootstrapped = self._bootstrap_chunk_scheduler(
            frame=normalized_frame,
            chunk_scheduler=chunk_scheduler,
        )
        plan_length = chunk_scheduler.active_source_plan_length
        source_key = callable_key(act_src_fn)
        if plan_length == 1 and source_key is not None:
            self._single_step_source_keys.add(source_key)
            chunk_scheduler.close()
            self._chunk_scheduler = None
            self._chunk_scheduler_key = None
            raise InterfaceValidationError(
                "InferenceRuntime(mode=ASYNC) requires act_src_fn=... "
                "to return a chunk with more than one action."
            )
        return bootstrapped

    def _run_step_impl(
        self,
        *,
        observe_fn: FrameSource | None = None,
        act_fn: ActionSink | None = None,
        act_src_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, object] | None = None,
        execute_action: bool | None = None,
        pace_control: bool = True,
        metadata_owner: object | None = None,
    ) -> StepResult:
        """Internal implementation for runtime-managed local execution."""

        source_key = callable_key(act_src_fn)
        frame_owner = (
            metadata_owner
            if metadata_owner is not None
            else resolve_runtime_owner(observe_fn, act_src_fn)
        )
        normalized_frame = _resolve_step_frame(
            observe_fn,
            frame,
            owner=frame_owner,
        )

        raw_action, plan_refreshed, plan_length = self._resolve_raw_action(
            frame=normalized_frame,
            act_src_fn=act_src_fn,
            source_key=source_key,
        )
        action = raw_action

        should_execute = True if execute_action is None else execute_action
        if should_execute:
            action = _execute_step_action(act_fn, action)

        control_wait_s = 0.0
        if pace_control and self.realtime_controller is not None:
            control_wait_s = self.realtime_controller.wait()

        return StepResult(
            frame=normalized_frame,
            raw_action=raw_action,
            action=action,
            plan_refreshed=plan_refreshed,
            control_wait_s=control_wait_s,
        )

    def run_step(
        self,
        *,
        observe_fn: FrameSource | None = None,
        act_fn: ActionSink | None = None,
        act_src_fn: ActionSource | None = None,
        frame: Frame | Mapping[str, object] | None = None,
        execute_action: bool | None = None,
        pace_control: bool = True,
    ) -> StepResult:
        """Run one runtime-managed step directly through this runtime instance."""

        metadata_owner = resolve_runtime_owner(
            observe_fn,
            act_fn,
            act_src_fn,
        )
        return self._run_step_impl(
            observe_fn=observe_fn,
            act_fn=act_fn,
            act_src_fn=act_src_fn,
            frame=frame,
            execute_action=True if execute_action is None else execute_action,
            pace_control=pace_control,
            metadata_owner=metadata_owner,
        )

    step = run_step


__all__ = ["InferenceMode", "InferenceRuntime"]
