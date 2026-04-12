"""Inference runtime engine built on inferaxis's normalized flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ..checks import validate_action
from ..flow import (
    StepResult,
    _execute_step_action,
    _resolve_step_frame,
)
from ...shared.action_source import (
    ActionSink,
    ActionSource,
    FrameSource,
    first_action_and_plan_length_from_action_call,
    callable_key,
    resolve_runtime_owner,
)
from .chunk_scheduler import ChunkScheduler
from ...shared.common import as_action, reset_if_possible
from .control import RealtimeController
from .optimizers import ActionEnsembler
from .protocols import ActionOptimizer, ActionOptimizerProtocol


class InferenceMode(StrEnum):
    """Named runtime modes for :class:`InferenceRuntime`."""

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
    _optimizer_source_key: object | None = field(
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
                    "the range [0, 1)."
                )
            self.overlap_ratio = float(self.overlap_ratio)
            if not 0.0 <= self.overlap_ratio < 1.0:
                raise InterfaceValidationError(
                    "InferenceRuntime.overlap_ratio must be in the range [0, 1), "
                    f"got {self.overlap_ratio!r}."
                )

    def reset(self) -> None:
        """Reset source state and any attached runtime components."""

        for optimizer in self.action_optimizers:
            reset_if_possible(optimizer)

        if self._chunk_scheduler is not None:
            self._chunk_scheduler.reset()
        self._chunk_scheduler_key = None
        self._optimizer_source_key = None

        if self.realtime_controller is not None:
            self.realtime_controller.reset()

    def close(self) -> None:
        """Release any background scheduler resources held by this runtime."""

        if self._chunk_scheduler is not None:
            self._chunk_scheduler.close()
            self._chunk_scheduler = None
        self._chunk_scheduler_key = None
        self._optimizer_source_key = None

    def _ensure_chunk_scheduler(
        self,
        *,
        act_src_fn: ActionSource | None,
    ) -> ChunkScheduler | None:
        """Create one hidden chunk scheduler lazily when runtime mode needs it."""

        scheduler_key = callable_key(act_src_fn)
        if (
            self.mode is not InferenceMode.ASYNC
            and self.overlap_ratio is None
            and scheduler_key in self._single_step_source_keys
        ):
            return None
        if self.mode is InferenceMode.ASYNC and scheduler_key in self._single_step_source_keys:
            raise InterfaceValidationError(
                "InferenceRuntime(mode=ASYNC) requires act_src_fn=... to return "
                "a chunk with more than one action."
            )
        if self._chunk_scheduler is not None:
            if (
                self._chunk_scheduler_key != scheduler_key
                or (
                    self.mode is not InferenceMode.ASYNC
                    and self.overlap_ratio is None
                    and scheduler_key in self._single_step_source_keys
                )
            ):
                self._chunk_scheduler.close()
                self._chunk_scheduler = None
                self._chunk_scheduler_key = None
            else:
                self._chunk_scheduler.overlap_ratio = (
                    self.overlap_ratio
                    if self.overlap_ratio is not None
                    else (0.2 if self.mode is InferenceMode.ASYNC else 0.0)
                )
                self._chunk_scheduler.use_overlap_blend = self._uses_overlap_blend()
                return self._chunk_scheduler

        if act_src_fn is None:
            raise InterfaceValidationError(
                "InferenceRuntime async/overlap scheduling requires act_src_fn=...."
            )

        scheduler_overlap_ratio = (
            self.overlap_ratio
            if self.overlap_ratio is not None
            else (0.2 if self.mode is InferenceMode.ASYNC else 0.0)
        )
        scheduler = ChunkScheduler(
            action_source=act_src_fn,
            overlap_ratio=scheduler_overlap_ratio,
            use_overlap_blend=self._uses_overlap_blend(),
        )
        self._chunk_scheduler = scheduler
        self._chunk_scheduler_key = scheduler_key
        return scheduler

    def _uses_overlap_blend(self) -> bool:
        """Return whether chunk handoff should blend overlap actions."""

        for optimizer in self.action_optimizers:
            if isinstance(optimizer, ActionEnsembler):
                return True
        return False

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
        known_single_step_source = source_key in self._single_step_source_keys
        chunk_scheduler = self._ensure_chunk_scheduler(act_src_fn=act_src_fn)

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

        if chunk_scheduler is None:
            if act_src_fn is None:
                raise InterfaceValidationError(
                    "run_step() requires act_src_fn=...."
                )
            raw_action, returned_plan_length = (
                first_action_and_plan_length_from_action_call(
                    act_src_fn,
                    normalized_frame,
                )
            )
            if known_single_step_source and returned_plan_length != 1:
                raise InterfaceValidationError(
                    "act_src_fn was previously classified as single-step for "
                    f"this runtime, but returned chunk length "
                    f"{returned_plan_length}."
                )
            if source_key is not None and returned_plan_length == 1:
                self._single_step_source_keys.add(source_key)
            plan_refreshed = True
            apply_runtime_processing = returned_plan_length > 1
        else:
            raw_action, plan_refreshed = chunk_scheduler.next_action(
                normalized_frame,
                prefetch_async=self.mode is InferenceMode.ASYNC,
            )
            source_plan_length = chunk_scheduler.active_source_plan_length
            apply_runtime_processing = source_plan_length > 1
            if source_plan_length == 1 and source_key is not None:
                self._single_step_source_keys.add(source_key)
                chunk_scheduler.close()
                self._chunk_scheduler = None
                self._chunk_scheduler_key = None
                if self.mode is InferenceMode.ASYNC:
                    raise InterfaceValidationError(
                        "InferenceRuntime(mode=ASYNC) requires act_src_fn=... "
                        "to return a chunk with more than one action."
                    )

        action = raw_action
        if apply_runtime_processing:
            if self._optimizer_source_key != source_key:
                for optimizer in self.action_optimizers:
                    reset_if_possible(optimizer)
                self._optimizer_source_key = source_key
            for index, optimizer in enumerate(self.action_optimizers):
                try:
                    optimized = optimizer(action, normalized_frame)
                except Exception as exc:
                    raise InterfaceValidationError(
                        f"action optimizer #{index} raised {type(exc).__name__}: {exc}"
                    ) from exc
                action = as_action(optimized)
                validate_action(action)
        else:
            for optimizer in self.action_optimizers:
                reset_if_possible(optimizer)
            self._optimizer_source_key = None

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
