"""Async chunk scheduling helpers for inference runtime."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import time
import warnings
from typing import Any

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ..checks import validate_action, validate_frame
from .common import as_action, as_frame, validate_positive_number
from .protocols import (
    ActionPlan,
    ActionPlanProvider,
    ActionPlanProviderProtocol,
    ChunkProvider,
    ChunkProviderProtocol,
    ChunkRequest,
)

FallbackActionSource = Callable[[Frame], Action | Mapping[str, Any]]


@dataclass(slots=True)
class _CompletedChunk:
    """One completed async chunk request."""

    request: ChunkRequest
    plan: list[Action]
    completed_at_s: float


@dataclass(slots=True)
class AsyncInference:
    """Async overlap-conditioned chunk scheduler.

    ``condition_steps`` controls how many actions from the active chunk are
    exposed to the next chunk request as overlap history. ``prefetch_steps``
    controls how early the scheduler starts asking for the next chunk. When
    ``control_hz`` is known, the scheduler can adapt the effective prefetch
    threshold upward based on observed request latency.

    The base configuration should satisfy ``condition_steps >= prefetch_steps``.
    If runtime latency later suggests an even larger safe window, the scheduler
    keeps running and emits warnings so the caller can widen both values.
    """

    chunk_provider: ChunkProviderProtocol | ChunkProvider | None = None
    plan_provider: ActionPlanProviderProtocol | ActionPlanProvider | None = None
    condition_steps: int = 2
    prefetch_steps: int | None = None
    max_chunk_size: int | None = None
    control_hz: float | None = None
    latency_budget_s: float | None = None
    safety_margin_steps: int = 1
    warning_interval_s: float = 5.0
    clock: Callable[[], float] = time.perf_counter
    warning_emitter: Callable[[str], None] | None = None
    _active_chunk: list[Action] = field(default_factory=list, init=False, repr=False)
    _active_index: int = field(default=0, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)
    _ready_response: _CompletedChunk | None = field(default=None, init=False, repr=False)
    _in_flight: Future[_CompletedChunk] | None = field(default=None, init=False, repr=False)
    _latency_samples_s: deque[float] = field(init=False, repr=False)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _last_warning_time: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and initialize runtime state."""

        if self.chunk_provider is not None and self.plan_provider is not None:
            raise InterfaceValidationError(
                "AsyncInference accepts chunk_provider or plan_provider, not both."
            )
        if isinstance(self.condition_steps, bool) or not isinstance(
            self.condition_steps, int
        ):
            raise InterfaceValidationError("condition_steps must be an int.")
        if self.condition_steps <= 0:
            raise InterfaceValidationError(
                f"condition_steps must be > 0, got {self.condition_steps!r}."
            )
        if self.prefetch_steps is None:
            self.prefetch_steps = self.condition_steps
        elif isinstance(self.prefetch_steps, bool) or not isinstance(
            self.prefetch_steps, int
        ):
            raise InterfaceValidationError("prefetch_steps must be an int when set.")
        elif self.prefetch_steps <= 0:
            raise InterfaceValidationError(
                f"prefetch_steps must be > 0, got {self.prefetch_steps!r}."
            )
        if self.condition_steps < int(self.prefetch_steps or self.condition_steps):
            raise InterfaceValidationError(
                "condition_steps must be >= prefetch_steps so the overlap window "
                "covers the request lead time."
            )

        if self.max_chunk_size is not None:
            if isinstance(self.max_chunk_size, bool) or not isinstance(
                self.max_chunk_size, int
            ):
                raise InterfaceValidationError(
                    "max_chunk_size must be an int when provided."
                )
            if self.max_chunk_size <= 0:
                raise InterfaceValidationError(
                    "max_chunk_size must be > 0 when provided, "
                    f"got {self.max_chunk_size!r}."
                )

        if self.control_hz is not None:
            self.control_hz = validate_positive_number(self.control_hz, "control_hz")
        if self.latency_budget_s is not None:
            self.latency_budget_s = validate_positive_number(
                self.latency_budget_s,
                "latency_budget_s",
            )
        if isinstance(self.safety_margin_steps, bool) or not isinstance(
            self.safety_margin_steps, int
        ):
            raise InterfaceValidationError("safety_margin_steps must be an int.")
        if self.safety_margin_steps < 0:
            raise InterfaceValidationError(
                f"safety_margin_steps must be >= 0, got {self.safety_margin_steps!r}."
            )
        self.warning_interval_s = validate_positive_number(
            self.warning_interval_s,
            "warning_interval_s",
        )
        self._latency_samples_s = deque(maxlen=128)

    def bind_control_hz(self, hz: float) -> None:
        """Bind the scheduler to one control frequency."""

        self.control_hz = validate_positive_number(hz, "control_hz")

    def reset(self) -> None:
        """Discard active/pending chunks but keep learned latency samples."""

        self._active_chunk.clear()
        self._active_index = 0
        self._global_step = 0
        self._ready_response = None
        self._last_warning_time = None
        if self._in_flight is not None:
            self._in_flight.cancel()
        self._in_flight = None

    def close(self) -> None:
        """Shut down background request execution."""

        self.reset()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    @property
    def estimated_p99_latency_s(self) -> float | None:
        """Return the observed p99 request latency when samples exist."""

        if not self._latency_samples_s:
            return None
        samples = sorted(self._latency_samples_s)
        index = max(math.ceil(len(samples) * 0.99) - 1, 0)
        return samples[index]

    def minimum_prefetch_steps(self) -> int:
        """Return the minimum prefetch window implied by observed latency.

        Practical rule: ``ceil(p99_latency_s * control_hz) + safety_margin_steps``.
        """

        if self.control_hz is None:
            return int(self.prefetch_steps or self.condition_steps)

        latency_s = self.estimated_p99_latency_s
        if latency_s is None:
            latency_s = self.latency_budget_s
        if latency_s is None:
            return int(self.prefetch_steps or self.condition_steps)

        return max(
            math.ceil(latency_s * self.control_hz) + self.safety_margin_steps,
            1,
        )

    def minimum_condition_steps(self) -> int:
        """Return the overlap size that fully covers the same latency window."""

        return self.minimum_prefetch_steps()

    def effective_prefetch_steps(self) -> int:
        """Return the active prefetch threshold used by the scheduler."""

        return max(
            int(self.prefetch_steps or self.condition_steps),
            self.minimum_prefetch_steps(),
        )

    def _emit_warning(self, message: str) -> None:
        """Emit one throttled runtime warning."""

        if self.warning_emitter is not None:
            self.warning_emitter(message)
            return
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _maybe_warn(self, message: str) -> None:
        """Emit one throttled warning when enough time has passed."""

        now = float(self.clock())
        if (
            self._last_warning_time is not None
            and now - self._last_warning_time < self.warning_interval_s
        ):
            return
        self._emit_warning(message)
        self._last_warning_time = now

    def _maybe_warn_timing_budget(self) -> None:
        """Warn when overlap/prefetch settings are smaller than latency suggests."""

        minimum_prefetch = self.minimum_prefetch_steps()
        minimum_condition = self.minimum_condition_steps()
        configured_prefetch = int(self.prefetch_steps or self.condition_steps)
        issues: list[str] = []
        if configured_prefetch < minimum_prefetch:
            issues.append(
                "prefetch_steps="
                f"{configured_prefetch}, recommended minimum={minimum_prefetch}"
            )
        if self.condition_steps < minimum_condition:
            issues.append(
                "condition_steps="
                f"{self.condition_steps}, recommended minimum={minimum_condition}"
            )
        if not issues:
            return

        hz_text = "unknown" if self.control_hz is None else f"{self.control_hz:.3f}"
        self._maybe_warn(
            "AsyncInference observed latency that exceeds the configured overlap "
            "window: "
            + "; ".join(issues)
            + f"; control_hz={hz_text}."
        )

    def _clone_action(self, action: Action) -> Action:
        """Return a detached copy of one standardized action."""

        return Action(
            mode=action.mode,
            value=list(action.value),
            gripper=action.gripper,
            ref_frame=action.ref_frame,
            dt=action.dt,
        )

    def _normalize_plan(self, plan: ActionPlan) -> list[Action]:
        """Coerce and validate one action plan or chunk."""

        if isinstance(plan, (Action, Mapping)):
            normalized = [as_action(plan)]
        elif isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
            raise InterfaceValidationError(
                "chunk_provider must return an action-like object or a sequence of "
                f"action-like objects, got {type(plan).__name__}."
            )
        else:
            normalized = [as_action(item) for item in plan]

        if self.max_chunk_size is not None:
            normalized = normalized[: self.max_chunk_size]
        if not normalized:
            raise InterfaceValidationError(
                "AsyncInference received an empty action chunk."
            )

        for action in normalized:
            validate_action(action)
        return [self._clone_action(action) for action in normalized]

    def _build_request(self) -> ChunkRequest:
        """Build one chunk request from the current scheduler state."""

        active_length = len(self._active_chunk)
        history_end = active_length
        history_start = max(history_end - self.condition_steps, 0)
        history_actions = [
            self._clone_action(action)
            for action in self._active_chunk[history_start:history_end]
        ]
        return ChunkRequest(
            request_step=self._global_step,
            request_time_s=float(self.clock()),
            history_start=history_start,
            history_end=history_end,
            active_chunk_length=active_length,
            remaining_steps=max(active_length - self._active_index, 0),
            condition_steps=self.condition_steps,
            prefetch_steps=self.effective_prefetch_steps(),
            history_actions=history_actions,
        )

    def _call_chunk_provider(
        self,
        source: object,
        frame: Frame,
        request: ChunkRequest,
        *,
        fallback_action_source: FallbackActionSource | None,
    ) -> list[Action]:
        """Run the configured chunk provider or fall back to single-step action."""

        raw_plan: ActionPlan
        provider = self.chunk_provider
        if provider is not None:
            try:
                raw_plan = provider(source, frame, request)
            except Exception as exc:
                raise InterfaceValidationError(
                    f"chunk_provider raised {type(exc).__name__}: {exc}"
                ) from exc
        elif self.plan_provider is not None:
            try:
                raw_plan = self.plan_provider(source, frame)
            except Exception as exc:
                raise InterfaceValidationError(
                    f"plan_provider raised {type(exc).__name__}: {exc}"
                ) from exc
        elif fallback_action_source is not None:
            try:
                raw_plan = fallback_action_source(frame)
            except Exception as exc:
                raise InterfaceValidationError(
                    f"fallback action source raised {type(exc).__name__}: {exc}"
                ) from exc
        else:
            raise InterfaceValidationError(
                "AsyncInference needs chunk_provider=..., plan_provider=..., or "
                "a fallback single-step action source."
            )

        return self._normalize_plan(raw_plan)

    def _execute_request(
        self,
        source: object,
        frame: Frame,
        request: ChunkRequest,
        fallback_action_source: FallbackActionSource | None,
    ) -> _CompletedChunk:
        """Execute one chunk request and capture completion timing."""

        plan = self._call_chunk_provider(
            source,
            frame,
            request,
            fallback_action_source=fallback_action_source,
        )
        return _CompletedChunk(
            request=request,
            plan=plan,
            completed_at_s=float(self.clock()),
        )

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Create the background executor lazily."""

        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="embodia-async-inference",
            )
        return self._executor

    def _submit_request(
        self,
        source: object,
        frame: Frame,
        request: ChunkRequest,
        fallback_action_source: FallbackActionSource | None,
    ) -> Future[_CompletedChunk]:
        """Submit one async chunk request."""

        executor = self._ensure_executor()
        return executor.submit(
            self._execute_request,
            source,
            frame,
            request,
            fallback_action_source,
        )

    def _collect_in_flight(self, *, block: bool) -> None:
        """Move one completed async request into the ready slot when available."""

        if self._in_flight is None:
            return
        if not block and not self._in_flight.done():
            return

        completed = self._in_flight.result()
        latency_s = max(
            completed.completed_at_s - completed.request.request_time_s,
            0.0,
        )
        self._latency_samples_s.append(latency_s)
        self._maybe_warn_timing_budget()
        self._ready_response = completed
        self._in_flight = None

    def _activate_ready_response(self) -> bool:
        """Try to activate one completed chunk response."""

        if self._ready_response is None:
            return False

        completed = self._ready_response
        consumed_since_request = max(
            self._global_step - completed.request.request_step,
            0,
        )
        if consumed_since_request >= completed.request.condition_steps:
            self._maybe_warn(
                "AsyncInference consumed the full configured overlap window "
                "before the next chunk became active; smooth handoff may "
                "degrade."
            )

        if consumed_since_request >= len(completed.plan):
            self._ready_response = None
            return False

        # Drop the prefix that became stale while the async request was in
        # flight, but keep any still-future actions so the runtime can salvage
        # late responses instead of discarding the whole chunk.
        trimmed = completed.plan[consumed_since_request:]
        if not trimmed:
            self._ready_response = None
            return False

        self._active_chunk = [self._clone_action(action) for action in trimmed]
        self._active_index = 0
        self._ready_response = None
        return True

    def _remaining_steps(self) -> int:
        """Return the number of actions left in the active chunk."""

        return max(len(self._active_chunk) - self._active_index, 0)

    def _maybe_start_prefetch(
        self,
        source: object,
        frame: Frame,
        fallback_action_source: FallbackActionSource | None,
    ) -> None:
        """Start the next async chunk request when the scheduler is close enough."""

        if self._in_flight is not None or self._ready_response is not None:
            return
        if self._remaining_steps() > self.effective_prefetch_steps():
            return

        request = self._build_request()
        self._in_flight = self._submit_request(
            source,
            frame,
            request,
            fallback_action_source,
        )

    def next_action(
        self,
        source: object,
        frame: Frame | Mapping[str, Any],
        *,
        fallback_action_source: FallbackActionSource | None = None,
    ) -> tuple[Action, bool]:
        """Return the next action and whether a new chunk became active."""

        normalized_frame = as_frame(frame)
        validate_frame(normalized_frame)
        plan_refreshed = False

        self._collect_in_flight(block=False)
        if self._remaining_steps() == 0:
            if self._ready_response is None:
                self._collect_in_flight(block=True)

            if self._remaining_steps() == 0 and not self._activate_ready_response():
                request = self._build_request()
                completed = self._execute_request(
                    source,
                    normalized_frame,
                    request,
                    fallback_action_source,
                )
                self._latency_samples_s.append(
                    max(completed.completed_at_s - request.request_time_s, 0.0)
                )
                self._ready_response = completed
                self._maybe_warn_timing_budget()
                if not self._activate_ready_response():
                    raise InterfaceValidationError(
                        "AsyncInference could not activate a usable chunk after "
                        "refresh."
                    )
            plan_refreshed = True

        action = self._clone_action(self._active_chunk[self._active_index])
        self._active_index += 1
        self._global_step += 1

        self._maybe_start_prefetch(
            source,
            normalized_frame,
            fallback_action_source,
        )
        return action, plan_refreshed


__all__ = ["AsyncInference"]
