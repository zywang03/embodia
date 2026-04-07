"""Chunk scheduling helpers for inference runtime."""

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
from ...core.schema import Action, Command, Frame
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
class _RobustMeanEstimator:
    """Small rolling robust-mean estimator for timing signals."""

    window_size: int = 64
    startup_ignore_samples: int = 1
    trim_ratio: float = 0.1
    mad_scale: float = 3.5
    _seen_samples: int = field(default=0, init=False, repr=False)
    _samples: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate estimator hyperparameters."""

        if isinstance(self.window_size, bool) or not isinstance(self.window_size, int):
            raise InterfaceValidationError("window_size must be an int.")
        if self.window_size <= 0:
            raise InterfaceValidationError(
                f"window_size must be > 0, got {self.window_size!r}."
            )
        if isinstance(self.startup_ignore_samples, bool) or not isinstance(
            self.startup_ignore_samples,
            int,
        ):
            raise InterfaceValidationError("startup_ignore_samples must be an int.")
        if self.startup_ignore_samples < 0:
            raise InterfaceValidationError(
                "startup_ignore_samples must be >= 0, got "
                f"{self.startup_ignore_samples!r}."
            )
        self.trim_ratio = float(self.trim_ratio)
        if not 0.0 <= self.trim_ratio < 0.5:
            raise InterfaceValidationError(
                f"trim_ratio must be in [0, 0.5), got {self.trim_ratio!r}."
            )
        self.mad_scale = validate_positive_number(self.mad_scale, "mad_scale")
        self._samples = deque(maxlen=self.window_size)

    def add(self, value: float) -> None:
        """Add one finite sample, skipping configured startup samples."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InterfaceValidationError(
                f"timing_sample_s must be a real number, got {type(value).__name__}."
            )
        sample = float(value)
        if not math.isfinite(sample):
            raise InterfaceValidationError(
                f"timing_sample_s must be finite, got {value!r}."
            )
        if sample <= 0.0:
            return
        self._seen_samples += 1
        if self._seen_samples <= self.startup_ignore_samples:
            return
        self._samples.append(sample)

    @property
    def mean(self) -> float | None:
        """Return the current robust mean."""

        if not self._samples:
            return None

        samples = sorted(self._samples)
        median = samples[len(samples) // 2]
        deviations = sorted(abs(sample - median) for sample in samples)
        mad = deviations[len(deviations) // 2]

        filtered = samples
        if mad > 0.0:
            scale = 1.4826 * mad
            threshold = self.mad_scale * scale
            filtered = [
                sample
                for sample in samples
                if abs(sample - median) <= threshold
            ]
            if not filtered:
                filtered = samples

        trim = int(len(filtered) * self.trim_ratio)
        if self.trim_ratio > 0.0 and len(filtered) >= 3:
            trim = max(trim, 1)
        if trim > 0 and len(filtered) > 2 * trim:
            filtered = filtered[trim:-trim]

        return sum(filtered) / len(filtered)


@dataclass(slots=True)
class _CompletedChunk:
    """One completed chunk request."""

    request: ChunkRequest
    plan: list[Action]
    completed_at_s: float


@dataclass(slots=True)
class ChunkScheduler:
    """Overlap-conditioned chunk scheduler with optional async prefetch.

    Users specify ``overlap_ratio`` instead of raw step counts. For a chunk of
    length ``N``, the scheduler uses ``ceil(N * overlap_ratio)`` overlap steps
    and, in async mode, estimates how many additional lead steps are needed to
    cover inference latency. The async request is started once:

    ``remaining_steps <= overlap_steps + estimated_latency_steps``

    When used through ``InferenceRuntime(mode='sync')``, embodia keeps the same
    overlap semantics but refreshes the next chunk synchronously once:

    ``remaining_steps <= overlap_steps``

    This keeps the policy API focused on ``obs -> action chunk`` while embodia
    owns the chunk handoff policy.
    """

    chunk_provider: ChunkProviderProtocol | ChunkProvider | None = None
    plan_provider: ActionPlanProviderProtocol | ActionPlanProvider | None = None
    overlap_ratio: float = 0.2
    min_overlap_steps: int = 1
    max_chunk_size: int | None = None
    control_hz: float | None = None
    latency_budget_s: float | None = None
    safety_margin_steps: int = 1
    timing_window_size: int = 64
    startup_ignore_inference_samples: int = 1
    timing_trim_ratio: float = 0.1
    warning_interval_s: float = 5.0
    clock: Callable[[], float] = time.perf_counter
    warning_emitter: Callable[[str], None] | None = None
    _active_chunk: list[Action] = field(default_factory=list, init=False, repr=False)
    _active_index: int = field(default=0, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)
    _ready_response: _CompletedChunk | None = field(default=None, init=False, repr=False)
    _in_flight: Future[_CompletedChunk] | None = field(default=None, init=False, repr=False)
    _latency_samples_s: deque[float] = field(init=False, repr=False)
    _step_time_estimator: _RobustMeanEstimator = field(init=False, repr=False)
    _inference_time_estimator: _RobustMeanEstimator = field(init=False, repr=False)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _last_warning_time: float | None = field(default=None, init=False, repr=False)
    _last_action_emitted_at_s: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and initialize runtime state."""

        if self.chunk_provider is not None and self.plan_provider is not None:
            raise InterfaceValidationError(
                "ChunkScheduler accepts chunk_provider or plan_provider, not both."
            )
        if isinstance(self.overlap_ratio, bool) or not isinstance(
            self.overlap_ratio,
            (int, float),
        ):
            raise InterfaceValidationError(
                "overlap_ratio must be a real number in the range (0, 1)."
            )
        self.overlap_ratio = float(self.overlap_ratio)
        if not 0.0 < self.overlap_ratio < 1.0:
            raise InterfaceValidationError(
                "overlap_ratio must be in the range (0, 1), got "
                f"{self.overlap_ratio!r}."
            )
        if isinstance(self.min_overlap_steps, bool) or not isinstance(
            self.min_overlap_steps,
            int,
        ):
            raise InterfaceValidationError(
                "min_overlap_steps must be an int."
            )
        if self.min_overlap_steps < 0:
            raise InterfaceValidationError(
                f"min_overlap_steps must be >= 0, got {self.min_overlap_steps!r}."
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
        if isinstance(self.timing_window_size, bool) or not isinstance(
            self.timing_window_size,
            int,
        ):
            raise InterfaceValidationError("timing_window_size must be an int.")
        if self.timing_window_size <= 0:
            raise InterfaceValidationError(
                f"timing_window_size must be > 0, got {self.timing_window_size!r}."
            )
        if isinstance(self.startup_ignore_inference_samples, bool) or not isinstance(
            self.startup_ignore_inference_samples,
            int,
        ):
            raise InterfaceValidationError(
                "startup_ignore_inference_samples must be an int."
            )
        if self.startup_ignore_inference_samples < 0:
            raise InterfaceValidationError(
                "startup_ignore_inference_samples must be >= 0, got "
                f"{self.startup_ignore_inference_samples!r}."
            )
        self.timing_trim_ratio = float(self.timing_trim_ratio)
        if not 0.0 <= self.timing_trim_ratio < 0.5:
            raise InterfaceValidationError(
                "timing_trim_ratio must be in [0, 0.5), got "
                f"{self.timing_trim_ratio!r}."
            )
        self._latency_samples_s = deque(maxlen=128)
        self._step_time_estimator = _RobustMeanEstimator(
            window_size=self.timing_window_size,
            startup_ignore_samples=0,
            trim_ratio=self.timing_trim_ratio,
        )
        self._inference_time_estimator = _RobustMeanEstimator(
            window_size=self.timing_window_size,
            startup_ignore_samples=self.startup_ignore_inference_samples,
            trim_ratio=self.timing_trim_ratio,
        )

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
        self._last_action_emitted_at_s = None
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

    @property
    def estimated_step_time_s(self) -> float | None:
        """Return the robust mean control-step interval."""

        return self._step_time_estimator.mean

    @property
    def estimated_inference_time_s(self) -> float | None:
        """Return the robust mean request latency, excluding startup samples."""

        return self._inference_time_estimator.mean

    def estimated_latency_steps(self) -> int:
        """Return the current request lead implied by observed latency."""

        latency_s = self.estimated_inference_time_s
        if latency_s is None:
            latency_s = self.estimated_p99_latency_s
        if latency_s is None:
            latency_s = self.latency_budget_s
        if latency_s is None or latency_s <= 0.0:
            return 0

        step_time_s = self.estimated_step_time_s
        if step_time_s is None:
            if self.control_hz is None:
                return 0
            step_time_s = 1.0 / self.control_hz

        return max(math.ceil(latency_s / step_time_s) + self.safety_margin_steps, 0)

    def overlap_steps_for_chunk(self, chunk_length: int) -> int:
        """Return the overlap size for one concrete chunk length."""

        if isinstance(chunk_length, bool) or not isinstance(chunk_length, int):
            raise InterfaceValidationError(
                f"chunk_length must be an int, got {type(chunk_length).__name__}."
            )
        if chunk_length < 0:
            raise InterfaceValidationError(
                f"chunk_length must be >= 0, got {chunk_length!r}."
            )
        if chunk_length == 0:
            return 0

        computed = math.ceil(chunk_length * self.overlap_ratio)
        return min(max(computed, self.min_overlap_steps), chunk_length)

    def request_trigger_steps(
        self,
        chunk_length: int,
        *,
        include_latency: bool = True,
    ) -> int:
        """Return the remaining-step threshold that starts the next request."""

        trigger_steps = self.overlap_steps_for_chunk(chunk_length)
        if include_latency:
            trigger_steps += self.estimated_latency_steps()
        return trigger_steps

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

    def _record_inference_time(self, latency_s: float) -> None:
        """Record one request latency into the timing estimators."""

        self._latency_samples_s.append(latency_s)
        self._inference_time_estimator.add(latency_s)

    def _record_action_emission(self, emitted_at_s: float) -> None:
        """Record the observed control-step interval between two emissions."""

        if self._last_action_emitted_at_s is not None:
            interval_s = max(emitted_at_s - self._last_action_emitted_at_s, 0.0)
            if interval_s > 0.0:
                self._step_time_estimator.add(interval_s)
        self._last_action_emitted_at_s = emitted_at_s

    def _maybe_warn_timing_budget(
        self,
        chunk_length: int,
        *,
        include_latency: bool,
    ) -> None:
        """Warn when latency consumes most or all of one chunk."""

        if not include_latency:
            return
        if chunk_length <= 0:
            return
        overlap_steps = self.overlap_steps_for_chunk(chunk_length)
        latency_steps = self.estimated_latency_steps()
        trigger_steps = self.request_trigger_steps(
            chunk_length,
            include_latency=True,
        )
        if trigger_steps < chunk_length:
            return

        hz_text = "unknown" if self.control_hz is None else f"{self.control_hz:.3f}"
        self._maybe_warn(
            "ChunkScheduler estimated latency requires requesting the next chunk "
            "before or immediately when the current chunk becomes active: "
            f"chunk_length={chunk_length}, overlap_steps={overlap_steps}, "
            f"latency_steps={latency_steps}, request_trigger_steps={trigger_steps}, "
            f"control_hz={hz_text}."
        )

    def _clone_action(self, action: Action) -> Action:
        """Return a detached copy of one standardized action."""

        return Action(
            commands=[
                Command(
                    target=command.target,
                    kind=command.kind,
                    value=list(command.value),
                    ref_frame=command.ref_frame,
                    meta=dict(command.meta),
                )
                for command in action.commands
            ],
            dt=action.dt,
            meta=dict(action.meta),
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
                "ChunkScheduler received an empty action chunk."
            )

        for action in normalized:
            validate_action(action)
        return [self._clone_action(action) for action in normalized]

    def _build_request(self, *, include_latency: bool) -> ChunkRequest:
        """Build one chunk request from the current scheduler state."""

        active_length = len(self._active_chunk)
        overlap_steps = self.overlap_steps_for_chunk(active_length)
        history_end = active_length
        history_start = max(history_end - overlap_steps, 0)
        history_actions = [
            self._clone_action(action)
            for action in self._active_chunk[history_start:history_end]
        ]
        latency_steps = self.estimated_latency_steps() if include_latency else 0
        current_chunk_start_step = max(self._global_step - self._active_index, 0)
        plan_start_step = current_chunk_start_step + history_start
        return ChunkRequest(
            request_step=self._global_step,
            request_time_s=float(self.clock()),
            history_start=history_start,
            history_end=history_end,
            active_chunk_length=active_length,
            remaining_steps=max(active_length - self._active_index, 0),
            overlap_steps=overlap_steps,
            latency_steps=latency_steps,
            request_trigger_steps=self.request_trigger_steps(
                active_length,
                include_latency=include_latency,
            ),
            plan_start_step=plan_start_step,
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
                "ChunkScheduler needs chunk_provider=..., plan_provider=..., or "
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
        if block and not self._in_flight.done():
            self._maybe_warn(
                "ChunkScheduler had to wait for the next chunk because observed "
                "latency exceeded the current request lead window."
            )

        completed = self._in_flight.result()
        latency_s = max(
            completed.completed_at_s - completed.request.request_time_s,
            0.0,
        )
        self._record_inference_time(latency_s)
        self._maybe_warn_timing_budget(
            completed.request.active_chunk_length,
            include_latency=True,
        )
        self._ready_response = completed
        self._in_flight = None

    def _activate_ready_response(self) -> bool:
        """Try to activate one completed chunk response."""

        if self._ready_response is None:
            return False

        completed = self._ready_response
        consumed_since_plan_start = max(
            self._global_step - completed.request.plan_start_step,
            0,
        )
        if consumed_since_plan_start >= len(completed.plan):
            self._ready_response = None
            return False

        # Drop the prefix that was already covered by the previously active
        # chunk while the request was in flight. The returned chunk is expected
        # to begin at request.plan_start_step, not at request.request_step.
        trimmed = completed.plan[consumed_since_plan_start:]
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

    def _refresh_ready_response(
        self,
        source: object,
        frame: Frame,
        fallback_action_source: FallbackActionSource | None,
        *,
        include_latency: bool,
    ) -> None:
        """Refresh the next chunk immediately and store it in the ready slot."""

        request = self._build_request(include_latency=include_latency)
        completed = self._execute_request(
            source,
            frame,
            request,
            fallback_action_source,
        )
        self._record_inference_time(
            max(completed.completed_at_s - request.request_time_s, 0.0)
        )
        self._ready_response = completed
        self._maybe_warn_timing_budget(
            request.active_chunk_length,
            include_latency=include_latency,
        )

    def _maybe_start_async_prefetch(
        self,
        source: object,
        frame: Frame,
        fallback_action_source: FallbackActionSource | None,
    ) -> None:
        """Start the next async chunk request when the scheduler is close enough."""

        if self._in_flight is not None or self._ready_response is not None:
            return
        if self._remaining_steps() > self.request_trigger_steps(
            len(self._active_chunk),
            include_latency=True,
        ):
            return

        request = self._build_request(include_latency=True)
        self._in_flight = self._submit_request(
            source,
            frame,
            request,
            fallback_action_source,
        )

    def _maybe_refresh_sync(
        self,
        source: object,
        frame: Frame,
        fallback_action_source: FallbackActionSource | None,
    ) -> None:
        """Refresh the next chunk synchronously once overlap becomes small."""

        if self._in_flight is not None or self._ready_response is not None:
            return
        if self._remaining_steps() > self.request_trigger_steps(
            len(self._active_chunk),
            include_latency=False,
        ):
            return
        self._refresh_ready_response(
            source,
            frame,
            fallback_action_source,
            include_latency=False,
        )

    def next_action(
        self,
        source: object,
        frame: Frame | Mapping[str, Any],
        *,
        fallback_action_source: FallbackActionSource | None = None,
        prefetch_async: bool = True,
    ) -> tuple[Action, bool]:
        """Return the next action and whether a new chunk became active.

        When ``prefetch_async`` is ``True``, the next chunk is requested in the
        background using overlap plus estimated latency. When ``False``, embodia
        still refreshes chunks before exhaustion, but it does so synchronously
        and uses only the overlap window as the trigger.
        """

        normalized_frame = as_frame(frame)
        validate_frame(normalized_frame)
        plan_refreshed = False

        self._collect_in_flight(block=False)
        if self._remaining_steps() == 0:
            if prefetch_async and self._ready_response is None:
                self._collect_in_flight(block=True)

            if self._remaining_steps() == 0 and not self._activate_ready_response():
                self._refresh_ready_response(
                    source,
                    normalized_frame,
                    fallback_action_source,
                    include_latency=prefetch_async,
                )
                if not self._activate_ready_response():
                    raise InterfaceValidationError(
                        "ChunkScheduler could not activate a usable chunk after "
                        "refresh."
                    )
            plan_refreshed = True

        action = self._clone_action(self._active_chunk[self._active_index])
        self._active_index += 1
        self._global_step += 1
        self._record_action_emission(float(self.clock()))

        if prefetch_async:
            self._maybe_start_async_prefetch(
                source,
                normalized_frame,
                fallback_action_source,
            )
        else:
            self._maybe_refresh_sync(
                source,
                normalized_frame,
                fallback_action_source,
            )
        return action, plan_refreshed


__all__ = ["ChunkScheduler"]
