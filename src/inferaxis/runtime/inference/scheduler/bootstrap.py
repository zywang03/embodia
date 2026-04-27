"""Startup bootstrap and validation helpers for chunk scheduling."""

from __future__ import annotations

from collections.abc import Sequence
import warnings

from ....core.errors import InterfaceValidationError
from ....core.schema import Action, Frame
from ....shared.coerce import as_frame_fast
from ...checks import validate_frame
from .state import _CompletedChunk


_SLOW_RTC_WARMUP_THRESHOLD_S = 0.5


def _validate_startup_execution_window(self, completed: _CompletedChunk) -> None:
    """Validate startup delay and execution-window constraints once."""

    if (
        self._startup_execution_window_validated
        or not self.enable_rtc
        or self.execution_steps is None
    ):
        return

    raw_delay_steps = self._estimated_request_latency_steps(
        control_latency_steps=self.estimated_latency_steps(),
        buffer_actions=completed.prepared_actions,
        execution_buffer_steps=0,
    )
    self._check_execution_window_delay(
        raw_delay_steps=raw_delay_steps,
    )
    self._startup_execution_window_validated = True


def _should_retry_with_local_rtc_seed(
    self,
    *,
    completed: _CompletedChunk,
    rtc_seed_chunk: list[Action] | None,
) -> bool:
    """Return whether startup should issue one extra RTC-seeded request."""

    if not self.enable_rtc:
        return False
    if completed.request.rtc_args is not None:
        return False
    if rtc_seed_chunk is not None:
        return False
    if self._raw_buffer.has_actions:
        return False
    return True


def _confirm_slow_rtc_bootstrap_request(
    self,
    *,
    inference_time_s: float,
) -> None:
    """Apply the configured slow RTC bootstrap policy."""

    if inference_time_s <= _SLOW_RTC_WARMUP_THRESHOLD_S:
        return

    threshold_ms = _SLOW_RTC_WARMUP_THRESHOLD_S * 1000.0
    duration_ms = inference_time_s * 1000.0
    message = (
        "The last RTC warmup request carrying prev_action_chunk took "
        f"{duration_ms:.1f} ms, exceeding the {threshold_ms:.1f} ms "
        "startup threshold."
    )
    if self.slow_rtc_bootstrap == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    if self.slow_rtc_bootstrap == "error":
        raise InterfaceValidationError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    try:
        response = input(f"{message} Continue startup anyway? [y/N]: ")
    except EOFError as exc:
        raise InterfaceValidationError(
            "RTC startup warmup needs confirmation after a slow "
            "prev_action_chunk request, but stdin is not interactive."
        ) from exc

    if response.strip().lower() not in {"y", "yes"}:
        raise InterfaceValidationError(
            "RTC startup warmup aborted after a slow prev_action_chunk request."
        )


def _bootstrap_async_latency(
    self,
    frame: Frame,
) -> _CompletedChunk | None:
    """Warm up async latency estimation with request-only probes."""

    if self.control_period_s is None or self.latency_estimate_ready():
        return None

    total_requests = self.warmup_requests + self.profile_delay_requests
    if total_requests <= 0:
        self._startup_latency_bootstrap_complete = True
        return None

    reusable_completed: _CompletedChunk | None = None
    profiled_steps: list[int] = []
    last_rtc_bootstrap_duration_s: float | None = None
    rtc_seed_chunk: Sequence[Action] | None = None
    for request_index in range(total_requests):
        job = self._build_request_job(
            include_latency=False,
            rtc_seed_chunk=rtc_seed_chunk,
        )
        request_start = float(self.clock())
        completed = self._execute_request(frame, job)
        inference_time_s = max(float(self.clock()) - request_start, 0.0)
        if completed.request.rtc_args is not None:
            last_rtc_bootstrap_duration_s = inference_time_s
        if self.enable_rtc:
            rtc_seed_chunk = completed.prepared_actions
        observed_steps = self._observed_latency_steps_from_duration(
            inference_time_s,
        )
        if request_index >= self.warmup_requests:
            profiled_steps.append(observed_steps)
        reusable_completed = completed
        if request_index < total_requests - 1 and self.live_profile is not None:
            self.live_profile.record_completed_without_accept(  # type: ignore[attr-defined]
                request_index=completed.request_index,
                request_step=completed.request.request_step,
                actions=completed.prepared_actions,
            )

    if last_rtc_bootstrap_duration_s is not None:
        self._confirm_slow_rtc_bootstrap_request(
            inference_time_s=last_rtc_bootstrap_duration_s,
        )

    if profiled_steps:
        self._latency_steps_estimate = float(sum(profiled_steps) / len(profiled_steps))
    else:
        self._latency_steps_estimate = max(self._latency_steps_estimate, 1.0)
    self._latency_observation_count = self.warmup_requests
    self._startup_latency_bootstrap_complete = True
    if reusable_completed is not None:
        self._validate_startup_execution_window(reusable_completed)

    return reusable_completed


def _maybe_complete_startup_validation(self) -> None:
    """Disable hot-path validation once startup checks have finished."""

    if not self.startup_validation_only:
        return
    if not self.latency_estimate_ready():
        return
    self._startup_validation_complete = True


def bootstrap(self, frame: Frame, *, validate_frame_input: bool = True) -> bool:
    """Run startup warmup/profile requests and seed the first executable chunk."""

    normalized_frame = as_frame_fast(frame)
    if validate_frame_input and self.runtime_validation_enabled():
        validate_frame(normalized_frame)

    if self.control_period_s is None or self.latency_estimate_ready():
        return False
    if self._raw_buffer.has_actions or self._pending_future is not None:
        return False

    bootstrapped_chunk = self._bootstrap_async_latency(normalized_frame)
    if bootstrapped_chunk is None:
        return False
    if not self._integrate_completed_chunk(bootstrapped_chunk):
        raise InterfaceValidationError(
            "ChunkScheduler could not produce a usable warmup chunk."
        )
    return True
