"""Serializable data models and estimators for inference profiling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
from os import PathLike
from pathlib import Path
from typing import Any

from ....core.errors import InterfaceValidationError
from ....shared.common import validate_positive_number
from ..engine import InferenceMode


@dataclass(slots=True)
class _RobustMeanEstimator:
    """Small rolling robust-mean estimator for profiling signals."""

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
                sample for sample in samples if abs(sample - median) <= threshold
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
class _ProfiledRequestSample:
    """One measured request sample from sync profiling."""

    request_index: int
    chunk_steps: int
    inference_time_s: float
    ignored_inference_sample: bool
    observed_latency_steps: int


@dataclass(slots=True)
class AsyncBufferTraceStep:
    """One simulated async-control step for buffer visualization."""

    step_index: int
    buffer_before_accept: int
    buffer_after_accept: int
    buffer_after_execute: int
    steps_before_request: int
    executed_wait_raw_steps: int
    latency_steps_estimate: int
    reference_chunk_steps: int
    request_started: bool = False
    started_request_index: int | None = None
    request_completed: bool = False
    completed_request_index: int | None = None
    executed_request_index: int | None = None
    blended_from_request_index: int | None = None
    underrun: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Export one trace step into a JSON-safe dictionary."""

        return {
            "step_index": self.step_index,
            "buffer_before_accept": self.buffer_before_accept,
            "buffer_after_accept": self.buffer_after_accept,
            "buffer_after_execute": self.buffer_after_execute,
            "steps_before_request": self.steps_before_request,
            "executed_wait_raw_steps": self.executed_wait_raw_steps,
            "latency_steps_estimate": self.latency_steps_estimate,
            "reference_chunk_steps": self.reference_chunk_steps,
            "request_started": self.request_started,
            "started_request_index": self.started_request_index,
            "request_completed": self.request_completed,
            "completed_request_index": self.completed_request_index,
            "executed_request_index": self.executed_request_index,
            "blended_from_request_index": self.blended_from_request_index,
            "underrun": self.underrun,
        }


@dataclass(slots=True)
class AsyncBufferTraceRequest:
    """One simulated async request event."""

    request_index: int
    start_step: int
    reply_step: int
    chunk_steps: int
    steps_before_request: int
    observed_latency_steps: int
    executed_wait_raw_steps: int
    aligned_chunk_steps: int
    blended_steps_after_accept: int
    ignored_inference_sample: bool

    def to_dict(self) -> dict[str, Any]:
        """Export one simulated request into a JSON-safe dictionary."""

        return {
            "request_index": self.request_index,
            "start_step": self.start_step,
            "reply_step": self.reply_step,
            "chunk_steps": self.chunk_steps,
            "steps_before_request": self.steps_before_request,
            "observed_latency_steps": self.observed_latency_steps,
            "executed_wait_raw_steps": self.executed_wait_raw_steps,
            "aligned_chunk_steps": self.aligned_chunk_steps,
            "blended_steps_after_accept": self.blended_steps_after_accept,
            "ignored_inference_sample": self.ignored_inference_sample,
        }


@dataclass(slots=True)
class AsyncBufferTrace:
    """Simulated async buffer evolution built from sync-profile samples."""

    target_hz: float
    target_period_s: float
    steps_before_request: int
    latency_ema_beta: float
    initial_latency_steps: float
    steps: list[AsyncBufferTraceStep]
    requests: list[AsyncBufferTraceRequest]

    def to_dict(self) -> dict[str, Any]:
        """Export the full trace into a JSON-safe dictionary."""

        return {
            "target_hz": self.target_hz,
            "target_period_s": self.target_period_s,
            "steps_before_request": self.steps_before_request,
            "latency_ema_beta": self.latency_ema_beta,
            "initial_latency_steps": self.initial_latency_steps,
            "steps": [step.to_dict() for step in self.steps],
            "requests": [request.to_dict() for request in self.requests],
        }

    def write_json(self, path: str | PathLike[str]) -> None:
        """Write the trace to one JSON file."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


@dataclass(slots=True)
class SyncInferenceProfile:
    """Serializable summary of one sync-inference profiling run."""

    target_hz: float
    target_period_s: float
    steps: int
    execute_action: bool
    startup_ignored_inference_samples: int
    stable_inference_sample_count: int
    estimated_step_time_s: float | None
    estimated_inference_time_s: float | None
    estimated_control_hz: float | None
    estimated_latency_steps: int | None
    estimated_chunk_steps: float | None = None
    estimated_max_buffered_hz: float | None = None
    steps_before_request: int = 0
    async_buffer_trace: AsyncBufferTrace | None = field(default=None, repr=False)

    def to_dict(self, *, include_trace: bool = False) -> dict[str, Any]:
        """Export the profile into a plain JSON-serializable dictionary."""

        payload = {
            "target_hz": self.target_hz,
            "target_period_s": self.target_period_s,
            "steps": self.steps,
            "execute_action": self.execute_action,
            "startup_ignored_inference_samples": self.startup_ignored_inference_samples,
            "stable_inference_sample_count": self.stable_inference_sample_count,
            "estimated_step_time_s": self.estimated_step_time_s,
            "estimated_inference_time_s": self.estimated_inference_time_s,
            "estimated_control_hz": self.estimated_control_hz,
            "estimated_latency_steps": self.estimated_latency_steps,
            "estimated_chunk_steps": self.estimated_chunk_steps,
            "estimated_max_buffered_hz": self.estimated_max_buffered_hz,
            "steps_before_request": self.steps_before_request,
        }
        if include_trace and self.async_buffer_trace is not None:
            payload["async_buffer_trace"] = self.async_buffer_trace.to_dict()
        return payload

    def write_json(
        self,
        path: str | PathLike[str],
        *,
        include_trace: bool = False,
    ) -> None:
        """Write the profile summary to one JSON file."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(
                self.to_dict(include_trace=include_trace), indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )

    def write_async_buffer_trace_json(self, path: str | PathLike[str]) -> None:
        """Write the simulated async buffer trace to one JSON file."""

        if self.async_buffer_trace is None:
            raise InterfaceValidationError(
                "This profile does not contain an async buffer trace."
            )
        self.async_buffer_trace.write_json(path)


@dataclass(slots=True)
class RuntimeProfileRequest:
    """One recorded async request from live runtime profiling."""

    request_index: int
    request_step: int
    launch_control_step: int
    launch_time_s: float
    reply_time_s: float | None
    accepted_time_s: float | None
    request_duration_s: float | None
    prepare_duration_s: float | None
    accept_delay_s: float | None
    usable_latency_s: float | None
    latency_hint_raw_steps: int
    waited_control_steps: int | None
    stale_raw_steps: int | None
    returned_chunk_length: int | None
    accepted_chunk_length: int | None
    accepted: bool
    dropped_as_stale: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Export the live request record into one JSON-safe dictionary."""

        return {
            "request_index": self.request_index,
            "request_step": self.request_step,
            "launch_control_step": self.launch_control_step,
            "launch_time_s": self.launch_time_s,
            "reply_time_s": self.reply_time_s,
            "accepted_time_s": self.accepted_time_s,
            "request_duration_s": self.request_duration_s,
            "prepare_duration_s": self.prepare_duration_s,
            "accept_delay_s": self.accept_delay_s,
            "usable_latency_s": self.usable_latency_s,
            "latency_hint_raw_steps": self.latency_hint_raw_steps,
            "waited_control_steps": self.waited_control_steps,
            "stale_raw_steps": self.stale_raw_steps,
            "returned_chunk_length": self.returned_chunk_length,
            "accepted_chunk_length": self.accepted_chunk_length,
            "accepted": self.accepted,
            "dropped_as_stale": self.dropped_as_stale,
            "error": self.error,
        }


@dataclass(slots=True)
class RuntimeProfileActionCommand:
    """One command value recorded for an emitted runtime action."""

    target: str
    command: str
    value: list[float]
    ref_frame: str | None

    def to_dict(self) -> dict[str, Any]:
        """Export the recorded command into a JSON-safe dictionary."""

        return {
            "target": self.target,
            "command": self.command,
            "value": list(self.value),
            "ref_frame": self.ref_frame,
        }


@dataclass(slots=True)
class RuntimeProfileActionStep:
    """One emitted action sample from live runtime profiling."""

    step_index: int
    action_time_s: float
    plan_refreshed: bool
    control_wait_s: float
    buffer_size: int | None
    execution_buffer_size: int | None
    raw_commands: list[RuntimeProfileActionCommand]
    action_commands: list[RuntimeProfileActionCommand]

    def to_dict(self) -> dict[str, Any]:
        """Export the emitted action sample into a JSON-safe dictionary."""

        return {
            "step_index": self.step_index,
            "action_time_s": self.action_time_s,
            "plan_refreshed": self.plan_refreshed,
            "control_wait_s": self.control_wait_s,
            "buffer_size": self.buffer_size,
            "execution_buffer_size": self.execution_buffer_size,
            "raw_commands": [command.to_dict() for command in self.raw_commands],
            "action_commands": [command.to_dict() for command in self.action_commands],
        }


@dataclass(slots=True)
class RuntimeProfileChunkAction:
    """One action candidate from a returned chunk, annotated after integration."""

    request_index: int
    request_step: int
    action_index: int
    step_index: int
    status: str
    commands: list[RuntimeProfileActionCommand]

    def to_dict(self) -> dict[str, Any]:
        """Export the chunk action candidate into a JSON-safe dictionary."""

        return {
            "request_index": self.request_index,
            "request_step": self.request_step,
            "action_index": self.action_index,
            "step_index": self.step_index,
            "status": self.status,
            "commands": [command.to_dict() for command in self.commands],
        }


@dataclass(slots=True)
class RuntimeInferenceProfile:
    """Serializable live request-level profile captured from async runtime use."""

    mode: str
    config: dict[str, Any]
    requests: list[RuntimeProfileRequest]
    action_steps: list[RuntimeProfileActionStep] = field(default_factory=list)
    chunk_actions: list[RuntimeProfileChunkAction] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return one compact request-level summary for the full session."""

        request_durations = [
            request.request_duration_s
            for request in self.requests
            if request.request_duration_s is not None
        ]
        usable_latencies = [
            request.usable_latency_s
            for request in self.requests
            if request.usable_latency_s is not None
        ]
        returned_chunk_lengths = [
            float(request.returned_chunk_length)
            for request in self.requests
            if request.returned_chunk_length is not None
        ]
        return {
            "total_requests": len(self.requests),
            "accepted_requests": sum(
                1 for request in self.requests if request.accepted
            ),
            "dropped_stale_requests": sum(
                1 for request in self.requests if request.dropped_as_stale
            ),
            "failed_requests": sum(
                1 for request in self.requests if request.error is not None
            ),
            "average_request_duration_s": (
                None
                if not request_durations
                else sum(request_durations) / len(request_durations)
            ),
            "average_usable_latency_s": (
                None
                if not usable_latencies
                else sum(usable_latencies) / len(usable_latencies)
            ),
            "max_usable_latency_s": (
                None if not usable_latencies else max(usable_latencies)
            ),
            "average_returned_chunk_length": (
                None
                if not returned_chunk_lengths
                else sum(returned_chunk_lengths) / len(returned_chunk_lengths)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export the live runtime profile into one JSON-safe dictionary."""

        return {
            "mode": self.mode,
            "config": dict(self.config),
            "summary": self.summary(),
            "requests": [request.to_dict() for request in self.requests],
            "action_steps": [step.to_dict() for step in self.action_steps],
            "chunk_actions": [action.to_dict() for action in self.chunk_actions],
        }

    def write_json(self, path: str | PathLike[str]) -> None:
        """Write the live runtime profile to one JSON file."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_html(self, path: str | PathLike[str]) -> None:
        """Write one interactive HTML chart for the live runtime profile."""

        from .render import _runtime_profile_html

        output_path = Path(path)
        output_path.write_text(
            _runtime_profile_html(self),
            encoding="utf-8",
        )


@dataclass(slots=True)
class InferenceModeRecommendation:
    """Serializable recommendation for choosing sync or async runtime mode."""

    target_hz: float
    target_period_s: float
    async_supported: bool
    sync_expected_to_meet_target: bool | None
    recommended_mode: InferenceMode
    reason: str
    profile: SyncInferenceProfile

    def to_dict(self) -> dict[str, Any]:
        """Export the recommendation into one JSON-serializable dictionary."""

        return {
            "target_hz": self.target_hz,
            "target_period_s": self.target_period_s,
            "async_supported": self.async_supported,
            "sync_expected_to_meet_target": self.sync_expected_to_meet_target,
            "recommended_mode": str(self.recommended_mode),
            "reason": self.reason,
            "profile": self.profile.to_dict(),
        }

    def write_json(self, path: str | PathLike[str]) -> None:
        """Write the recommendation to one JSON file."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
