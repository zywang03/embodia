"""Sync profiling helpers for choosing runtime modes and scheduling settings."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape
import json
import math
from os import PathLike
from pathlib import Path
import time
from typing import Any

import numpy as np

from ...core.errors import InterfaceValidationError
from ...core.schema import Frame
from ..flow import _execute_step_action, _resolve_step_frame
from ...shared.action_source import (
    ActionSink,
    ActionSource,
    FrameSource,
    first_action_and_plan_length_from_action_call,
    resolve_runtime_owner,
)
from ...shared.common import validate_positive_number
from .engine import InferenceMode


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
    trigger_threshold: int
    overlap_steps: int
    latency_steps_estimate: int
    reference_chunk_steps: int
    request_started: bool = False
    started_request_index: int | None = None
    request_completed: bool = False
    completed_request_index: int | None = None
    underrun: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Export one trace step into a JSON-safe dictionary."""

        return {
            "step_index": self.step_index,
            "buffer_before_accept": self.buffer_before_accept,
            "buffer_after_accept": self.buffer_after_accept,
            "buffer_after_execute": self.buffer_after_execute,
            "trigger_threshold": self.trigger_threshold,
            "overlap_steps": self.overlap_steps,
            "latency_steps_estimate": self.latency_steps_estimate,
            "reference_chunk_steps": self.reference_chunk_steps,
            "request_started": self.request_started,
            "started_request_index": self.started_request_index,
            "request_completed": self.request_completed,
            "completed_request_index": self.completed_request_index,
            "underrun": self.underrun,
        }


@dataclass(slots=True)
class AsyncBufferTraceRequest:
    """One simulated async request event."""

    request_index: int
    start_step: int
    reply_step: int
    chunk_steps: int
    overlap_steps: int
    observed_latency_steps: int
    aligned_chunk_steps: int
    ignored_inference_sample: bool

    def to_dict(self) -> dict[str, Any]:
        """Export one simulated request into a JSON-safe dictionary."""

        return {
            "request_index": self.request_index,
            "start_step": self.start_step,
            "reply_step": self.reply_step,
            "chunk_steps": self.chunk_steps,
            "overlap_steps": self.overlap_steps,
            "observed_latency_steps": self.observed_latency_steps,
            "aligned_chunk_steps": self.aligned_chunk_steps,
            "ignored_inference_sample": self.ignored_inference_sample,
        }


@dataclass(slots=True)
class AsyncBufferTrace:
    """Simulated async buffer evolution built from sync-profile samples."""

    target_hz: float
    target_period_s: float
    overlap_ratio: float
    latency_ema_beta: float
    initial_latency_steps: float
    steps: list[AsyncBufferTraceStep]
    requests: list[AsyncBufferTraceRequest]

    def to_dict(self) -> dict[str, Any]:
        """Export the full trace into a JSON-safe dictionary."""

        return {
            "target_hz": self.target_hz,
            "target_period_s": self.target_period_s,
            "overlap_ratio": self.overlap_ratio,
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

    def write_svg(self, path: str | PathLike[str]) -> None:
        """Write a dependency-free SVG plot for the simulated buffer trace."""

        output_path = Path(path)
        output_path.write_text(
            _async_buffer_trace_svg(self),
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
    estimated_max_overlap_safe_hz: float | None = None
    overlap_ratio: float | None = None
    suggested_overlap_steps: int | None = None
    suggested_request_trigger_steps: int | None = None
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
            "estimated_max_overlap_safe_hz": self.estimated_max_overlap_safe_hz,
            "overlap_ratio": self.overlap_ratio,
            "suggested_overlap_steps": self.suggested_overlap_steps,
            "suggested_request_trigger_steps": self.suggested_request_trigger_steps,
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
            json.dumps(self.to_dict(include_trace=include_trace), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_async_buffer_trace_json(self, path: str | PathLike[str]) -> None:
        """Write the simulated async buffer trace to one JSON file."""

        if self.async_buffer_trace is None:
            raise InterfaceValidationError(
                "This profile does not contain an async buffer trace."
            )
        self.async_buffer_trace.write_json(path)

    def write_async_buffer_trace_svg(self, path: str | PathLike[str]) -> None:
        """Write the simulated async buffer trace to one SVG file."""

        if self.async_buffer_trace is None:
            raise InterfaceValidationError(
                "This profile does not contain an async buffer trace."
            )
        self.async_buffer_trace.write_svg(path)


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


@dataclass(slots=True)
class _PendingAsyncTraceRequest:
    """One in-flight simulated async request."""

    sample: _ProfiledRequestSample
    start_step: int
    reply_step: int
    overlap_steps: int


def _observed_latency_steps(
    *,
    inference_time_s: float,
    target_period_s: float,
) -> int:
    """Convert one measured inference duration into control-step latency."""

    if inference_time_s <= 0.0:
        return 1
    return max(int(math.ceil(inference_time_s / target_period_s)), 1)


def _step_plot_points(
    values: list[int],
    *,
    x_count: int,
    y_max: int,
    margin_left: int,
    margin_top: int,
    plot_width: int,
    plot_height: int,
) -> str:
    """Build one simple step-plot polyline point string."""

    if not values:
        return ""

    def x_of(step: int) -> float:
        if x_count <= 0:
            return float(margin_left)
        return margin_left + (plot_width * step / x_count)

    def y_of(value: int) -> float:
        return margin_top + plot_height * (1.0 - (value / y_max))

    points: list[str] = []
    first_y = y_of(values[0])
    points.append(f"{x_of(0):.2f},{first_y:.2f}")
    for index, value in enumerate(values):
        left_x = x_of(index)
        right_x = x_of(index + 1)
        y = y_of(value)
        points.append(f"{left_x:.2f},{y:.2f}")
        points.append(f"{right_x:.2f},{y:.2f}")
    return " ".join(points)


def _async_buffer_trace_svg(trace: AsyncBufferTrace) -> str:
    """Render one lightweight SVG visualization for the async buffer trace."""

    steps = trace.steps
    width = 960
    height = 480
    margin_left = 64
    margin_right = 24
    margin_top = 52
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_count = max(len(steps), 1)

    max_buffer = max(
        [1]
        + [step.buffer_after_accept for step in steps]
        + [step.buffer_after_execute for step in steps]
        + [step.trigger_threshold for step in steps]
        + [request.chunk_steps for request in trace.requests]
    )
    y_max = max(max_buffer, 1)

    buffer_points = _step_plot_points(
        [step.buffer_after_accept for step in steps],
        x_count=x_count,
        y_max=y_max,
        margin_left=margin_left,
        margin_top=margin_top,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    threshold_points = _step_plot_points(
        [step.trigger_threshold for step in steps],
        x_count=x_count,
        y_max=y_max,
        margin_left=margin_left,
        margin_top=margin_top,
        plot_width=plot_width,
        plot_height=plot_height,
    )

    def x_of(step: int) -> float:
        return margin_left + (plot_width * step / x_count)

    def y_of(value: int) -> float:
        return margin_top + plot_height * (1.0 - (value / y_max))

    grid_lines: list[str] = []
    for y_tick in range(y_max + 1):
        y = y_of(y_tick)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" '
            'font-size="11" text-anchor="end" fill="#6b7280">'
            f"{y_tick}</text>"
        )

    x_ticks: list[str] = []
    x_tick_count = min(max(len(steps), 2), 8)
    for tick_index in range(x_tick_count + 1):
        step = round(len(steps) * tick_index / x_tick_count)
        x = x_of(step)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{height - margin_bottom}" stroke="#f3f4f6" stroke-width="1" />'
        )
        x_ticks.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 22}" '
            'font-size="11" text-anchor="middle" fill="#6b7280">'
            f"{step}</text>"
        )

    markers: list[str] = []
    for step in steps:
        x = x_of(step.step_index)
        y = y_of(step.buffer_after_accept)
        if step.request_started:
            markers.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" '
                'fill="#10b981" stroke="#065f46" stroke-width="1.2" />'
            )
        if step.request_completed:
            size = 5.0
            markers.append(
                "<polygon "
                f'points="{x:.2f},{y - size:.2f} {x + size:.2f},{y:.2f} '
                f'{x:.2f},{y + size:.2f} {x - size:.2f},{y:.2f}" '
                'fill="#f59e0b" stroke="#92400e" stroke-width="1.2" />'
            )
        if step.underrun:
            y0 = y_of(0)
            size = 5.0
            markers.append(
                f'<line x1="{x - size:.2f}" y1="{y0 - size:.2f}" '
                f'x2="{x + size:.2f}" y2="{y0 + size:.2f}" '
                'stroke="#dc2626" stroke-width="2" />'
            )
            markers.append(
                f'<line x1="{x - size:.2f}" y1="{y0 + size:.2f}" '
                f'x2="{x + size:.2f}" y2="{y0 - size:.2f}" '
                'stroke="#dc2626" stroke-width="2" />'
            )

    request_summary = (
        f"requests={len(trace.requests)} steps={len(trace.steps)} "
        f"target_hz={trace.target_hz:.2f} overlap_ratio={trace.overlap_ratio:.2f} "
        f"latency_beta={trace.latency_ema_beta:.2f}"
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Async buffer trace">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
        f'<text x="{margin_left}" y="24" font-size="18" font-weight="700" '
        'fill="#111827">Async Buffer Trace</text>'
        f'<text x="{margin_left}" y="42" font-size="12" fill="#4b5563">'
        f"{escape(request_summary)}</text>"
        + "".join(grid_lines)
        + "".join(x_ticks)
        + (
            f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" '
            f'points="{buffer_points}" />'
            if buffer_points
            else ""
        )
        + (
            f'<polyline fill="none" stroke="#6b7280" stroke-width="2" '
            'stroke-dasharray="6 4" '
            f'points="{threshold_points}" />'
            if threshold_points
            else ""
        )
        + "".join(markers)
        + f'<text x="{margin_left}" y="{height - 26}" font-size="12" fill="#111827">'
        "step</text>"
        + f'<text x="18" y="{margin_top - 12}" font-size="12" fill="#111827">'
        "buffer</text>"
        + f'<line x1="{margin_left + 140}" y1="22" x2="{margin_left + 176}" y2="22" '
        'stroke="#2563eb" stroke-width="2.5" />'
        + f'<text x="{margin_left + 184}" y="26" font-size="11" fill="#374151">'
        "buffer_after_accept</text>"
        + f'<line x1="{margin_left + 320}" y1="22" x2="{margin_left + 356}" y2="22" '
        'stroke="#6b7280" stroke-width="2" stroke-dasharray="6 4" />'
        + f'<text x="{margin_left + 364}" y="26" font-size="11" fill="#374151">'
        "trigger_threshold</text>"
        + f'<circle cx="{margin_left + 515}" cy="22" r="4.5" fill="#10b981" '
        'stroke="#065f46" stroke-width="1.2" />'
        + f'<text x="{margin_left + 526}" y="26" font-size="11" fill="#374151">'
        "request_started</text>"
        + f'<polygon points="{margin_left + 662},17 {margin_left + 667},22 '
        f'{margin_left + 662},27 {margin_left + 657},22" fill="#f59e0b" '
        'stroke="#92400e" stroke-width="1.2" />'
        + f'<text x="{margin_left + 674}" y="26" font-size="11" fill="#374151">'
        "request_completed</text>"
        + f'<line x1="{margin_left + 830}" y1="17" x2="{margin_left + 840}" y2="27" '
        'stroke="#dc2626" stroke-width="2" />'
        + f'<line x1="{margin_left + 830}" y1="27" x2="{margin_left + 840}" y2="17" '
        'stroke="#dc2626" stroke-width="2" />'
        + f'<text x="{margin_left + 848}" y="26" font-size="11" fill="#374151">'
        "underrun</text>"
        + "</svg>"
    )


def _build_async_buffer_trace(
    *,
    request_samples: list[_ProfiledRequestSample],
    target_hz: float,
    overlap_ratio: float | None,
    latency_ema_beta: float = 0.5,
    initial_latency_steps: float = 0.0,
) -> AsyncBufferTrace:
    """Simulate async buffer growth and depletion from sync-profiled requests."""

    target_period_s = 1.0 / target_hz
    ratio = 0.0 if overlap_ratio is None else float(overlap_ratio)
    if not request_samples:
        return AsyncBufferTrace(
            target_hz=target_hz,
            target_period_s=target_period_s,
            overlap_ratio=ratio,
            latency_ema_beta=latency_ema_beta,
            initial_latency_steps=initial_latency_steps,
            steps=[],
            requests=[],
        )

    first_sample = request_samples[0]
    buffer_steps = first_sample.chunk_steps
    reference_chunk_steps = first_sample.chunk_steps
    latency_steps_estimate = float(initial_latency_steps)
    request_cursor = 1
    pending: _PendingAsyncTraceRequest | None = None
    steps: list[AsyncBufferTraceStep] = []
    requests: list[AsyncBufferTraceRequest] = [
        AsyncBufferTraceRequest(
            request_index=first_sample.request_index,
            start_step=0,
            reply_step=0,
            chunk_steps=first_sample.chunk_steps,
            overlap_steps=0,
            observed_latency_steps=first_sample.observed_latency_steps,
            aligned_chunk_steps=first_sample.chunk_steps,
            ignored_inference_sample=first_sample.ignored_inference_sample,
        )
    ]

    max_steps = max(
        sum(sample.chunk_steps for sample in request_samples)
        + sum(sample.observed_latency_steps for sample in request_samples)
        + len(request_samples) * 4,
        1,
    )

    for step_index in range(max_steps):
        buffer_before_accept = buffer_steps
        request_started = False
        started_request_index: int | None = None
        request_completed = False
        completed_request_index: int | None = None

        if pending is not None and step_index >= pending.reply_step:
            waited_steps = max(step_index - pending.start_step, 0)
            if not pending.sample.ignored_inference_sample:
                latency_steps_estimate = (
                    (1.0 - latency_ema_beta) * latency_steps_estimate
                    + latency_ema_beta * float(waited_steps)
                )
            aligned_chunk_steps = max(pending.sample.chunk_steps - waited_steps, 0)
            requests.append(
                AsyncBufferTraceRequest(
                    request_index=pending.sample.request_index,
                    start_step=pending.start_step,
                    reply_step=step_index,
                    chunk_steps=pending.sample.chunk_steps,
                    overlap_steps=pending.overlap_steps,
                    observed_latency_steps=waited_steps,
                    aligned_chunk_steps=aligned_chunk_steps,
                    ignored_inference_sample=pending.sample.ignored_inference_sample,
                )
            )
            if aligned_chunk_steps > 0:
                buffer_steps = aligned_chunk_steps
                reference_chunk_steps = pending.sample.chunk_steps
            request_completed = True
            completed_request_index = pending.sample.request_index
            pending = None

        buffer_after_accept = buffer_steps
        overlap_steps = max(int(math.floor(reference_chunk_steps * ratio)), 0)
        latency_steps_ceiled = max(int(math.ceil(latency_steps_estimate)), 0)
        trigger_threshold = latency_steps_ceiled + overlap_steps

        if (
            pending is None
            and request_cursor < len(request_samples)
            and buffer_after_accept <= trigger_threshold
        ):
            sample = request_samples[request_cursor]
            pending = _PendingAsyncTraceRequest(
                sample=sample,
                start_step=step_index,
                reply_step=step_index + max(sample.observed_latency_steps, 1),
                overlap_steps=overlap_steps,
            )
            request_started = True
            started_request_index = sample.request_index
            request_cursor += 1

        underrun = buffer_after_accept <= 0
        if underrun:
            buffer_after_execute = 0
        else:
            buffer_steps = max(buffer_after_accept - 1, 0)
            buffer_after_execute = buffer_steps

        steps.append(
            AsyncBufferTraceStep(
                step_index=step_index,
                buffer_before_accept=buffer_before_accept,
                buffer_after_accept=buffer_after_accept,
                buffer_after_execute=buffer_after_execute,
                trigger_threshold=trigger_threshold,
                overlap_steps=overlap_steps,
                latency_steps_estimate=latency_steps_ceiled,
                reference_chunk_steps=reference_chunk_steps,
                request_started=request_started,
                started_request_index=started_request_index,
                request_completed=request_completed,
                completed_request_index=completed_request_index,
                underrun=underrun,
            )
        )

        if request_cursor >= len(request_samples) and pending is None and buffer_steps == 0:
            break

    return AsyncBufferTrace(
        target_hz=target_hz,
        target_period_s=target_period_s,
        overlap_ratio=ratio,
        latency_ema_beta=latency_ema_beta,
        initial_latency_steps=initial_latency_steps,
        steps=steps,
        requests=requests,
    )


def _summarize_array(
    value: np.ndarray,
    *,
    include_values: bool,
) -> Any:
    """Return one small JSON-safe summary for an ndarray payload."""

    if include_values and value.size <= 12 and value.ndim <= 2:
        return value.tolist()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _to_json_safe_summary(value: Any) -> Any:
    """Return one compact JSON-safe summary for request tracing."""

    if isinstance(value, np.ndarray):
        return _summarize_array(
            value,
            include_values=value.ndim <= 1,
        )
    if isinstance(value, dict):
        return {
            str(key): _to_json_safe_summary(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_safe_summary(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _frame_request_summary(frame: Frame) -> dict[str, Any]:
    """Build one compact request-trace summary for a frame."""

    summary: dict[str, Any] = {
        "images": {
            key: _summarize_array(value, include_values=False)
            for key, value in frame.images.items()
        },
        "state": {
            key: _summarize_array(value, include_values=True)
            for key, value in frame.state.items()
        },
    }
    if frame.task:
        summary["task"] = _to_json_safe_summary(frame.task)
    if frame.meta:
        summary["meta"] = _to_json_safe_summary(frame.meta)
    if frame.sequence_id is not None:
        summary["sequence_id"] = frame.sequence_id
    return summary


def _emit_profile_request_trace(
    *,
    request_log_fn: Callable[[str], None],
    request_index: int,
    ignored_inference_sample: bool,
    frame: Frame,
    returned_plan_length: int,
    inference_time_s: float,
    step_time_s: float,
) -> None:
    """Emit one human-readable profile trace line."""

    request_log_fn(
        "[profile_sync_inference] "
        f"request={request_index} "
        f"inference_sample={'ignored' if ignored_inference_sample else 'used'} "
        f"returned_chunk_steps={returned_plan_length} "
        f"inference_ms={inference_time_s * 1000.0:.3f} "
        f"step_ms={step_time_s * 1000.0:.3f} "
        f"obs={json.dumps(_frame_request_summary(frame), ensure_ascii=True, sort_keys=True)}"
    )


def profile_sync_inference(
    *,
    observe_fn: FrameSource,
    target_hz: float,
    act_fn: ActionSink | None = None,
    act_src_fn: ActionSource | None = None,
    execute_action: bool = False,
    startup_ignore_inference_samples: int = 1,
    stable_inference_sample_count: int = 4,
    timing_window_size: int = 64,
    timing_trim_ratio: float = 0.1,
    overlap_ratio: float | None = None,
    safety_margin_steps: int = 1,
    output_path: str | PathLike[str] | None = None,
    request_log_fn: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SyncInferenceProfile:
    """Profile sync inference and optionally write one JSON summary."""

    target_hz = validate_positive_number(target_hz, "target_hz")
    target_period_s = 1.0 / target_hz

    if isinstance(execute_action, bool) is False:
        raise InterfaceValidationError("execute_action must be a bool.")
    if request_log_fn is not None and not callable(request_log_fn):
        raise InterfaceValidationError("request_log_fn must be callable when provided.")
    if isinstance(startup_ignore_inference_samples, bool) or not isinstance(
        startup_ignore_inference_samples,
        int,
    ):
        raise InterfaceValidationError(
            "startup_ignore_inference_samples must be an int."
        )
    if startup_ignore_inference_samples < 0:
        raise InterfaceValidationError(
            "startup_ignore_inference_samples must be >= 0, got "
            f"{startup_ignore_inference_samples!r}."
        )
    if isinstance(timing_window_size, bool) or not isinstance(timing_window_size, int):
        raise InterfaceValidationError("timing_window_size must be an int.")
    if timing_window_size <= 0:
        raise InterfaceValidationError(
            f"timing_window_size must be > 0, got {timing_window_size!r}."
        )
    if isinstance(safety_margin_steps, bool) or not isinstance(
        safety_margin_steps,
        int,
    ):
        raise InterfaceValidationError("safety_margin_steps must be an int.")
    if safety_margin_steps < 0:
        raise InterfaceValidationError(
            f"safety_margin_steps must be >= 0, got {safety_margin_steps!r}."
        )

    if overlap_ratio is not None:
        if isinstance(overlap_ratio, bool) or not isinstance(overlap_ratio, (int, float)):
            raise InterfaceValidationError(
                "overlap_ratio must be a real number in [0, 1) when provided."
            )
        overlap_ratio = float(overlap_ratio)
        if not 0.0 <= overlap_ratio < 1.0:
            raise InterfaceValidationError(
                f"overlap_ratio must be in [0, 1), got {overlap_ratio!r}."
            )

    if isinstance(stable_inference_sample_count, bool) or not isinstance(
        stable_inference_sample_count,
        int,
    ):
        raise InterfaceValidationError(
            "stable_inference_sample_count must be an int."
        )
    if stable_inference_sample_count <= 0:
        raise InterfaceValidationError(
            "stable_inference_sample_count must be > 0, got "
            f"{stable_inference_sample_count!r}."
        )

    if act_src_fn is None:
        raise InterfaceValidationError("profile_sync_inference() requires act_src_fn=....")

    total_requests = (
        startup_ignore_inference_samples + stable_inference_sample_count
    )

    step_estimator = _RobustMeanEstimator(
        window_size=timing_window_size,
        startup_ignore_samples=0,
        trim_ratio=timing_trim_ratio,
    )
    inference_estimator = _RobustMeanEstimator(
        window_size=stable_inference_sample_count,
        startup_ignore_samples=startup_ignore_inference_samples,
        trim_ratio=timing_trim_ratio,
    )
    chunk_step_estimator = _RobustMeanEstimator(
        window_size=stable_inference_sample_count,
        startup_ignore_samples=startup_ignore_inference_samples,
        trim_ratio=0.0,
    )
    request_samples: list[_ProfiledRequestSample] = []
    frame_owner = resolve_runtime_owner(observe_fn, act_src_fn)

    for request_index in range(total_requests):
        step_start = float(clock())
        frame = _resolve_step_frame(
            observe_fn,
            None,
            owner=frame_owner,
        )

        inference_start = float(clock())
        action, returned_plan_length = first_action_and_plan_length_from_action_call(
            act_src_fn,
            frame,
        )
        inference_end = float(clock())

        if execute_action:
            _execute_step_action(act_fn, action)
        step_end = float(clock())
        step_time_s = max(step_end - step_start, 0.0)
        inference_time_s = max(inference_end - inference_start, 0.0)
        ignored_inference_sample = request_index < startup_ignore_inference_samples
        observed_latency_steps = _observed_latency_steps(
            inference_time_s=inference_time_s,
            target_period_s=target_period_s,
        )

        step_estimator.add(step_time_s)
        inference_estimator.add(inference_time_s)
        chunk_step_estimator.add(float(returned_plan_length))
        request_samples.append(
            _ProfiledRequestSample(
                request_index=request_index,
                chunk_steps=returned_plan_length,
                inference_time_s=inference_time_s,
                ignored_inference_sample=ignored_inference_sample,
                observed_latency_steps=observed_latency_steps,
            )
        )
        if request_log_fn is not None:
            _emit_profile_request_trace(
                request_log_fn=request_log_fn,
                request_index=request_index,
                ignored_inference_sample=ignored_inference_sample,
                frame=frame,
                returned_plan_length=returned_plan_length,
                inference_time_s=inference_time_s,
                step_time_s=step_time_s,
            )

    estimated_step_time_s = step_estimator.mean
    estimated_inference_time_s = inference_estimator.mean
    estimated_control_hz = None
    estimated_latency_steps = None
    estimated_chunk_steps = chunk_step_estimator.mean
    estimated_max_buffered_hz = None
    estimated_max_overlap_safe_hz = None
    suggested_overlap_steps = None
    suggested_request_trigger_steps = None

    if (
        estimated_inference_time_s is not None
        and estimated_inference_time_s > 0.0
        and estimated_chunk_steps is not None
        and estimated_chunk_steps > 0.0
    ):
        estimated_max_buffered_hz = estimated_chunk_steps / estimated_inference_time_s
        if overlap_ratio is not None:
            overlap_steps_for_capacity = math.floor(
                estimated_chunk_steps * overlap_ratio
            )
            effective_chunk_steps = max(
                estimated_chunk_steps - float(overlap_steps_for_capacity),
                0.0,
            )
            estimated_max_overlap_safe_hz = (
                effective_chunk_steps / estimated_inference_time_s
            )

    if estimated_inference_time_s is not None:
        estimated_latency_steps = max(
            math.ceil(estimated_inference_time_s / target_period_s)
            + safety_margin_steps,
            0,
        )

    if estimated_step_time_s is not None and estimated_step_time_s > 0.0:
        estimated_control_hz = 1.0 / estimated_step_time_s

    if (
        overlap_ratio is not None
        and estimated_chunk_steps is not None
        and estimated_chunk_steps > 0.0
    ):
        suggested_overlap_steps = max(
            math.floor(estimated_chunk_steps * overlap_ratio),
            0,
        )
        suggested_request_trigger_steps = suggested_overlap_steps
        if estimated_latency_steps is not None:
            suggested_request_trigger_steps += estimated_latency_steps

    profile = SyncInferenceProfile(
        target_hz=target_hz,
        target_period_s=target_period_s,
        steps=total_requests,
        execute_action=execute_action,
        startup_ignored_inference_samples=startup_ignore_inference_samples,
        stable_inference_sample_count=stable_inference_sample_count,
        estimated_step_time_s=estimated_step_time_s,
        estimated_inference_time_s=estimated_inference_time_s,
        estimated_control_hz=estimated_control_hz,
        estimated_latency_steps=estimated_latency_steps,
        estimated_chunk_steps=estimated_chunk_steps,
        estimated_max_buffered_hz=estimated_max_buffered_hz,
        estimated_max_overlap_safe_hz=estimated_max_overlap_safe_hz,
        overlap_ratio=overlap_ratio,
        suggested_overlap_steps=suggested_overlap_steps,
        suggested_request_trigger_steps=suggested_request_trigger_steps,
        async_buffer_trace=_build_async_buffer_trace(
            request_samples=request_samples,
            target_hz=target_hz,
            overlap_ratio=overlap_ratio,
        ),
    )
    if output_path is not None:
        profile.write_json(output_path)
    return profile


def recommend_inference_mode(
    *,
    observe_fn: FrameSource,
    act_fn: ActionSink | None = None,
    act_src_fn: ActionSource | None = None,
    target_hz: float,
    execute_action: bool = False,
    startup_ignore_inference_samples: int = 1,
    stable_inference_sample_count: int = 4,
    timing_window_size: int = 64,
    timing_trim_ratio: float = 0.1,
    overlap_ratio: float | None = None,
    safety_margin_steps: int = 1,
    output_path: str | PathLike[str] | None = None,
    request_log_fn: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> InferenceModeRecommendation:
    """Profile sync inference and recommend one runtime mode for a target rate."""

    target_hz = validate_positive_number(target_hz, "target_hz")
    profile = profile_sync_inference(
        observe_fn=observe_fn,
        target_hz=target_hz,
        act_fn=act_fn,
        act_src_fn=act_src_fn,
        execute_action=execute_action,
        startup_ignore_inference_samples=startup_ignore_inference_samples,
        stable_inference_sample_count=stable_inference_sample_count,
        timing_window_size=timing_window_size,
        timing_trim_ratio=timing_trim_ratio,
        overlap_ratio=overlap_ratio,
        safety_margin_steps=safety_margin_steps,
        output_path=None,
        request_log_fn=request_log_fn,
        clock=clock,
    )
    async_supported = (
        profile.estimated_chunk_steps is not None
        and profile.estimated_chunk_steps > 1.0
    )

    estimated_step_time_s = profile.estimated_step_time_s
    target_period_s = profile.target_period_s
    sync_expected_to_meet_target = None
    if estimated_step_time_s is not None and estimated_step_time_s > 0.0:
        sync_expected_to_meet_target = estimated_step_time_s <= target_period_s

    if sync_expected_to_meet_target is True:
        recommended_mode = InferenceMode.SYNC
        reason = (
            f"Estimated sync step time {estimated_step_time_s:.6f}s fits within "
            f"the requested {target_period_s:.6f}s control period "
            f"({target_hz:.3f} Hz)."
        )
    elif sync_expected_to_meet_target is False and async_supported:
        recommended_mode = InferenceMode.ASYNC
        reason = (
            f"Estimated sync step time {estimated_step_time_s:.6f}s exceeds the "
            f"requested {target_period_s:.6f}s control period "
            f"({target_hz:.3f} Hz), and act_src_fn=... is "
            "available for async scheduling."
        )
        if profile.estimated_latency_steps is not None:
            reason += (
                f" Estimated inference latency is about "
                f"{profile.estimated_latency_steps} control steps."
            )
    elif sync_expected_to_meet_target is False:
        recommended_mode = InferenceMode.SYNC
        reason = (
            f"Estimated sync step time {estimated_step_time_s:.6f}s exceeds the "
            f"requested {target_period_s:.6f}s control period "
            f"({target_hz:.3f} Hz), but the profiled source does not return "
            "a chunk with more than one action, so async scheduling is unavailable."
        )
    elif async_supported:
        recommended_mode = InferenceMode.ASYNC
        reason = (
            "No stable sync step-time estimate was produced, but act_src_fn=... "
            "is available, so async scheduling is the safer "
            "choice when you need a target-hz recommendation."
        )
    else:
        recommended_mode = InferenceMode.SYNC
        reason = (
            "No stable sync step-time estimate was produced, and the profiled "
            "source did not demonstrate chunk output with more than one action."
        )

    recommendation = InferenceModeRecommendation(
        target_hz=target_hz,
        target_period_s=target_period_s,
        async_supported=async_supported,
        sync_expected_to_meet_target=sync_expected_to_meet_target,
        recommended_mode=recommended_mode,
        reason=reason,
        profile=profile,
    )
    if output_path is not None:
        recommendation.write_json(output_path)
    return recommendation


__all__ = [
    "InferenceModeRecommendation",
    "SyncInferenceProfile",
    "profile_sync_inference",
    "recommend_inference_mode",
]
