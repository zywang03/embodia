"""Async buffer trace simulation helpers for sync profiling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .models import (
    _ProfiledRequestSample,
    AsyncBufferTrace,
    AsyncBufferTraceRequest,
    AsyncBufferTraceStep,
)


@dataclass(slots=True)
class _PendingAsyncTraceRequest:
    """One in-flight simulated async request."""

    sample: _ProfiledRequestSample
    start_step: int
    start_executed_step: int
    reply_step: int


@dataclass(slots=True)
class _TraceActionSlot:
    """One simulated buffered action annotated by its request of origin."""

    request_index: int
    blended_from_request_index: int | None = None


def _observed_latency_steps(
    *,
    inference_time_s: float,
    target_period_s: float,
) -> int:
    """Convert one measured inference duration into control-step latency."""

    if inference_time_s <= 0.0:
        return 1
    return max(int(math.ceil(inference_time_s / target_period_s)), 1)


def _build_async_buffer_trace(
    *,
    request_samples: list[_ProfiledRequestSample],
    target_hz: float,
    steps_before_request: int = 0,
    latency_ema_beta: float = 0.5,
    initial_latency_steps: float = 0.0,
) -> AsyncBufferTrace:
    """Simulate async buffer growth and depletion from sync-profiled requests."""

    target_period_s = 1.0 / target_hz
    if not request_samples:
        return AsyncBufferTrace(
            target_hz=target_hz,
            target_period_s=target_period_s,
            steps_before_request=steps_before_request,
            latency_ema_beta=latency_ema_beta,
            initial_latency_steps=initial_latency_steps,
            steps=[],
            requests=[],
        )

    first_sample = request_samples[0]
    buffer_slots: deque[_TraceActionSlot] = deque(
        _TraceActionSlot(request_index=first_sample.request_index)
        for _ in range(first_sample.chunk_steps)
    )
    reference_chunk_steps = first_sample.chunk_steps
    latency_steps_estimate = float(initial_latency_steps)
    request_cursor = 1
    executed_steps = 0
    executed_wait_raw_steps = 0
    pending: _PendingAsyncTraceRequest | None = None
    steps: list[AsyncBufferTraceStep] = []
    requests: list[AsyncBufferTraceRequest] = [
        AsyncBufferTraceRequest(
            request_index=first_sample.request_index,
            start_step=0,
            reply_step=0,
            chunk_steps=first_sample.chunk_steps,
            steps_before_request=steps_before_request,
            observed_latency_steps=first_sample.observed_latency_steps,
            executed_wait_raw_steps=0,
            aligned_chunk_steps=first_sample.chunk_steps,
            blended_steps_after_accept=0,
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
        buffer_before_accept = len(buffer_slots)
        request_started = False
        started_request_index: int | None = None
        request_completed = False
        completed_request_index: int | None = None
        executed_request_index: int | None = None
        blended_from_request_index: int | None = None

        if pending is not None and step_index >= pending.reply_step:
            waited_slot_steps = max(step_index - pending.start_step, 0)
            waited_executed_steps = max(
                executed_steps - pending.start_executed_step,
                0,
            )
            if not pending.sample.ignored_inference_sample:
                latency_steps_estimate = (
                    (1.0 - latency_ema_beta) * latency_steps_estimate
                    + latency_ema_beta * float(waited_executed_steps)
                )
            aligned_chunk_steps = max(
                pending.sample.chunk_steps - waited_executed_steps,
                0,
            )
            surviving_old_slots = list(buffer_slots)
            blended_steps_after_accept = min(
                len(surviving_old_slots),
                aligned_chunk_steps,
            )
            requests.append(
                AsyncBufferTraceRequest(
                    request_index=pending.sample.request_index,
                    start_step=pending.start_step,
                    reply_step=step_index,
                    chunk_steps=pending.sample.chunk_steps,
                    steps_before_request=steps_before_request,
                    observed_latency_steps=waited_slot_steps,
                    executed_wait_raw_steps=waited_executed_steps,
                    aligned_chunk_steps=aligned_chunk_steps,
                    blended_steps_after_accept=blended_steps_after_accept,
                    ignored_inference_sample=pending.sample.ignored_inference_sample,
                )
            )
            if aligned_chunk_steps > 0:
                blended_slots = [
                    _TraceActionSlot(
                        request_index=pending.sample.request_index,
                        blended_from_request_index=surviving_old_slots[index].request_index,
                    )
                    for index in range(blended_steps_after_accept)
                ]
                tail_slots = [
                    _TraceActionSlot(request_index=pending.sample.request_index)
                    for _ in range(aligned_chunk_steps - blended_steps_after_accept)
                ]
                buffer_slots = deque(blended_slots + tail_slots)
                reference_chunk_steps = pending.sample.chunk_steps
                executed_wait_raw_steps = 0
            request_completed = True
            completed_request_index = pending.sample.request_index
            pending = None

        buffer_after_accept = len(buffer_slots)
        latency_steps_ceiled = max(int(math.ceil(latency_steps_estimate)), 0)

        if (
            pending is None
            and request_cursor < len(request_samples)
            and executed_wait_raw_steps >= steps_before_request
        ):
            sample = request_samples[request_cursor]
            pending = _PendingAsyncTraceRequest(
                sample=sample,
                start_step=step_index,
                start_executed_step=executed_steps,
                reply_step=step_index + max(sample.observed_latency_steps, 1),
            )
            request_started = True
            started_request_index = sample.request_index
            request_cursor += 1

        underrun = buffer_after_accept <= 0
        if underrun:
            buffer_after_execute = 0
        else:
            emitted_slot = buffer_slots.popleft()
            executed_request_index = emitted_slot.request_index
            blended_from_request_index = emitted_slot.blended_from_request_index
            buffer_after_execute = len(buffer_slots)
            executed_steps += 1
            executed_wait_raw_steps += 1

        steps.append(
            AsyncBufferTraceStep(
                step_index=step_index,
                buffer_before_accept=buffer_before_accept,
                buffer_after_accept=buffer_after_accept,
                buffer_after_execute=buffer_after_execute,
                steps_before_request=steps_before_request,
                executed_wait_raw_steps=executed_wait_raw_steps,
                latency_steps_estimate=latency_steps_ceiled,
                reference_chunk_steps=reference_chunk_steps,
                request_started=request_started,
                started_request_index=started_request_index,
                request_completed=request_completed,
                completed_request_index=completed_request_index,
                executed_request_index=executed_request_index,
                blended_from_request_index=blended_from_request_index,
                underrun=underrun,
            )
        )

        if request_cursor >= len(request_samples) and pending is None and not buffer_slots:
            break

    return AsyncBufferTrace(
        target_hz=target_hz,
        target_period_s=target_period_s,
        steps_before_request=steps_before_request,
        latency_ema_beta=latency_ema_beta,
        initial_latency_steps=initial_latency_steps,
        steps=steps,
        requests=requests,
    )
