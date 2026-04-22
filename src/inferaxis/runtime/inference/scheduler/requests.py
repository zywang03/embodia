"""Request construction, trigger accounting, and request execution helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence

from ....core.errors import InterfaceValidationError
from ....core.schema import Action, Frame
from ..contracts import ChunkRequest
from .state import _CompletedChunk, _RequestJob


def _steps_before_request_satisfied(self) -> bool:
    """Return whether the current active chunk may launch its next request."""

    if isinstance(self.steps_before_request, bool) or not isinstance(
        self.steps_before_request,
        int,
    ):
        raise InterfaceValidationError(
            "steps_before_request must be an int >= 0."
        )
    if self.steps_before_request < 0:
        raise InterfaceValidationError(
            "steps_before_request must be >= 0, got "
            f"{self.steps_before_request!r}."
        )
    return self._active_chunk_waited_raw_steps >= self.steps_before_request


def _build_request_job(
    self,
    *,
    include_latency: bool,
    rtc_seed_chunk: Sequence[Action] | None = None,
) -> _RequestJob:
    """Build one request together with its launch-time buffer snapshot."""

    request_step = self._global_step
    buffer_actions = self._buffer
    buffer_length = len(buffer_actions)
    launch_buffer = list(buffer_actions) if self.use_overlap_blend and buffer_length else []
    if include_latency:
        control_latency_steps = self.estimated_latency_steps()
        latency_steps = self._estimated_request_latency_steps(
            control_latency_steps=control_latency_steps,
            buffer_actions=buffer_actions,
            execution_buffer_steps=len(self._execution_buffer),
        )
        if buffer_length:
            self._check_execution_window_delay(
                raw_delay_steps=latency_steps,
            )
    else:
        latency_steps = 0
    return _RequestJob(
        request=ChunkRequest(
            request_step=request_step,
            request_time_s=float(self.clock()),
            active_chunk_length=buffer_length,
            remaining_steps=buffer_length,
            latency_steps=latency_steps,
            rtc_args=self._build_rtc_args(
                remaining_chunk=buffer_actions,
                inference_delay=latency_steps,
                rtc_seed_chunk=rtc_seed_chunk,
            ),
        ),
        launch_buffer=launch_buffer,
        launch_control_step=self._control_step,
    )


def _execute_request(
    self,
    frame: Frame,
    job: _RequestJob,
) -> _CompletedChunk:
    """Execute one request and prepare its candidate future chunk."""

    if self.action_source is None:
        raise InterfaceValidationError("ChunkScheduler needs action_source=....")

    request = job.request
    try:
        raw_plan = self.action_source(frame, request)
    except Exception as exc:
        raise InterfaceValidationError(
            f"act_src_fn(frame, request) raised {type(exc).__name__}: {exc}"
        ) from exc

    plan = self._normalize_plan(raw_plan)
    self._validate_chunk_length(len(plan))
    if self.use_overlap_blend and job.launch_buffer:
        overlap_count = min(len(job.launch_buffer), len(plan))
        fused = [
            self._blend_overlap_action(
                job.launch_buffer[index],
                plan[index],
                overlap_index=index,
                overlap_count=overlap_count,
            )
            for index in range(overlap_count)
        ]
        prepared_actions = fused + list(plan[overlap_count:])
    else:
        prepared_actions = list(plan)

    return _CompletedChunk(
        request=request,
        prepared_actions=prepared_actions,
        source_plan_length=len(plan),
        launch_control_step=job.launch_control_step,
    )


def _ensure_executor(self) -> ThreadPoolExecutor:
    """Create the background executor lazily."""

    if self._executor is None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="inferaxis-async-inference",
        )
    return self._executor
