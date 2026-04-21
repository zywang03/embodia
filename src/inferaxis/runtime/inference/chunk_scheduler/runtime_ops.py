"""Bootstrap, integration, and main-loop helpers for chunk scheduling."""

from __future__ import annotations

from collections import deque
import warnings

from ....core.errors import InterfaceValidationError
from ....core.schema import Action, Frame
from ....shared.common import as_frame
from ...checks import validate_frame
from .shared import (
    _CompletedChunk,
    _SLOW_RTC_WARMUP_THRESHOLD_S,
)


class ChunkSchedulerRuntimeOpsMixin:
    """Helpers that manage bootstrap, chunk integration, and emission."""

    __slots__ = ()

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
        if self._buffer or self._execution_buffer:
            return False
        return True

    def _confirm_slow_rtc_bootstrap_request(
        self,
        *,
        inference_time_s: float,
    ) -> None:
        """Warn and require confirmation after a slow RTC bootstrap request."""

        if inference_time_s <= _SLOW_RTC_WARMUP_THRESHOLD_S:
            return

        threshold_ms = _SLOW_RTC_WARMUP_THRESHOLD_S * 1000.0
        duration_ms = inference_time_s * 1000.0
        message = (
            "The last RTC warmup request carrying prev_action_chunk took "
            f"{duration_ms:.1f} ms, exceeding the {threshold_ms:.1f} ms "
            "startup threshold."
        )
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
        rtc_seed_chunk: list[Action] | None = None
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
                rtc_seed_chunk = self._clone_actions(completed.prepared_actions)
            observed_steps = self._observed_latency_steps_from_duration(
                inference_time_s,
            )
            if request_index >= self.warmup_requests:
                profiled_steps.append(observed_steps)
            reusable_completed = completed

        if last_rtc_bootstrap_duration_s is not None:
            self._confirm_slow_rtc_bootstrap_request(
                inference_time_s=last_rtc_bootstrap_duration_s,
            )

        if profiled_steps:
            self._latency_steps_estimate = float(
                sum(profiled_steps) / len(profiled_steps)
            )
        else:
            self._latency_steps_estimate = max(self._latency_steps_estimate, 1.0)
        self._latency_observation_count = self.warmup_requests
        self._startup_latency_bootstrap_complete = True
        if reusable_completed is not None:
            self._validate_startup_execution_window(reusable_completed)

        return reusable_completed

    def bootstrap(self, frame: Frame) -> bool:
        """Run startup warmup/profile requests and seed the first executable chunk."""

        normalized_frame = as_frame(frame)
        validate_frame(normalized_frame)

        if self.control_period_s is None or self.latency_estimate_ready():
            return False
        if self._buffer or self._pending_future is not None:
            return False

        bootstrapped_chunk = self._bootstrap_async_latency(normalized_frame)
        if bootstrapped_chunk is None:
            return False
        if not self._integrate_completed_chunk(bootstrapped_chunk):
            raise InterfaceValidationError(
                "ChunkScheduler could not produce a usable warmup chunk."
            )
        return True

    def _integrate_completed_chunk(self, completed: _CompletedChunk) -> bool:
        """Commit one prepared reply by dropping the stale executed prefix."""

        self._validate_chunk_length(completed.source_plan_length)
        integration_step = self._global_step
        stale_steps = max(integration_step - completed.request.request_step, 0)
        waited_control_steps = max(
            self._control_step - completed.launch_control_step,
            0,
        )
        self._update_latency_estimate(waited_control_steps)

        if stale_steps >= len(completed.prepared_actions):
            return False

        next_buffer = deque(completed.prepared_actions[stale_steps:])

        self._active_chunk_snapshot = self._clone_actions(completed.prepared_actions)
        self._active_chunk_consumed_steps = stale_steps
        self._active_chunk_waited_raw_steps = 0
        self._buffer = next_buffer
        self._execution_buffer.clear()
        self._active_source_plan_length = completed.source_plan_length
        return True

    def _accept_pending_chunk(self, *, block: bool) -> bool:
        """Integrate a finished async request when available."""

        if self._pending_future is None:
            return False
        if not block and not self._pending_future.done():
            return False

        completed = self._pending_future.result()
        self._pending_future = None
        return self._integrate_completed_chunk(completed)

    def _accept_ready_pending_chunk(self) -> bool:
        """Try to accept one already-finished async request."""

        if self._execution_buffer:
            return False
        return self._accept_pending_chunk(block=False)

    def _accept_blocking_pending_chunk(self) -> bool:
        """Block until the in-flight request finishes and integrate it."""

        if self._pending_future is None:
            return False
        refreshed = self._accept_pending_chunk(block=True)
        self._ensure_execution_buffer()
        return refreshed

    def _request_until_execution_buffer_ready(
        self,
        frame: Frame,
        *,
        include_latency: bool,
    ) -> bool:
        """Run inline requests until at least one executable action is ready."""

        plan_refreshed = False
        startup_rtc_seed_chunk: list[Action] | None = None
        while not self._execution_buffer:
            completed = self._execute_request(
                frame,
                self._build_request_job(
                    include_latency=include_latency,
                    rtc_seed_chunk=startup_rtc_seed_chunk,
                ),
            )
            if self._should_retry_with_local_rtc_seed(
                completed=completed,
                rtc_seed_chunk=startup_rtc_seed_chunk,
            ):
                startup_rtc_seed_chunk = self._clone_actions(
                    completed.prepared_actions,
                )
                continue
            self._validate_startup_execution_window(completed)
            if not self._integrate_completed_chunk(completed):
                raise InterfaceValidationError(
                    "ChunkScheduler could not produce a usable action chunk."
                )
            plan_refreshed = True
            self._ensure_execution_buffer()
        return plan_refreshed

    def _ensure_executable_actions(
        self,
        frame: Frame,
        *,
        prefetch_async: bool,
    ) -> bool:
        """Make sure one executable action is available before emission."""

        self._ensure_execution_buffer()
        if self._execution_buffer:
            return False

        plan_refreshed = False
        if prefetch_async and self._pending_future is not None:
            plan_refreshed = self._accept_blocking_pending_chunk()
        if self._execution_buffer:
            return plan_refreshed

        return self._request_until_execution_buffer_ready(
            frame,
            include_latency=prefetch_async,
        ) or plan_refreshed

    def _maybe_launch_next_request(
        self,
        frame: Frame,
        *,
        prefetch_async: bool,
        plan_refreshed: bool,
    ) -> bool:
        """Launch or inline the next refresh request when the trigger allows it."""

        if self._pending_future is not None:
            return plan_refreshed
        if not self._buffer or self._active_source_plan_length == 1:
            return plan_refreshed
        if not self._steps_before_request_satisfied():
            return plan_refreshed

        job = self._build_request_job(include_latency=prefetch_async)
        if prefetch_async:
            self._pending_future = self._ensure_executor().submit(
                self._execute_request,
                frame,
                job,
            )
            return plan_refreshed

        completed = self._execute_request(frame, job)
        return self._integrate_completed_chunk(completed) or plan_refreshed

    def _pop_next_action(self) -> Action:
        """Pop one action from the buffer and advance the control step."""

        self._ensure_execution_buffer()
        if not self._execution_buffer:
            raise InterfaceValidationError(
                "ChunkScheduler has no buffered action to emit."
            )
        action = self._execution_buffer.popleft()
        self._control_step += 1
        if not self._execution_buffer:
            self._advance_raw_step()
        return action

    def next_action(
        self,
        frame: Frame,
        *,
        prefetch_async: bool = True,
    ) -> tuple[Action, bool]:
        """Return the next action and whether the buffer was refreshed."""

        normalized_frame = as_frame(frame)
        validate_frame(normalized_frame)

        plan_refreshed = self._accept_ready_pending_chunk()
        plan_refreshed = (
            self._ensure_executable_actions(
                normalized_frame,
                prefetch_async=prefetch_async,
            )
            or plan_refreshed
        )
        plan_refreshed = self._maybe_launch_next_request(
            normalized_frame,
            prefetch_async=prefetch_async,
            plan_refreshed=plan_refreshed,
        )

        return self._pop_next_action(), plan_refreshed
