"""Bootstrap, integration, and main-loop helpers for chunk scheduling."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import warnings

from ....core.errors import InterfaceValidationError
from ....core.schema import Action, Frame
from ....shared.common import as_frame
from ...checks import validate_frame
from .shared import (
    _BRIDGE_CONTEXT_WINDOW,
    _CompletedChunk,
    _SLOW_RTC_WARMUP_THRESHOLD_S,
)


class ChunkSchedulerRuntimeOpsMixin:
    """Helpers that manage bootstrap, chunk integration, and emission."""

    __slots__ = ()

    def _should_capture_rtc_seed_chunk(self, completed: _CompletedChunk) -> bool:
        """Return whether one completed request should seed RTC context."""

        if not self.enable_rtc:
            return False
        if completed.request.rtc_args is not None:
            return False
        if self._active_chunk_snapshot or self._rtc_seed_chunk or self._buffer:
            return False
        return True

    def _set_rtc_seed_chunk(self, actions: Sequence[Action]) -> None:
        """Store one full chunk snapshot for the next RTC request."""

        self._rtc_seed_chunk = self._clone_actions(actions)

    def _refresh_bootstrap_rtc_seed_chunk(self, completed: _CompletedChunk) -> None:
        """Keep RTC bootstrap requests chained through the latest full chunk."""

        if not self.enable_rtc:
            return
        if self._active_chunk_snapshot or self._buffer:
            return
        self._set_rtc_seed_chunk(completed.prepared_actions)

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
            self._rtc_seed_chunk.clear()
            raise InterfaceValidationError(
                "RTC startup warmup needs confirmation after a slow "
                "prev_action_chunk request, but stdin is not interactive."
            ) from exc

        if response.strip().lower() not in {"y", "yes"}:
            self._rtc_seed_chunk.clear()
            raise InterfaceValidationError(
                "RTC startup warmup aborted after a slow prev_action_chunk request."
            )

    def _capture_rtc_warmup_chunk(self, completed: _CompletedChunk) -> bool:
        """Capture the first non-RTC reply as RTC seed without executing it."""

        if not self._should_capture_rtc_seed_chunk(completed):
            return False

        waited_control_steps = max(
            self._control_step - completed.launch_control_step,
            0,
        )
        stale_steps = max(self._global_step - completed.request.request_step, 0)
        self._update_latency_estimate(waited_control_steps)
        if stale_steps >= len(completed.prepared_actions):
            return False

        self._set_rtc_seed_chunk(completed.prepared_actions)
        return True

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
        for request_index in range(total_requests):
            job = self._build_request_job(include_latency=False)
            request_start = float(self.clock())
            completed = self._execute_request(frame, job)
            inference_time_s = max(float(self.clock()) - request_start, 0.0)
            self._refresh_bootstrap_rtc_seed_chunk(completed)
            if completed.request.rtc_args is not None:
                last_rtc_bootstrap_duration_s = inference_time_s
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
        transition_bridge = deque()
        old_context = list(self._recent_executed_raw_actions)
        if self.enable_mismatch_bridge and old_context and next_buffer:
            transition_bridge = self._build_transition_bridge(
                old_context=old_context,
                new_context=list(next_buffer)[:_BRIDGE_CONTEXT_WINDOW],
            )

        self._active_chunk_snapshot = self._clone_actions(completed.prepared_actions)
        self._active_chunk_consumed_steps = stale_steps
        self._buffer = next_buffer
        self._execution_buffer.clear()
        self._transition_bridge_buffer = transition_bridge
        self._reference_chunk_size = completed.source_plan_length
        self._active_source_plan_length = completed.source_plan_length
        self._rtc_seed_chunk.clear()
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

    def _pop_next_action(self) -> Action:
        """Pop one action from the buffer and advance the control step."""

        if self._transition_bridge_buffer:
            action = self._transition_bridge_buffer.popleft()
            self._control_step += 1
            return action

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

        plan_refreshed = False
        if not self._transition_bridge_buffer and not self._execution_buffer:
            plan_refreshed = self._accept_pending_chunk(block=False)

        self._ensure_execution_buffer()

        if not self._transition_bridge_buffer and not self._execution_buffer:
            if prefetch_async and self._pending_future is not None:
                plan_refreshed = (
                    self._accept_pending_chunk(block=True) or plan_refreshed
                )
                self._ensure_execution_buffer()

            if not self._transition_bridge_buffer and not self._execution_buffer:
                while not self._transition_bridge_buffer and not self._execution_buffer:
                    completed = self._execute_request(
                        normalized_frame,
                        self._build_request_job(include_latency=prefetch_async),
                    )
                    if self._capture_rtc_warmup_chunk(completed):
                        continue
                    if not self._integrate_completed_chunk(completed):
                        raise InterfaceValidationError(
                            "ChunkScheduler could not produce a usable action chunk."
                        )
                    plan_refreshed = True
                    self._ensure_execution_buffer()

        if self._pending_future is None and self._buffer:
            threshold = self.request_trigger_steps(
                self._reference_chunk_size_for_request(),
                include_latency=prefetch_async,
            )
            if self._remaining_control_steps() <= threshold:
                job = self._build_request_job(include_latency=prefetch_async)
                if prefetch_async:
                    self._pending_future = self._ensure_executor().submit(
                        self._execute_request,
                        normalized_frame,
                        job,
                    )
                else:
                    completed = self._execute_request(normalized_frame, job)
                    plan_refreshed = (
                        self._integrate_completed_chunk(completed) or plan_refreshed
                    )

        return self._pop_next_action(), plan_refreshed
