"""Runtime integration and main-loop helpers for chunk scheduling."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from itertools import islice

from ....core.errors import InterfaceValidationError
from ....core.schema import Action, Frame
from ....shared.coerce import as_frame_fast
from ...checks import validate_frame
from .state import _CompletedChunk


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

    next_buffer = deque(islice(completed.prepared_actions, stale_steps, None))

    self._active_chunk_snapshot = completed.prepared_actions
    self._active_chunk_consumed_steps = stale_steps
    self._active_chunk_waited_raw_steps = 0
    self._buffer = next_buffer
    self._execution_buffer.clear()
    self._active_source_plan_length = completed.source_plan_length
    self._maybe_complete_startup_validation()
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
    startup_rtc_seed_chunk: Sequence[Action] | None = None
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
            startup_rtc_seed_chunk = completed.prepared_actions
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
    validate_frame_input: bool = True,
) -> tuple[Action, bool]:
    """Return the next action and whether the buffer was refreshed."""

    normalized_frame = as_frame_fast(frame)
    if validate_frame_input and self.runtime_validation_enabled():
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
