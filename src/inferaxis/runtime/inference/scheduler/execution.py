"""Runtime integration and main-loop helpers for chunk scheduling."""

from __future__ import annotations

from collections.abc import Sequence

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
    accepted_length = max(len(completed.prepared_actions) - stale_steps, 0)
    profiler = self.live_profile

    if stale_steps >= len(completed.prepared_actions):
        if profiler is not None:
            profiler.record_chunk_actions(  # type: ignore[attr-defined]
                request_index=completed.request_index,
                request_step=completed.request.request_step,
                actions=completed.prepared_actions,
                stale_steps=stale_steps,
                accepted_length=0,
            )
            profiler.record_accept(  # type: ignore[attr-defined]
                request_index=completed.request_index,
                accepted_time_s=None,
                waited_control_steps=waited_control_steps,
                stale_raw_steps=stale_steps,
                accepted_chunk_length=0,
                dropped_as_stale=True,
            )
        return False

    accepted_time_s = float(self.clock()) if profiler is not None else None
    if profiler is not None:
        profiler.record_chunk_actions(  # type: ignore[attr-defined]
            request_index=completed.request_index,
            request_step=completed.request.request_step,
            actions=completed.prepared_actions,
            stale_steps=stale_steps,
            accepted_length=accepted_length,
        )
        profiler.record_accept(  # type: ignore[attr-defined]
            request_index=completed.request_index,
            accepted_time_s=accepted_time_s,
            waited_control_steps=waited_control_steps,
            stale_raw_steps=stale_steps,
            accepted_chunk_length=accepted_length,
            dropped_as_stale=False,
        )

    self._raw_buffer.accept_chunk(
        actions=completed.prepared_actions,
        request_step=completed.request.request_step,
        current_raw_step=integration_step,
        source_plan_length=completed.source_plan_length,
    )
    self._execution_cursor.reset()
    self._maybe_complete_startup_validation()
    return True


def _accept_pending_chunk(self, *, block: bool) -> bool:
    """Integrate a finished async request when available."""

    pending = self._pipeline.pending
    if pending is None:
        return False
    if not block and not pending.done():
        return False

    completed = pending.result()
    self._pipeline.clear_pending()
    return self._integrate_completed_chunk(completed)


def _accept_ready_pending_chunk(self) -> bool:
    """Try to accept one already-finished async request."""

    if not self._execution_cursor.at_raw_boundary:
        return False
    return self._accept_pending_chunk(block=False)


def _accept_blocking_pending_chunk(self) -> bool:
    """Block until the in-flight request finishes and integrate it."""

    if self._pipeline.pending is None:
        return False
    return self._accept_pending_chunk(block=True)


def _record_completed_pending_profile_request(self) -> None:
    """Mark a completed but unaccepted pending request before profiler flush."""

    profiler = self.live_profile
    future = self._pipeline.pending
    if profiler is None or future is None or not future.done():
        return
    try:
        completed = future.result()
    except Exception:
        return
    profiler.record_completed_without_accept(  # type: ignore[attr-defined]
        request_index=completed.request_index,
        request_step=completed.request.request_step,
        actions=completed.prepared_actions,
    )


def _request_until_execution_buffer_ready(
    self,
    frame: Frame,
    *,
    include_latency: bool,
) -> bool:
    """Run inline requests until at least one executable action is ready."""

    plan_refreshed = False
    startup_rtc_seed_chunk: Sequence[Action] | None = None
    while not self._raw_buffer.has_actions:
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
            if self.live_profile is not None:
                self.live_profile.record_completed_without_accept(  # type: ignore[attr-defined]
                    request_index=completed.request_index,
                    request_step=completed.request.request_step,
                    actions=completed.prepared_actions,
                )
            startup_rtc_seed_chunk = completed.prepared_actions
            continue
        self._validate_startup_execution_window(completed)
        if not self._integrate_completed_chunk(completed):
            raise InterfaceValidationError(
                "ChunkScheduler could not produce a usable action chunk."
            )
        plan_refreshed = True
    return plan_refreshed


def _ensure_executable_actions(
    self,
    frame: Frame,
    *,
    prefetch_async: bool,
) -> bool:
    """Make sure one executable action is available before emission."""

    if self._raw_buffer.has_actions:
        return False

    plan_refreshed = False
    if prefetch_async and self._pipeline.pending is not None:
        plan_refreshed = self._accept_blocking_pending_chunk()
    if self._raw_buffer.has_actions:
        return plan_refreshed

    return (
        self._request_until_execution_buffer_ready(
            frame,
            include_latency=prefetch_async,
        )
        or plan_refreshed
    )


def _maybe_launch_next_request(
    self,
    frame: Frame,
    *,
    prefetch_async: bool,
    plan_refreshed: bool,
) -> bool:
    """Launch or inline the next refresh request when the trigger allows it."""

    if self._pipeline.pending is not None:
        return plan_refreshed
    if not self._raw_buffer.has_actions or self._active_source_plan_length == 1:
        return plan_refreshed
    if not self._steps_before_request_satisfied():
        return plan_refreshed

    job = self._build_request_job(include_latency=prefetch_async)
    if prefetch_async:
        self._pipeline.pending = self._pipeline.ensure_executor().submit(
            self._execute_request,
            frame,
            job,
        )
        return plan_refreshed

    completed = self._execute_request(frame, job)
    return self._integrate_completed_chunk(completed) or plan_refreshed


def _pop_next_action(self) -> Action:
    """Pop one action from the buffer and advance the control step."""

    if not self._raw_buffer.has_actions:
        raise InterfaceValidationError("ChunkScheduler has no buffered action to emit.")
    self._sync_execution_cursor_config()
    action = self._execution_cursor.next_action()
    self._control_step += 1
    self._sync_execution_cursor_config()
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
