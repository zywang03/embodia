"""Request construction, overlap accounting, and RTC helpers."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import math

from ....core.errors import InterfaceValidationError
from ....core.schema import Action, Frame
from ..protocols import ChunkRequest, RtcArgs
from .shared import _CompletedChunk, _RequestJob


class ChunkSchedulerRequestOpsMixin:
    """Helpers that build requests and track control-step latency."""

    __slots__ = ()

    def overlap_steps_for_chunk(self, chunk_length: int) -> int:
        """Return ``floor(overlap_ratio * chunk_length)``."""

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
        return min(int(math.floor(self.overlap_ratio * chunk_length)), chunk_length)

    def request_trigger_steps(
        self,
        chunk_length: int,
        *,
        include_latency: bool = True,
    ) -> int:
        """Return the control-step trigger threshold for one raw chunk length."""

        overlap_steps = self.overlap_steps_for_chunk(chunk_length)
        overlap_control_steps = self._control_steps_for_raw_count(overlap_steps)
        if not include_latency:
            return overlap_control_steps
        return self.estimated_latency_steps() + overlap_control_steps

    def _control_steps_for_raw_count(self, raw_steps: int) -> int:
        """Return control-step count for ``raw_steps`` raw actions."""

        if isinstance(raw_steps, bool) or not isinstance(raw_steps, int):
            raise InterfaceValidationError(
                f"raw_steps must be an int, got {type(raw_steps).__name__}."
            )
        if raw_steps < 0:
            raise InterfaceValidationError(
                f"raw_steps must be >= 0, got {raw_steps!r}."
            )
        if raw_steps == 0:
            return 0
        return raw_steps + max(raw_steps - 1, 0) * self.interpolation_steps

    def _control_steps_for_actions(self, actions: Sequence[Action]) -> int:
        """Return control-step count for one raw-action sequence."""

        return self._control_steps_for_raw_count(len(actions))

    def _raw_segment_control_steps(self, *, has_successor: bool) -> int:
        """Return control steps emitted for one raw action segment."""

        if has_successor:
            return 1 + self.interpolation_steps
        return 1

    def _remaining_control_steps(self) -> int:
        """Return remaining executable control steps from the current state."""

        total_steps = len(self._transition_bridge_buffer)
        if self._execution_buffer:
            total_steps += len(self._execution_buffer)
            total_steps += self._control_steps_for_raw_count(max(len(self._buffer) - 1, 0))
            return total_steps
        total_steps += self._control_steps_for_actions(self._buffer)
        return total_steps

    def _reference_chunk_size_for_request(self) -> int:
        """Return the chunk size used for overlap and trigger calculations."""

        if self._reference_chunk_size > 0:
            return self._reference_chunk_size
        return len(self._buffer)

    def _build_request_job(self, *, include_latency: bool) -> _RequestJob:
        """Build one request together with its launch-time buffer snapshot."""

        request_step = self._global_step
        buffer_list = self._clone_actions(self._buffer)
        reference_chunk_size = self._reference_chunk_size_for_request()
        overlap_steps = self.overlap_steps_for_chunk(reference_chunk_size)
        history_count = min(overlap_steps, len(buffer_list))
        history_start = len(buffer_list) - history_count
        history_actions = self._clone_actions(buffer_list[history_start:])
        latency_steps = self.estimated_latency_steps() if include_latency else 0
        rtc_inference_delay = self._estimated_rtc_inference_delay(
            control_latency_steps=latency_steps,
        )
        return _RequestJob(
            request=ChunkRequest(
                request_step=request_step,
                request_time_s=float(self.clock()),
                history_start=history_start,
                history_end=len(buffer_list),
                active_chunk_length=len(buffer_list),
                remaining_steps=len(buffer_list),
                overlap_steps=overlap_steps,
                latency_steps=latency_steps,
                request_trigger_steps=self.request_trigger_steps(
                    reference_chunk_size,
                    include_latency=include_latency,
                ),
                plan_start_step=request_step + history_start,
                history_actions=history_actions,
                rtc_args=self._build_rtc_args(
                    remaining_chunk=buffer_list,
                    inference_delay=rtc_inference_delay,
                ),
            ),
            launch_buffer=buffer_list,
            launch_control_step=self._control_step,
        )

    def _build_rtc_args(
        self,
        *,
        remaining_chunk: Sequence[Action],
        inference_delay: int,
    ) -> RtcArgs | None:
        """Build optional RTC hints for one policy request."""

        del remaining_chunk

        if not self.enable_rtc:
            return None

        if self._active_chunk_snapshot:
            cloned_chunk = self._clone_actions(self._active_chunk_snapshot)
            consumed_steps = min(
                self._active_chunk_consumed_steps,
                len(cloned_chunk),
            )
        elif self._rtc_seed_chunk:
            cloned_chunk = self._clone_actions(self._rtc_seed_chunk)
            consumed_steps = 0
        else:
            return None

        execute_horizon = max(len(cloned_chunk) - consumed_steps, 0)
        return RtcArgs(
            prev_action_chunk=cloned_chunk,
            inference_delay=min(
                max(int(inference_delay), 1),
                execute_horizon,
            ),
            execute_horizon=execute_horizon,
        )

    def _estimated_rtc_inference_delay(
        self,
        *,
        control_latency_steps: int,
    ) -> int:
        """Project control-step latency back into raw-step RTC semantics."""

        if self._active_chunk_snapshot:
            execute_horizon = max(
                len(self._active_chunk_snapshot) - self._active_chunk_consumed_steps,
                0,
            )
        elif self._rtc_seed_chunk:
            execute_horizon = len(self._rtc_seed_chunk)
        else:
            return max(int(control_latency_steps), 0)

        if execute_horizon <= 0:
            return 0
        if not self._active_chunk_snapshot:
            return min(max(int(control_latency_steps), 1), execute_horizon)

        remaining_control_steps = max(int(control_latency_steps), 0)
        if self._transition_bridge_buffer:
            bridge_steps = len(self._transition_bridge_buffer)
            if remaining_control_steps <= bridge_steps:
                return 0
            remaining_control_steps -= bridge_steps

        if not self._buffer:
            return 0

        for raw_offset in range(len(self._buffer)):
            if raw_offset == 0 and self._execution_buffer:
                segment_steps = len(self._execution_buffer)
            else:
                segment_steps = self._raw_segment_control_steps(
                    has_successor=raw_offset < (len(self._buffer) - 1),
                )
            if remaining_control_steps <= segment_steps:
                return raw_offset + 1
            remaining_control_steps -= segment_steps
        return len(self._buffer)

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
        returned_plan_start_step = request.request_step
        history_actions = request.history_actions
        if (
            history_actions
            and len(plan) >= len(history_actions)
            and all(
                self._actions_match(plan[index], history_actions[index])
                for index in range(len(history_actions))
            )
        ):
            returned_plan_start_step = request.plan_start_step

        prefix_keep_steps = max(returned_plan_start_step - request.request_step, 0)
        if prefix_keep_steps > len(job.launch_buffer):
            raise InterfaceValidationError(
                "ChunkScheduler received a chunk that begins after the launch "
                "buffer ends."
            )

        prefix = self._clone_actions(job.launch_buffer[:prefix_keep_steps])
        remaining = job.launch_buffer[prefix_keep_steps:]
        if self.use_overlap_blend and remaining:
            overlap_count = min(len(remaining), len(plan))
            fused = [
                self._blend_overlap_action(
                    remaining[index],
                    plan[index],
                    overlap_index=index,
                    overlap_count=overlap_count,
                )
                for index in range(overlap_count)
            ]
            prepared_actions = prefix + fused + plan[overlap_count:]
        else:
            prepared_actions = prefix + self._clone_actions(plan)

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

    def _update_latency_estimate(self, waited_steps: int) -> None:
        """Update ``H_hat`` using the latest observed control-step delay."""

        self._latency_observation_count += 1
        if self._latency_observation_count <= self.warmup_requests:
            return
        if (
            self._latency_observation_count == self.warmup_requests + 1
            and self._latency_steps_estimate <= 0.0
        ):
            self._latency_steps_estimate = float(waited_steps)
            return
        self._latency_steps_estimate = (
            (1.0 - self.latency_ema_beta) * self._latency_steps_estimate
            + self.latency_ema_beta * float(waited_steps)
        )

    def _observed_latency_steps_from_duration(self, inference_time_s: float) -> int:
        """Convert one measured request duration into control-step latency."""

        if self.control_period_s is None:
            return 1
        if inference_time_s <= 0.0:
            return 1
        return max(int(math.ceil(inference_time_s / self.control_period_s)), 1)
