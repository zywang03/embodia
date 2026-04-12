"""Chunk scheduling helpers for inference runtime.

The scheduling rule follows a step-based async controller:

1. Keep one executable action buffer.
2. Keep at most one in-flight chunk request.
3. Estimate request latency directly in emitted control steps.
4. Trigger the next request once:

   ``len(buffer) <= ceil(H_hat) + floor(overlap_ratio * chunk_size)``

5. The background request uses the launch-time buffer snapshot to prepare one
   candidate future chunk.
6. When that reply is accepted, the scheduler only needs to drop the stale
   prefix that became unusable while the request was in flight.

`global_step` is owned entirely by this scheduler. It advances only when one
action is emitted from the buffer. Wall-clock pacing belongs to
``RealtimeController`` and stays outside this module.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import time

import numpy as np

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Command, Frame
from ...shared.common import as_action, as_frame
from ..checks import validate_action, validate_frame
from .protocols import (
    ActionPlan,
    ActionSource,
    ActionSourceProtocol,
    ChunkRequest,
)


@dataclass(slots=True)
class _CompletedChunk:
    """One finished chunk request prepared against its launch buffer."""

    request: ChunkRequest
    prepared_actions: list[Action]
    source_plan_length: int


@dataclass(slots=True)
class _RequestJob:
    """One request plus the immutable buffer snapshot used to prepare it."""

    request: ChunkRequest
    launch_buffer: list[Action]


@dataclass(slots=True)
class ChunkScheduler:
    """Step-based async chunk scheduler.

    ``action_source(frame, request)`` may return either:

    - one ``Action`` (treated as chunk size ``1``), or
    - a sequence of ``Action`` objects.

    The scheduler keeps one executable buffer of future actions. In async mode,
    it requests the next chunk before the buffer is exhausted. In sync mode, it
    uses the same overlap rule but performs the refresh inline.

    ``request.history_actions`` exposes the tail overlap from the current
    buffer. Sources may use it only as conditioning context, or they may return
    a plan that repeats that overlap prefix. The scheduler aligns both forms.
    """

    action_source: ActionSourceProtocol | ActionSource | None = None
    overlap_ratio: float = 0.2
    latency_ema_beta: float = 0.5
    initial_latency_steps: float = 0.0
    max_chunk_size: int | None = None
    use_overlap_blend: bool = False
    overlap_current_weight: float = 0.5
    clock: Callable[[], float] = time.perf_counter
    _buffer: deque[Action] = field(default_factory=deque, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)
    _reference_chunk_size: int = field(default=0, init=False, repr=False)
    _active_source_plan_length: int = field(default=0, init=False, repr=False)
    _latency_steps_estimate: float = field(default=0.0, init=False, repr=False)
    _pending_future: Future[_CompletedChunk] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and initialize runtime state."""

        if isinstance(self.overlap_ratio, bool) or not isinstance(
            self.overlap_ratio,
            (int, float),
        ):
            raise InterfaceValidationError(
                "overlap_ratio must be a real number in the range [0, 1)."
            )
        self.overlap_ratio = float(self.overlap_ratio)
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise InterfaceValidationError(
                "overlap_ratio must be in the range [0, 1), got "
                f"{self.overlap_ratio!r}."
            )

        if isinstance(self.latency_ema_beta, bool) or not isinstance(
            self.latency_ema_beta,
            (int, float),
        ):
            raise InterfaceValidationError(
                "latency_ema_beta must be a real number in the range (0, 1]."
            )
        self.latency_ema_beta = float(self.latency_ema_beta)
        if not 0.0 < self.latency_ema_beta <= 1.0:
            raise InterfaceValidationError(
                "latency_ema_beta must be in the range (0, 1], got "
                f"{self.latency_ema_beta!r}."
            )

        if isinstance(self.initial_latency_steps, bool) or not isinstance(
            self.initial_latency_steps,
            (int, float),
        ):
            raise InterfaceValidationError(
                "initial_latency_steps must be a real number >= 0."
            )
        self.initial_latency_steps = float(self.initial_latency_steps)
        if self.initial_latency_steps < 0.0:
            raise InterfaceValidationError(
                "initial_latency_steps must be >= 0, got "
                f"{self.initial_latency_steps!r}."
            )

        if self.max_chunk_size is not None:
            if isinstance(self.max_chunk_size, bool) or not isinstance(
                self.max_chunk_size,
                int,
            ):
                raise InterfaceValidationError(
                    "max_chunk_size must be an int when provided."
                )
            if self.max_chunk_size <= 0:
                raise InterfaceValidationError(
                    "max_chunk_size must be > 0 when provided, got "
                    f"{self.max_chunk_size!r}."
                )

        if isinstance(self.overlap_current_weight, bool) or not isinstance(
            self.overlap_current_weight,
            (int, float),
        ):
            raise InterfaceValidationError(
                "overlap_current_weight must be a real number in [0, 1]."
            )
        self.overlap_current_weight = float(self.overlap_current_weight)
        if not 0.0 <= self.overlap_current_weight <= 1.0:
            raise InterfaceValidationError(
                "overlap_current_weight must be in [0, 1], got "
                f"{self.overlap_current_weight!r}."
            )

        self._latency_steps_estimate = self.initial_latency_steps

    def reset(self) -> None:
        """Discard buffered and in-flight chunks but keep learned latency."""

        self._buffer.clear()
        self._global_step = 0
        self._reference_chunk_size = 0
        self._active_source_plan_length = 0
        if self._pending_future is not None:
            self._pending_future.cancel()
        self._pending_future = None

    def close(self) -> None:
        """Shut down background request execution."""

        self.reset()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    @property
    def active_source_plan_length(self) -> int:
        """Return the source chunk length most recently accepted."""

        return self._active_source_plan_length

    def estimated_latency_steps(self) -> int:
        """Return the current step-latency estimate used for triggering."""

        return max(int(math.ceil(self._latency_steps_estimate)), 0)

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
        """Return the request threshold for one chunk length."""

        overlap_steps = self.overlap_steps_for_chunk(chunk_length)
        if not include_latency:
            return overlap_steps
        return self.estimated_latency_steps() + overlap_steps

    def _clone_action(self, action: Action) -> Action:
        """Return a detached copy of one standardized action."""

        return Action(
            commands={
                target: Command(
                    command=command.command,
                    value=command.value.copy(),
                    ref_frame=command.ref_frame,
                    meta=dict(command.meta),
                )
                for target, command in action.commands.items()
            },
            meta=dict(action.meta),
        )

    def _clone_actions(self, actions: Sequence[Action]) -> list[Action]:
        """Return detached copies for one sequence of standardized actions."""

        return [self._clone_action(action) for action in actions]

    def _commands_share_layout(self, left: Action, right: Action) -> bool:
        """Return whether two actions have the same command structure."""

        if left.commands.keys() != right.commands.keys():
            return False
        for target, left_command in left.commands.items():
            right_command = right.commands[target]
            if left_command.command != right_command.command:
                return False
            if left_command.ref_frame != right_command.ref_frame:
                return False
            if left_command.meta != right_command.meta:
                return False
            if left_command.value.shape != right_command.value.shape:
                return False
        return True

    def _actions_match(self, left: Action, right: Action) -> bool:
        """Return whether two actions are structurally identical."""

        if left.meta != right.meta:
            return False
        if not self._commands_share_layout(left, right):
            return False
        for target, left_command in left.commands.items():
            right_command = right.commands[target]
            if not np.array_equal(left_command.value, right_command.value):
                return False
        return True

    def _blend_overlap_action(
        self,
        old_action: Action,
        new_action: Action,
    ) -> Action:
        """Blend one aligned overlap step from old/new chunks."""

        if not self._commands_share_layout(old_action, new_action):
            return self._clone_action(new_action)

        new_weight = self.overlap_current_weight
        old_weight = 1.0 - new_weight

        blended_commands: dict[str, Command] = {}
        for target, new_command in new_action.commands.items():
            old_command = old_action.commands[target]
            blended_commands[target] = Command(
                command=new_command.command,
                value=old_command.value * old_weight
                + new_command.value * new_weight,
                ref_frame=new_command.ref_frame,
                meta=dict(new_command.meta),
            )
        return Action(
            commands=blended_commands,
            meta=dict(new_action.meta),
        )

    def _normalize_plan(self, plan: ActionPlan) -> list[Action]:
        """Coerce and validate one returned chunk."""

        if isinstance(plan, Action):
            normalized = [as_action(plan)]
        elif isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
            raise InterfaceValidationError(
                "act_src_fn(frame, request) must return an Action or a sequence "
                f"of Action, got {type(plan).__name__}."
            )
        else:
            normalized = [as_action(item) for item in plan]

        if self.max_chunk_size is not None:
            normalized = normalized[: self.max_chunk_size]
        if not normalized:
            raise InterfaceValidationError(
                "ChunkScheduler received an empty action chunk."
            )

        for action in normalized:
            validate_action(action)
        return self._clone_actions(normalized)

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
        return _RequestJob(
            request=ChunkRequest(
                request_step=request_step,
                request_time_s=float(self.clock()),
                history_start=history_start,
                history_end=len(buffer_list),
                active_chunk_length=len(buffer_list),
                remaining_steps=len(buffer_list),
                overlap_steps=overlap_steps,
                latency_steps=self.estimated_latency_steps() if include_latency else 0,
                request_trigger_steps=self.request_trigger_steps(
                    reference_chunk_size,
                    include_latency=include_latency,
                ),
                plan_start_step=request_step + history_start,
                history_actions=history_actions,
            ),
            launch_buffer=buffer_list,
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
        """Update ``H_hat`` using the latest observed request duration."""

        self._latency_steps_estimate = (
            (1.0 - self.latency_ema_beta) * self._latency_steps_estimate
            + self.latency_ema_beta * float(waited_steps)
        )

    def _integrate_completed_chunk(self, completed: _CompletedChunk) -> bool:
        """Commit one prepared reply by dropping the stale executed prefix."""

        integration_step = self._global_step
        stale_steps = max(integration_step - completed.request.request_step, 0)
        self._update_latency_estimate(stale_steps)

        if stale_steps >= len(completed.prepared_actions):
            return False

        self._buffer = deque(completed.prepared_actions[stale_steps:])
        self._reference_chunk_size = completed.source_plan_length
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

    def _pop_next_action(self) -> Action:
        """Pop one action from the buffer and advance the control step."""

        if not self._buffer:
            raise InterfaceValidationError(
                "ChunkScheduler has no buffered action to emit."
            )
        action = self._buffer.popleft()
        self._global_step += 1
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

        # First consume any reply that is already ready.
        plan_refreshed = self._accept_pending_chunk(block=False)

        # If nothing is executable, wait for the pending reply or run one
        # inline request immediately.
        if not self._buffer:
            if prefetch_async and self._pending_future is not None:
                plan_refreshed = (
                    self._accept_pending_chunk(block=True) or plan_refreshed
                )

            if not self._buffer:
                completed = self._execute_request(
                    normalized_frame,
                    self._build_request_job(include_latency=prefetch_async),
                )
                if not self._integrate_completed_chunk(completed):
                    raise InterfaceValidationError(
                        "ChunkScheduler could not produce a usable action chunk."
                    )
                plan_refreshed = True

        # Once we have executable actions, decide whether it is time to request
        # the next chunk for overlap refresh.
        if self._pending_future is None and self._buffer:
            threshold = self.request_trigger_steps(
                self._reference_chunk_size_for_request(),
                include_latency=prefetch_async,
            )
            if len(self._buffer) <= threshold:
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


__all__ = ["ChunkScheduler"]
