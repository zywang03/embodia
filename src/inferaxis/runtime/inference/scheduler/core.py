"""Core ChunkScheduler state container and method wiring."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import time

from ....core.schema import Action
from ..optimizers import BlendWeight
from ..protocols import ActionSource, ActionSourceProtocol
from . import actions, bootstrap, config, execution, latency, requests, rtc
from .state import _CompletedChunk


@dataclass(slots=True)
class ChunkScheduler:
    """Step-based async chunk scheduler."""

    action_source: ActionSourceProtocol | ActionSource | None = None
    steps_before_request: int = 0
    execution_steps: int | None = None
    latency_ema_beta: float = 0.5
    initial_latency_steps: float = 0.0
    fixed_latency_steps: float | None = None
    control_period_s: float | None = None
    warmup_requests: int = 3
    profile_delay_requests: int = 0
    interpolation_steps: int = 0
    max_chunk_size: int | None = None
    use_overlap_blend: bool = False
    overlap_current_weight: BlendWeight = 0.5
    enable_rtc: bool = False
    latency_steps_offset: int = 0
    startup_validation_only: bool = True
    clock: Callable[[], float] = time.perf_counter
    _buffer: deque[Action] = field(default_factory=deque, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)
    _control_step: int = field(default=0, init=False, repr=False)
    _active_chunk_snapshot: list[Action] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _active_chunk_consumed_steps: int = field(default=0, init=False, repr=False)
    _active_chunk_waited_raw_steps: int = field(default=0, init=False, repr=False)
    _active_source_plan_length: int = field(default=0, init=False, repr=False)
    _latency_steps_estimate: float = field(default=0.0, init=False, repr=False)
    _latency_observation_count: int = field(default=0, init=False, repr=False)
    _startup_latency_bootstrap_complete: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _startup_execution_window_validated: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _pending_future: Future[_CompletedChunk] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _execution_buffer: deque[Action] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )
    _rtc_chunk_total_length: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _startup_validation_complete: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate configuration and initialize runtime state."""

        self._validate_configuration()

    def reset(self) -> None:
        """Discard buffered and in-flight chunks but keep learned latency."""

        self._buffer.clear()
        self._global_step = 0
        self._control_step = 0
        self._active_chunk_snapshot = []
        self._active_chunk_consumed_steps = 0
        self._active_chunk_waited_raw_steps = 0
        self._active_source_plan_length = 0
        self._startup_execution_window_validated = False
        self._execution_buffer.clear()
        self._rtc_chunk_total_length = None
        self._startup_validation_complete = False
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

    def runtime_validation_enabled(self) -> bool:
        """Return whether hot-path frame/action validation should run."""

        if not self.startup_validation_only:
            return True
        return not self._startup_validation_complete

    _validate_configuration = config._validate_configuration
    refresh_latency_mode = config.refresh_latency_mode
    _validated_latency_steps_offset = config._validated_latency_steps_offset

    estimated_latency_steps = latency.estimated_latency_steps
    _base_estimated_latency_steps = latency._base_estimated_latency_steps
    latency_estimate_ready = latency.latency_estimate_ready
    _control_steps_for_raw_count = latency._control_steps_for_raw_count
    _control_steps_for_actions = latency._control_steps_for_actions
    _raw_segment_control_steps = latency._raw_segment_control_steps
    _remaining_control_steps = latency._remaining_control_steps
    _project_control_latency_to_raw_steps = latency._project_control_latency_to_raw_steps
    _estimated_request_latency_steps = latency._estimated_request_latency_steps
    _update_latency_estimate = latency._update_latency_estimate
    _observed_latency_steps_from_duration = latency._observed_latency_steps_from_duration

    _materialize_action = actions._materialize_action
    _materialize_command = actions._materialize_command
    _commands_share_layout = actions._commands_share_layout
    _commands_share_target_layout = actions._commands_share_target_layout
    _actions_match = actions._actions_match
    _blend_overlap_action = actions._blend_overlap_action
    _overlap_new_weight = actions._overlap_new_weight
    _interpolate_action = actions._interpolate_action
    _build_execution_segment = actions._build_execution_segment
    _ensure_execution_buffer = actions._ensure_execution_buffer
    _advance_raw_step = actions._advance_raw_step
    _normalize_plan = actions._normalize_plan

    _build_rtc_args = rtc._build_rtc_args
    _build_prev_action_chunk = rtc._build_prev_action_chunk
    _validate_chunk_length = rtc._validate_chunk_length
    _lock_rtc_chunk_total_length = rtc._lock_rtc_chunk_total_length
    _validate_rtc_execution_window_structure = rtc._validate_rtc_execution_window_structure
    _check_execution_window_delay = rtc._check_execution_window_delay

    _steps_before_request_satisfied = requests._steps_before_request_satisfied
    _build_request_job = requests._build_request_job
    _execute_request = requests._execute_request
    _ensure_executor = requests._ensure_executor

    _validate_startup_execution_window = bootstrap._validate_startup_execution_window
    _should_retry_with_local_rtc_seed = bootstrap._should_retry_with_local_rtc_seed
    _confirm_slow_rtc_bootstrap_request = bootstrap._confirm_slow_rtc_bootstrap_request
    _bootstrap_async_latency = bootstrap._bootstrap_async_latency
    _maybe_complete_startup_validation = bootstrap._maybe_complete_startup_validation
    bootstrap = bootstrap.bootstrap

    _integrate_completed_chunk = execution._integrate_completed_chunk
    _accept_pending_chunk = execution._accept_pending_chunk
    _accept_ready_pending_chunk = execution._accept_ready_pending_chunk
    _accept_blocking_pending_chunk = execution._accept_blocking_pending_chunk
    _request_until_execution_buffer_ready = execution._request_until_execution_buffer_ready
    _ensure_executable_actions = execution._ensure_executable_actions
    _maybe_launch_next_request = execution._maybe_launch_next_request
    _pop_next_action = execution._pop_next_action
    next_action = execution.next_action


__all__ = ["ChunkScheduler"]
