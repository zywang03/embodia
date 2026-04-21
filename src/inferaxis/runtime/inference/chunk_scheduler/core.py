"""Core ChunkScheduler state container and public entrypoints."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import time

from ....core.schema import Action
from ..optimizers import BlendWeight
from ..protocols import ActionSource, ActionSourceProtocol
from .action_ops import ChunkSchedulerActionOpsMixin
from .config_ops import ChunkSchedulerConfigOpsMixin
from .latency_ops import ChunkSchedulerLatencyOpsMixin
from .request_ops import ChunkSchedulerRequestOpsMixin
from .rtc_ops import ChunkSchedulerRtcOpsMixin
from .runtime_ops import ChunkSchedulerRuntimeOpsMixin
from .shared import _CompletedChunk


@dataclass(slots=True)
class ChunkScheduler(
    ChunkSchedulerRuntimeOpsMixin,
    ChunkSchedulerRequestOpsMixin,
    ChunkSchedulerRtcOpsMixin,
    ChunkSchedulerLatencyOpsMixin,
    ChunkSchedulerActionOpsMixin,
    ChunkSchedulerConfigOpsMixin,
):
    """Step-based async chunk scheduler.

    ``action_source(frame, request)`` may return either:

    - one ``Action`` (treated as chunk size ``1``), or
    - a sequence of ``Action`` objects.

    The scheduler keeps one executable buffer of future actions. In async mode,
    it requests the next chunk before the buffer is exhausted. In sync mode, it
    uses the same overlap rule but performs the refresh inline.

    Sources are expected to return future-only chunks that begin at the current
    ``request_step``.
    """

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

    def __post_init__(self) -> None:
        """Validate configuration and initialize runtime state."""

        self._validate_configuration()

    def reset(self) -> None:
        """Discard buffered and in-flight chunks but keep learned latency."""

        self._buffer.clear()
        self._global_step = 0
        self._control_step = 0
        self._active_chunk_snapshot.clear()
        self._active_chunk_consumed_steps = 0
        self._active_chunk_waited_raw_steps = 0
        self._active_source_plan_length = 0
        self._startup_execution_window_validated = False
        self._execution_buffer.clear()
        self._rtc_chunk_total_length = None
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


__all__ = ["ChunkScheduler"]
