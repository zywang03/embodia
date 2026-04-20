"""Core ChunkScheduler state container and public entrypoints."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import time

from ....core.errors import InterfaceValidationError
from ....core.schema import Action
from ..optimizers import BlendWeight, _normalize_blend_weight
from ..protocols import ActionSource, ActionSourceProtocol
from .action_ops import ChunkSchedulerActionOpsMixin
from .request_ops import ChunkSchedulerRequestOpsMixin
from .runtime_ops import ChunkSchedulerRuntimeOpsMixin
from .shared import _BRIDGE_CONTEXT_WINDOW, _CompletedChunk


@dataclass(slots=True)
class ChunkScheduler(
    ChunkSchedulerRuntimeOpsMixin,
    ChunkSchedulerRequestOpsMixin,
    ChunkSchedulerActionOpsMixin,
):
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
    control_period_s: float | None = None
    warmup_requests: int = 3
    profile_delay_requests: int = 0
    interpolation_steps: int = 0
    enable_mismatch_bridge: bool = True
    max_chunk_size: int | None = None
    use_overlap_blend: bool = False
    overlap_current_weight: BlendWeight = 0.5
    enable_rtc: bool = False
    clock: Callable[[], float] = time.perf_counter
    _buffer: deque[Action] = field(default_factory=deque, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)
    _control_step: int = field(default=0, init=False, repr=False)
    _reference_chunk_size: int = field(default=0, init=False, repr=False)
    _active_chunk_snapshot: list[Action] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _active_chunk_consumed_steps: int = field(default=0, init=False, repr=False)
    _active_source_plan_length: int = field(default=0, init=False, repr=False)
    _latency_steps_estimate: float = field(default=0.0, init=False, repr=False)
    _latency_observation_count: int = field(default=0, init=False, repr=False)
    _rtc_seed_chunk: list[Action] = field(default_factory=list, init=False, repr=False)
    _startup_latency_bootstrap_complete: bool = field(
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
    _transition_bridge_buffer: deque[Action] = field(
        default_factory=deque,
        init=False,
        repr=False,
    )
    _recent_executed_raw_actions: deque[Action] = field(
        default_factory=lambda: deque(maxlen=_BRIDGE_CONTEXT_WINDOW),
        init=False,
        repr=False,
    )
    _avg_continuous_step_length_ema: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

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

        if self.control_period_s is not None:
            if isinstance(self.control_period_s, bool) or not isinstance(
                self.control_period_s,
                (int, float),
            ):
                raise InterfaceValidationError(
                    "control_period_s must be a real number > 0 when provided."
                )
            self.control_period_s = float(self.control_period_s)
            if self.control_period_s <= 0.0:
                raise InterfaceValidationError(
                    "control_period_s must be > 0 when provided, got "
                    f"{self.control_period_s!r}."
                )

        for field_name in ("warmup_requests", "profile_delay_requests"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise InterfaceValidationError(
                    f"{field_name} must be an int >= 0."
                )
            if value < 0:
                raise InterfaceValidationError(
                    f"{field_name} must be >= 0, got {value!r}."
                )

        if isinstance(self.interpolation_steps, bool) or not isinstance(
            self.interpolation_steps,
            int,
        ):
            raise InterfaceValidationError("interpolation_steps must be an int >= 0.")
        if self.interpolation_steps < 0:
            raise InterfaceValidationError(
                "interpolation_steps must be >= 0, got "
                f"{self.interpolation_steps!r}."
            )
        if not isinstance(self.enable_mismatch_bridge, bool):
            raise InterfaceValidationError("enable_mismatch_bridge must be a bool.")

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

        if not isinstance(self.enable_rtc, bool):
            raise InterfaceValidationError("enable_rtc must be a bool.")

        self.overlap_current_weight = _normalize_blend_weight(
            self.overlap_current_weight,
            field_name="overlap_current_weight",
        )

        self._latency_steps_estimate = self.initial_latency_steps
        self._startup_latency_bootstrap_complete = (
            self.control_period_s is None
            or (self.warmup_requests + self.profile_delay_requests) == 0
        )

    def reset(self) -> None:
        """Discard buffered and in-flight chunks but keep learned latency."""

        self._buffer.clear()
        self._global_step = 0
        self._control_step = 0
        self._reference_chunk_size = 0
        self._active_chunk_snapshot.clear()
        self._active_chunk_consumed_steps = 0
        self._active_source_plan_length = 0
        self._rtc_seed_chunk.clear()
        self._execution_buffer.clear()
        self._transition_bridge_buffer.clear()
        self._recent_executed_raw_actions.clear()
        self._avg_continuous_step_length_ema = None
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

    def latency_estimate_ready(self) -> bool:
        """Return whether async execution can trust the current latency estimate."""

        return self._startup_latency_bootstrap_complete


__all__ = ["ChunkScheduler"]
