"""Latency estimation and raw/control-step projection helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from ....core.errors import InterfaceValidationError
from ....core.schema import Action


@dataclass(slots=True)
class LatencyTracker:
    """Track latency estimates and projection state for scheduler requests.

    Startup profile sampling remains the responsibility of bootstrap orchestration.
    """

    latency_ema_beta: float = 0.5
    initial_latency_steps: float = 0.0
    fixed_latency_steps: float | None = None
    control_period_s: float | None = None
    warmup_requests: int = 3
    profile_delay_requests: int = 0
    interpolation_steps: int = 0
    latency_steps_offset: int = 0
    estimate: float = field(default=0.0, init=False)
    observation_count: int = field(default=0, init=False)
    bootstrap_complete: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.refresh_mode()

    def refresh_mode(self) -> None:
        if self.fixed_latency_steps is not None:
            self.estimate = self.fixed_latency_steps
            self.bootstrap_complete = True
            return
        self.estimate = self.initial_latency_steps
        self.bootstrap_complete = (
            self.control_period_s is None
            or (self.warmup_requests + self.profile_delay_requests) == 0
        )

    def estimated_latency_steps(self) -> int:
        return max(int(math.ceil(self.estimate)), 0)

    def latency_estimate_ready(self) -> bool:
        return self.bootstrap_complete

    def control_steps_for_raw_count(self, raw_steps: int) -> int:
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

    def raw_segment_control_steps(self, *, has_successor: bool) -> int:
        if has_successor:
            return 1 + self.interpolation_steps
        return 1

    def project_control_latency_to_raw_steps(
        self,
        *,
        control_latency_steps: int,
        raw_count: int,
        execution_buffer_steps: int,
    ) -> int:
        if raw_count <= 0:
            return max(int(control_latency_steps), 0)
        remaining = max(int(control_latency_steps), 0)
        for raw_offset in range(raw_count):
            if raw_offset == 0 and execution_buffer_steps:
                segment_steps = max(int(execution_buffer_steps), 0)
            else:
                segment_steps = self.raw_segment_control_steps(
                    has_successor=raw_offset < (raw_count - 1),
                )
            if remaining <= segment_steps:
                return raw_offset + 1
            remaining -= segment_steps
        return raw_count

    def estimated_request_latency_steps(
        self,
        *,
        control_latency_steps: int,
        raw_count: int,
        execution_buffer_steps: int,
    ) -> int:
        projected = self.project_control_latency_to_raw_steps(
            control_latency_steps=control_latency_steps,
            raw_count=raw_count,
            execution_buffer_steps=execution_buffer_steps,
        )
        return max(projected + self.latency_steps_offset, 0)

    def update(self, *, waited_steps: int) -> None:
        if self.fixed_latency_steps is not None:
            self.estimate = self.fixed_latency_steps
            self.bootstrap_complete = True
            return
        self.observation_count += 1
        if self.observation_count <= self.warmup_requests:
            return
        if self.observation_count == self.warmup_requests + 1 and self.estimate <= 0:
            self.estimate = float(waited_steps)
            self.bootstrap_complete = True
            return
        self.estimate = (
            1.0 - self.latency_ema_beta
        ) * self.estimate + self.latency_ema_beta * float(waited_steps)
        self.bootstrap_complete = True

    def observed_latency_steps_from_duration(self, inference_time_s: float) -> int:
        if self.control_period_s is None or inference_time_s <= 0.0:
            return 1
        return max(int(math.ceil(inference_time_s / self.control_period_s)), 1)


def estimated_latency_steps(self) -> int:
    """Return the internal control-step latency estimate."""

    return self._base_estimated_latency_steps()


def _base_estimated_latency_steps(self) -> int:
    """Return the internal measured control-step latency before offsets."""

    return max(int(math.ceil(self._latency_steps_estimate)), 0)


def latency_estimate_ready(self) -> bool:
    """Return whether async execution can trust the current latency estimate."""

    return self._startup_latency_bootstrap_complete


def _control_steps_for_raw_count(self, raw_steps: int) -> int:
    """Return control-step count for ``raw_steps`` raw actions."""

    if isinstance(raw_steps, bool) or not isinstance(raw_steps, int):
        raise InterfaceValidationError(
            f"raw_steps must be an int, got {type(raw_steps).__name__}."
        )
    if raw_steps < 0:
        raise InterfaceValidationError(f"raw_steps must be >= 0, got {raw_steps!r}.")
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

    if self._execution_buffer:
        total_steps = len(self._execution_buffer)
        total_steps += self._control_steps_for_raw_count(max(len(self._buffer) - 1, 0))
        return total_steps
    return self._control_steps_for_actions(self._buffer)


def _project_control_latency_to_raw_steps(
    self,
    *,
    control_latency_steps: int,
    buffer_actions: Sequence[Action] | None = None,
    execution_buffer_steps: int | None = None,
) -> int:
    """Project control-step latency back into raw-step chunk semantics."""

    active_buffer = self._buffer if buffer_actions is None else buffer_actions
    if not active_buffer:
        return max(int(control_latency_steps), 0)

    remaining_control_steps = max(int(control_latency_steps), 0)
    current_execution_steps = (
        len(self._execution_buffer)
        if execution_buffer_steps is None
        else max(int(execution_buffer_steps), 0)
    )

    for raw_offset in range(len(active_buffer)):
        if raw_offset == 0 and current_execution_steps:
            segment_steps = current_execution_steps
        else:
            segment_steps = self._raw_segment_control_steps(
                has_successor=raw_offset < (len(active_buffer) - 1),
            )
        if remaining_control_steps <= segment_steps:
            return raw_offset + 1
        remaining_control_steps -= segment_steps
    return len(active_buffer)


def _estimated_request_latency_steps(
    self,
    *,
    control_latency_steps: int,
    buffer_actions: Sequence[Action] | None = None,
    execution_buffer_steps: int | None = None,
) -> int:
    """Return the raw-step latency hint exposed on async requests."""

    projected_steps = self._project_control_latency_to_raw_steps(
        control_latency_steps=control_latency_steps,
        buffer_actions=buffer_actions,
        execution_buffer_steps=execution_buffer_steps,
    )
    return max(
        projected_steps + self._validated_latency_steps_offset(),
        0,
    )


def _update_latency_estimate(self, waited_steps: int) -> None:
    """Update ``H_hat`` using the latest observed control-step delay."""

    if self.fixed_latency_steps is not None:
        self._latency_steps_estimate = self.fixed_latency_steps
        return

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
        1.0 - self.latency_ema_beta
    ) * self._latency_steps_estimate + self.latency_ema_beta * float(waited_steps)


def _observed_latency_steps_from_duration(self, inference_time_s: float) -> int:
    """Convert one measured request duration into control-step latency."""

    if self.control_period_s is None:
        return 1
    if inference_time_s <= 0.0:
        return 1
    return max(int(math.ceil(inference_time_s / self.control_period_s)), 1)
