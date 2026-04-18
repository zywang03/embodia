"""Protocol and type definitions for inference-time helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame


@runtime_checkable
class ActionOptimizerProtocol(Protocol):
    """Callable optimizer that transforms one standardized action."""

    def __call__(
        self,
        action: Action,
        frame: Frame,
    ) -> Action:
        """Return an optimized action."""


ActionOptimizer = Callable[[Action, Frame], Action]
ActionChunk = Sequence[Action]
ActionPlan = Action | ActionChunk


@runtime_checkable
class ActionSourceProtocol(Protocol):
    """Callable that returns one future action or one future action chunk."""

    def __call__(
        self,
        frame: Frame,
        request: "ChunkRequest",
    ) -> ActionPlan:
        """Return one action or one action chunk."""


ActionSource = Callable[[Frame, "ChunkRequest"], ActionPlan]


@dataclass(slots=True)
class RtcArgs:
    """Optional runtime-to-policy chunk execution hints."""

    prev_action_chunk: list[Action] = field(default_factory=list)
    inference_delay: int = 1
    execute_horizon: int = 0


@dataclass(slots=True)
class ChunkRequest:
    """Runtime context for one overlap-aware action request.

    ``history_actions`` contains the overlap tail from the currently active
    chunk. Sources can use it as conditioning context when producing the next
    future actions.

    ``plan_start_step`` marks where an overlap-prefixed response would begin in
    global-step coordinates. Future-only responses typically begin at
    ``request_step`` instead.

    ``rtc_args`` is populated when the owning runtime enables real-time chunk
    hints. It exposes the full currently active chunk snapshot, the index where
    RTC guidance should begin inside that snapshot, and the snapshot length.
    For easier interoperability with RTC-style policy code, the same values
    are also mirrored onto ``prev_action_chunk``, ``inference_delay``, and
    ``execute_horizon`` directly on this object.
    """

    request_step: int
    request_time_s: float
    history_start: int
    history_end: int
    active_chunk_length: int
    remaining_steps: int
    overlap_steps: int
    latency_steps: int
    request_trigger_steps: int
    plan_start_step: int
    history_actions: list[Action] = field(default_factory=list)
    prev_action_chunk: list[Action] | None = None
    inference_delay: int | None = None
    execute_horizon: int | None = None
    rtc_args: RtcArgs | None = None

    def __post_init__(self) -> None:
        """Keep RTC mirrors synchronized when either form is provided."""

        if self.rtc_args is not None:
            if self.prev_action_chunk is None:
                self.prev_action_chunk = self.rtc_args.prev_action_chunk
            elif self.prev_action_chunk != self.rtc_args.prev_action_chunk:
                raise InterfaceValidationError(
                    "ChunkRequest.prev_action_chunk must match "
                    "ChunkRequest.rtc_args.prev_action_chunk when both are provided."
                )

            if self.inference_delay is None:
                self.inference_delay = self.rtc_args.inference_delay
            elif self.inference_delay != self.rtc_args.inference_delay:
                raise InterfaceValidationError(
                    "ChunkRequest.inference_delay must match "
                    "ChunkRequest.rtc_args.inference_delay when both are provided."
                )

            if self.execute_horizon is None:
                self.execute_horizon = self.rtc_args.execute_horizon
            elif self.execute_horizon != self.rtc_args.execute_horizon:
                raise InterfaceValidationError(
                    "ChunkRequest.execute_horizon must match "
                    "ChunkRequest.rtc_args.execute_horizon when both are provided."
                )
            return

        if (
            self.prev_action_chunk is not None
            or self.inference_delay is not None
            or self.execute_horizon is not None
        ):
            self.rtc_args = RtcArgs(
                prev_action_chunk=[] if self.prev_action_chunk is None else self.prev_action_chunk,
                inference_delay=1 if self.inference_delay is None else self.inference_delay,
                execute_horizon=0 if self.execute_horizon is None else self.execute_horizon,
            )


__all__ = [
    "ActionChunk",
    "ActionOptimizer",
    "ActionOptimizerProtocol",
    "ActionPlan",
    "ActionSource",
    "ActionSourceProtocol",
    "ChunkRequest",
    "RtcArgs",
]
