"""Protocol and type definitions for inference-time helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

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
class ChunkRequest:
    """Runtime context for one overlap-aware action request.

    ``history_actions`` contains the overlap tail from the currently active
    chunk. Sources can use it as conditioning context when producing the next
    future actions.

    ``plan_start_step`` marks where an overlap-prefixed response would begin in
    global-step coordinates. Future-only responses typically begin at
    ``request_step`` instead.
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


__all__ = [
    "ActionChunk",
    "ActionOptimizer",
    "ActionOptimizerProtocol",
    "ActionPlan",
    "ActionSource",
    "ActionSourceProtocol",
    "ChunkRequest",
]
