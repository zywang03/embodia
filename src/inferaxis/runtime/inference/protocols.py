"""Protocol and type definitions for inference-time helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...core.schema import Action, Frame


@runtime_checkable
class ActionOptimizerProtocol(Protocol):
    """Callable optimizer that transforms one standardized action."""

    def __call__(
        self,
        action: Action,
        frame: Frame,
    ) -> Action | Mapping[str, Any]:
        """Return an optimized action-like value."""


ActionOptimizer = Callable[[Action, Frame], Action | Mapping[str, Any]]
ActionChunk = Sequence[Action | Mapping[str, Any]]
ActionPlan = Action | Mapping[str, Any] | ActionChunk


@runtime_checkable
class ActionPlanProviderProtocol(Protocol):
    """Callable that builds one runtime-managed action plan from a frame."""

    def __call__(
        self,
        source: object,
        frame: Frame,
    ) -> ActionPlan:
        """Return one action or one action chunk."""


ActionPlanProvider = Callable[[object, Frame], ActionPlan]


@dataclass(slots=True)
class ChunkRequest:
    """Runtime context for one async chunk request.

    ``history_actions`` contains the overlap tail slice from the currently
    active chunk that the next chunk should condition on.

    ``request_step`` is the global action index that had already been emitted
    when the request was sent.

    ``plan_start_step`` is the global action index where the returned chunk is
    expected to begin logically. This is earlier than the handoff point by
    ``overlap_steps`` so the scheduler can trim stale overlap actions when the
    response becomes active.
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


@runtime_checkable
class ChunkProviderProtocol(Protocol):
    """Callable that builds one overlap-conditioned action chunk."""

    def __call__(
        self,
        source: object,
        frame: Frame,
        request: ChunkRequest,
    ) -> ActionPlan:
        """Return one action-like chunk for the next runtime window."""


ChunkProvider = Callable[[object, Frame, ChunkRequest], ActionPlan]


__all__ = [
    "ActionChunk",
    "ChunkProvider",
    "ChunkProviderProtocol",
    "ChunkRequest",
    "ActionOptimizer",
    "ActionOptimizerProtocol",
    "ActionPlan",
    "ActionPlanProvider",
    "ActionPlanProviderProtocol",
]
