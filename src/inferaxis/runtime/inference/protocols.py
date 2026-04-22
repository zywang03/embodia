"""Protocol definitions for inference-time action sources."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ...core.schema import Frame
from .contracts import ActionPlan, ChunkRequest


@runtime_checkable
class ActionSourceProtocol(Protocol):
    """Callable that returns one future action or one future action chunk."""

    def __call__(
        self,
        frame: Frame,
        request: ChunkRequest,
    ) -> ActionPlan:
        """Return one action or one action chunk."""


ActionSource = Callable[[Frame, ChunkRequest], ActionPlan]


__all__ = [
    "ActionSource",
    "ActionSourceProtocol",
]
