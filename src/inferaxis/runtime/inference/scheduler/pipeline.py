"""Request pipeline helpers for async chunk scheduling."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass

from .state import _CompletedChunk


@dataclass(slots=True)
class RequestPipeline:
    """Track pending request state for the scheduler orchestrator."""

    pending: Future[_CompletedChunk] | None = None

    @property
    def has_pending(self) -> bool:
        return self.pending is not None

    def clear_pending(self) -> None:
        self.pending = None


__all__ = ["RequestPipeline"]
