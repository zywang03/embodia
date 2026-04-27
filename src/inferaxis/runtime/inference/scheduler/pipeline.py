"""Request pipeline helpers for async chunk scheduling."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .state import _CompletedChunk


@dataclass(slots=True)
class RequestPipeline:
    """Track pending request state for the scheduler orchestrator."""

    pending: Future[_CompletedChunk] | None = None
    executor: ThreadPoolExecutor | None = None

    @property
    def has_pending(self) -> bool:
        return self.pending is not None

    @property
    def has_ready_pending(self) -> bool:
        return self.pending is not None and self.pending.done()

    def clear_pending(self) -> None:
        self.pending = None

    def close(self) -> None:
        if self.pending is not None:
            self.pending.cancel()
            self.pending = None
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def ensure_executor(self) -> ThreadPoolExecutor:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="inferaxis-async-inference",
            )
        return self.executor


__all__ = ["RequestPipeline"]
