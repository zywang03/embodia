"""Lightweight data containers for chunk scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from ....core.schema import Action
from ..contracts import ChunkRequest


@dataclass(slots=True)
class _CompletedChunk:
    """One finished chunk request prepared against its launch buffer."""

    request: ChunkRequest
    prepared_actions: list[Action]
    source_plan_length: int
    launch_control_step: int = 0


@dataclass(slots=True)
class _RequestJob:
    """One request plus the launch-time buffer snapshot used to prepare it."""

    request: ChunkRequest
    launch_buffer: list[Action]
    launch_control_step: int = 0


__all__ = [
    "_CompletedChunk",
    "_RequestJob",
]
