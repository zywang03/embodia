"""Shared constants and lightweight data containers for chunk scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from ....core.schema import Action, BuiltinCommandKind
from ..protocols import ChunkRequest

_NON_BLENDABLE_OVERLAP_COMMANDS = frozenset(
    {
        BuiltinCommandKind.GRIPPER_POSITION,
        BuiltinCommandKind.GRIPPER_POSITION_DELTA,
        BuiltinCommandKind.GRIPPER_VELOCITY,
        BuiltinCommandKind.GRIPPER_OPEN_CLOSE,
    }
)
_SLOW_RTC_WARMUP_THRESHOLD_S = 0.5


@dataclass(slots=True)
class _CompletedChunk:
    """One finished chunk request prepared against its launch buffer."""

    request: ChunkRequest
    prepared_actions: list[Action]
    source_plan_length: int
    launch_control_step: int = 0


@dataclass(slots=True)
class _RequestJob:
    """One request plus the immutable buffer snapshot used to prepare it."""

    request: ChunkRequest
    launch_buffer: list[Action]
    launch_control_step: int = 0


__all__ = [
    "_CompletedChunk",
    "_NON_BLENDABLE_OVERLAP_COMMANDS",
    "_RequestJob",
    "_SLOW_RTC_WARMUP_THRESHOLD_S",
]
