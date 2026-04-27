"""Raw chunk buffering and execution cursor helpers for scheduler v2."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....core.schema import Action
from . import actions


@dataclass(slots=True)
class RawChunkBuffer:
    """Track raw scheduler actions without rebuilding stale prefixes."""

    actions: list[Action] = field(default_factory=list)
    start_index: int = 0
    global_step: int = 0
    active_chunk_consumed_steps: int = 0
    active_chunk_waited_raw_steps: int = 0
    active_source_plan_length: int = 0

    @property
    def has_actions(self) -> bool:
        return self.start_index < len(self.actions)

    @property
    def remaining_raw_count(self) -> int:
        return max(len(self.actions) - self.start_index, 0)

    def reset(self) -> None:
        self.actions.clear()
        self.start_index = 0
        self.global_step = 0
        self.active_chunk_consumed_steps = 0
        self.active_chunk_waited_raw_steps = 0
        self.active_source_plan_length = 0

    def current_action(self) -> Action:
        return self.actions[self.start_index]

    def next_action(self) -> Action | None:
        next_index = self.start_index + 1
        if next_index >= len(self.actions):
            return None
        return self.actions[next_index]

    def remaining_actions(self) -> list[Action]:
        return self.actions[self.start_index :]

    def accept_chunk(
        self,
        *,
        actions: list[Action],
        request_step: int,
        current_raw_step: int,
        source_plan_length: int,
    ) -> int:
        stale_steps = max(current_raw_step - request_step, 0)
        self.actions = actions
        self.start_index = min(stale_steps, len(actions))
        self.active_chunk_consumed_steps = self.start_index
        self.active_chunk_waited_raw_steps = 0
        self.active_source_plan_length = source_plan_length
        return stale_steps

    def advance_raw_step(self) -> None:
        if not self.has_actions:
            return
        self.start_index += 1
        self.active_chunk_consumed_steps = min(
            self.active_chunk_consumed_steps + 1,
            len(self.actions),
        )
        self.active_chunk_waited_raw_steps += 1
        self.global_step += 1


@dataclass(slots=True)
class ExecutionCursor:
    """Emit execution actions lazily from raw actions plus interpolation."""

    buffer: RawChunkBuffer
    interpolation_steps: int = 0
    _segment_slot: int = 0

    def reset(self) -> None:
        self._segment_slot = 0

    @property
    def at_raw_boundary(self) -> bool:
        return self._segment_slot == 0

    @property
    def remaining_segment_steps(self) -> int:
        if not self.buffer.has_actions:
            return 0
        if self.interpolation_steps <= 0 or self.buffer.next_action() is None:
            return 1 if self._segment_slot == 0 else 0
        return max(self.interpolation_steps + 1 - self._segment_slot, 0)

    def next_action(self) -> Action:
        left_action = self.buffer.current_action()
        right_action = self.buffer.next_action()

        if self.interpolation_steps <= 0 or right_action is None:
            self.buffer.advance_raw_step()
            self._segment_slot = 0
            return left_action

        if self._segment_slot == 0:
            self._segment_slot = 1
            return left_action

        right_weight = self._segment_slot / float(self.interpolation_steps + 1)
        emitted = actions.interpolate_action(
            left_action,
            right_action,
            right_weight=right_weight,
        )

        if self._segment_slot >= self.interpolation_steps:
            self.buffer.advance_raw_step()
            self._segment_slot = 0
        else:
            self._segment_slot += 1

        return emitted
