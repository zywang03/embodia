"""Action normalization, overlap blending, and interpolation helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from ....core.errors import InterfaceValidationError
from ....core.schema import Action, Command
from ....shared.common import as_action
from ...checks import validate_action
from ..optimizers import _normalize_blend_weight
from ..protocols import ActionPlan
from .shared import _NON_BLENDABLE_OVERLAP_COMMANDS


class ChunkSchedulerActionOpsMixin:
    """Helpers that operate on actions and execution-only smoothing."""

    __slots__ = ()

    def _clone_action(self, action: Action) -> Action:
        """Return a detached copy of one standardized action."""

        return Action(
            commands={
                target: self._clone_command(command)
                for target, command in action.commands.items()
            },
            meta=dict(action.meta),
        )

    def _clone_command(self, command: Command) -> Command:
        """Return a detached copy of one standardized command."""

        return Command(
            command=command.command,
            value=command.value.copy(),
            ref_frame=command.ref_frame,
            meta=dict(command.meta),
        )

    def _clone_actions(self, actions: Sequence[Action]) -> list[Action]:
        """Return detached copies for one sequence of standardized actions."""

        return [self._clone_action(action) for action in actions]

    def _commands_share_layout(self, left: Action, right: Action) -> bool:
        """Return whether two actions have the same command structure."""

        if left.commands.keys() != right.commands.keys():
            return False
        for target, left_command in left.commands.items():
            right_command = right.commands[target]
            if not self._commands_share_target_layout(left_command, right_command):
                return False
        return True

    def _commands_share_target_layout(
        self,
        left_command: Command,
        right_command: Command,
    ) -> bool:
        """Return whether two commands can be aligned structurally."""

        if left_command.command != right_command.command:
            return False
        if left_command.ref_frame != right_command.ref_frame:
            return False
        if left_command.meta != right_command.meta:
            return False
        if left_command.value.shape != right_command.value.shape:
            return False
        return True

    def _actions_match(self, left: Action, right: Action) -> bool:
        """Return whether two actions are structurally identical."""

        if left.meta != right.meta:
            return False
        if not self._commands_share_layout(left, right):
            return False
        for target, left_command in left.commands.items():
            right_command = right.commands[target]
            if not np.array_equal(left_command.value, right_command.value):
                return False
        return True

    def _blend_overlap_action(
        self,
        old_action: Action,
        new_action: Action,
        *,
        overlap_index: int = 0,
        overlap_count: int = 1,
    ) -> Action:
        """Blend one aligned overlap step from old/new chunks."""

        if not self._commands_share_layout(old_action, new_action):
            return self._clone_action(new_action)

        new_weight = self._overlap_new_weight(
            overlap_index=overlap_index,
            overlap_count=overlap_count,
        )
        old_weight = 1.0 - new_weight

        blended_commands: dict[str, Command] = {}
        for target, new_command in new_action.commands.items():
            old_command = old_action.commands[target]
            if new_command.command in _NON_BLENDABLE_OVERLAP_COMMANDS:
                blended_commands[target] = Command(
                    command=new_command.command,
                    value=new_command.value.copy(),
                    ref_frame=new_command.ref_frame,
                    meta=dict(new_command.meta),
                )
                continue
            blended_commands[target] = Command(
                command=new_command.command,
                value=old_command.value * old_weight
                + new_command.value * new_weight,
                ref_frame=new_command.ref_frame,
                meta=dict(new_command.meta),
            )
        return Action(
            commands=blended_commands,
            meta=dict(new_action.meta),
        )

    def _overlap_new_weight(
        self,
        *,
        overlap_index: int,
        overlap_count: int,
    ) -> float:
        """Return the new-chunk blend weight for one overlap step."""

        if overlap_count <= 0:
            raise InterfaceValidationError(
                f"overlap_count must be > 0, got {overlap_count!r}."
            )
        if not 0 <= overlap_index < overlap_count:
            raise InterfaceValidationError(
                "overlap_index must satisfy "
                f"0 <= overlap_index < overlap_count, got "
                f"overlap_index={overlap_index!r}, overlap_count={overlap_count!r}."
            )

        normalized = _normalize_blend_weight(
            self.overlap_current_weight,
            field_name="overlap_current_weight",
        )
        if isinstance(normalized, float):
            return normalized

        low, high = normalized
        if overlap_count == 1:
            return low
        progress = overlap_index / float(overlap_count - 1)
        return low + (high - low) * progress

    def _interpolate_action(
        self,
        left_action: Action,
        right_action: Action,
        *,
        right_weight: float,
    ) -> Action:
        """Return one execution-only interpolated action between raw steps."""

        if right_weight <= 0.0:
            return self._clone_action(left_action)
        if right_weight >= 1.0:
            return self._clone_action(right_action)

        interpolated_commands: dict[str, Command] = {}
        for target, left_command in left_action.commands.items():
            right_command = right_action.commands.get(target)
            if (
                right_command is None
                or not self._commands_share_target_layout(left_command, right_command)
                or left_command.command in _NON_BLENDABLE_OVERLAP_COMMANDS
            ):
                interpolated_commands[target] = self._clone_command(left_command)
                continue

            interpolated_commands[target] = Command(
                command=left_command.command,
                value=left_command.value * (1.0 - right_weight)
                + right_command.value * right_weight,
                ref_frame=left_command.ref_frame,
                meta=dict(left_command.meta),
            )

        return Action(
            commands=interpolated_commands,
            meta=dict(left_action.meta),
        )

    def _build_execution_segment(self) -> deque[Action]:
        """Expand the current raw step into execution actions."""

        if not self._buffer:
            return deque()

        buffer_list = list(self._buffer)
        left_action = buffer_list[0]
        segment: list[Action] = [self._clone_action(left_action)]
        if self.interpolation_steps <= 0 or len(buffer_list) <= 1:
            return deque(segment)

        right_action = buffer_list[1]
        for interpolation_index in range(1, self.interpolation_steps + 1):
            right_weight = interpolation_index / float(self.interpolation_steps + 1)
            segment.append(
                self._interpolate_action(
                    left_action,
                    right_action,
                    right_weight=right_weight,
                )
            )
        return deque(segment)

    def _ensure_execution_buffer(self) -> None:
        """Populate one execution segment for the current raw step."""

        if self._execution_buffer or not self._buffer:
            return
        self._execution_buffer = self._build_execution_segment()

    def _advance_raw_step(self) -> None:
        """Advance raw scheduler state once the current execution segment finishes."""

        if not self._buffer:
            return
        self._buffer.popleft()
        if self._active_chunk_snapshot:
            self._active_chunk_consumed_steps = min(
                self._active_chunk_consumed_steps + 1,
                len(self._active_chunk_snapshot),
            )
        self._active_chunk_waited_raw_steps += 1
        self._global_step += 1

    def _normalize_plan(self, plan: ActionPlan) -> list[Action]:
        """Coerce and validate one returned chunk."""

        if isinstance(plan, Action):
            normalized = [as_action(plan)]
        elif isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
            raise InterfaceValidationError(
                "act_src_fn(frame, request) must return an Action or a sequence "
                f"of Action, got {type(plan).__name__}."
            )
        else:
            normalized = [as_action(item) for item in plan]

        if self.max_chunk_size is not None:
            normalized = normalized[: self.max_chunk_size]
        if not normalized:
            raise InterfaceValidationError(
                "ChunkScheduler received an empty action chunk."
            )

        for action in normalized:
            validate_action(action)
        return self._clone_actions(normalized)
