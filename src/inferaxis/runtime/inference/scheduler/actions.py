"""Action normalization, overlap blending, and interpolation helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from ....core.errors import InterfaceValidationError
from ....core.schema import Action, BuiltinCommandKind, Command
from ....shared.coerce import as_action_fast
from ...checks import validate_action
from ..contracts import ActionPlan
from ..optimizers import _normalize_blend_weight


_NON_BLENDABLE_OVERLAP_COMMANDS = frozenset(
    {
        BuiltinCommandKind.GRIPPER_POSITION,
        BuiltinCommandKind.GRIPPER_POSITION_DELTA,
        BuiltinCommandKind.GRIPPER_VELOCITY,
        BuiltinCommandKind.GRIPPER_OPEN_CLOSE,
    }
)


def _materialize_action(
    self,
    *,
    commands: dict[str, Command],
    meta: dict[str, Any],
) -> Action:
    """Build one trusted ``Action`` without re-running schema coercion."""

    action = object.__new__(Action)
    action.commands = commands
    action.meta = meta
    return action


def _materialize_command(
    self,
    *,
    command: str,
    value: np.ndarray,
    ref_frame: str | None,
    meta: dict[str, Any],
) -> Command:
    """Build one trusted ``Command`` without re-running schema coercion."""

    materialized = object.__new__(Command)
    materialized.command = command
    materialized.value = value
    materialized.ref_frame = ref_frame
    materialized.meta = meta
    return materialized


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
        return new_action

    if all(
        command.command in _NON_BLENDABLE_OVERLAP_COMMANDS
        for command in new_action.commands.values()
    ):
        return new_action

    new_weight = self._overlap_new_weight(
        overlap_index=overlap_index,
        overlap_count=overlap_count,
    )
    old_weight = 1.0 - new_weight

    blended_commands: dict[str, Command] = {}
    for target, new_command in new_action.commands.items():
        old_command = old_action.commands[target]
        if new_command.command in _NON_BLENDABLE_OVERLAP_COMMANDS:
            blended_commands[target] = new_command
            continue
        blended_commands[target] = self._materialize_command(
            command=new_command.command,
            value=old_command.value * old_weight + new_command.value * new_weight,
            ref_frame=new_command.ref_frame,
            meta=new_command.meta,
        )
    return self._materialize_action(
        commands=blended_commands,
        meta=new_action.meta,
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
        return left_action
    if right_weight >= 1.0:
        return right_action

    interpolated_commands: dict[str, Command] = {}
    has_interpolated_target = False
    for target, left_command in left_action.commands.items():
        right_command = right_action.commands.get(target)
        if (
            right_command is None
            or not self._commands_share_target_layout(left_command, right_command)
            or left_command.command in _NON_BLENDABLE_OVERLAP_COMMANDS
        ):
            interpolated_commands[target] = left_command
            continue

        has_interpolated_target = True
        interpolated_commands[target] = self._materialize_command(
            command=left_command.command,
            value=left_command.value * (1.0 - right_weight)
            + right_command.value * right_weight,
            ref_frame=left_command.ref_frame,
            meta=left_command.meta,
        )

    if not has_interpolated_target:
        return left_action

    return self._materialize_action(
        commands=interpolated_commands,
        meta=left_action.meta,
    )


def _build_execution_segment(self) -> deque[Action]:
    """Expand the current raw step into execution actions."""

    if not self._buffer:
        return deque()

    left_action = self._buffer[0]
    if self.interpolation_steps <= 0 or len(self._buffer) <= 1:
        return deque([left_action])

    segment: list[Action] = [left_action]
    right_action = self._buffer[1]
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
        normalized = [as_action_fast(plan)]
    elif isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
        raise InterfaceValidationError(
            "act_src_fn(frame, request) must return an Action or a sequence "
            f"of Action, got {type(plan).__name__}."
        )
    else:
        normalized = [as_action_fast(item) for item in plan]

    if self.max_chunk_size is not None:
        normalized = normalized[: self.max_chunk_size]
    if not normalized:
        raise InterfaceValidationError("ChunkScheduler received an empty action chunk.")

    if self.runtime_validation_enabled():
        for action in normalized:
            validate_action(action)
    return list(normalized)
