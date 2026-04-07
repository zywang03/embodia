"""Action-level inference optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Command, Frame
from ..checks import validate_action
from .common import as_action


def _clone_command(command: Command) -> Command:
    """Return a detached copy of one command."""

    return Command(
        target=command.target,
        kind=command.kind,
        value=list(command.value),
        ref_frame=command.ref_frame,
        meta=dict(command.meta),
    )


def _clone_action(action: Action) -> Action:
    """Return a detached copy of one action."""

    return Action(
        commands=[_clone_command(command) for command in action.commands],
        dt=action.dt,
        meta=dict(action.meta),
    )


def _command_map(action: Action) -> dict[str, Command]:
    """Index commands by target."""

    return {command.target: command for command in action.commands}


def _compatible_command(left: Command, right: Command) -> bool:
    """Return whether two commands can be blended safely."""

    return (
        left.target == right.target
        and left.kind == right.kind
        and left.ref_frame == right.ref_frame
        and len(left.value) == len(right.value)
    )


def _compatible_action(left: Action, right: Action) -> bool:
    """Return whether two actions can be blended/interpolated."""

    if not math.isclose(left.dt, right.dt, rel_tol=0.0, abs_tol=1e-12):
        return False

    left_map = _command_map(left)
    right_map = _command_map(right)
    if set(left_map) != set(right_map):
        return False

    return all(
        _compatible_command(left_map[target], right_map[target])
        for target in left_map
    )


def _same_target(left: Action, right: Action) -> bool:
    """Return whether two actions describe the same command targets/values."""

    if not _compatible_action(left, right):
        return False

    left_map = _command_map(left)
    right_map = _command_map(right)
    for target in left_map:
        left_command = left_map[target]
        right_command = right_map[target]
        if not all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
            for a, b in zip(left_command.value, right_command.value)
        ):
            return False
    return True


def _blend_action(left: Action, right: Action, ratio: float) -> Action:
    """Blend two compatible actions."""

    left_map = _command_map(left)
    blended_commands: list[Command] = []
    for command in right.commands:
        previous = left_map[command.target]
        blended_commands.append(
            Command(
                target=command.target,
                kind=command.kind,
                value=[
                    start + (target - start) * ratio
                    for start, target in zip(previous.value, command.value)
                ],
                ref_frame=command.ref_frame,
                meta=dict(command.meta),
            )
        )

    return Action(
        commands=blended_commands,
        dt=right.dt,
        meta=dict(right.meta),
    )


@dataclass(slots=True)
class ActionEnsembler:
    """Smooth actions by blending the latest target with the previous output."""

    current_weight: float = 0.5
    _last_action: Action | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate blending configuration."""

        if isinstance(self.current_weight, bool) or not isinstance(
            self.current_weight,
            (int, float),
        ):
            raise InterfaceValidationError(
                "current_weight must be a real number in [0, 1]."
            )
        self.current_weight = float(self.current_weight)
        if not 0.0 <= self.current_weight <= 1.0:
            raise InterfaceValidationError(
                "current_weight must be in [0, 1], got "
                f"{self.current_weight!r}."
            )

    def reset(self) -> None:
        """Clear smoothing history."""

        self._last_action = None

    def __call__(self, action: Action, frame: Frame) -> Action:
        """Return one smoothed action."""

        del frame

        normalized = as_action(action)
        validate_action(normalized)

        if self._last_action is None:
            self._last_action = _clone_action(normalized)
            return _clone_action(normalized)

        if not _compatible_action(self._last_action, normalized):
            self._last_action = _clone_action(normalized)
            return _clone_action(normalized)

        blended = _blend_action(self._last_action, normalized, self.current_weight)
        self._last_action = _clone_action(blended)
        return blended


@dataclass(slots=True)
class ActionInterpolator:
    """Smooth action transitions with a short finite-step linear interpolation."""

    steps: int = 1
    _last_output: Action | None = field(default=None, init=False, repr=False)
    _transition_start: Action | None = field(default=None, init=False, repr=False)
    _target_action: Action | None = field(default=None, init=False, repr=False)
    _transition_phase: int = field(default=0, init=False, repr=False)
    _transition_total: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate interpolation settings."""

        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise InterfaceValidationError("steps must be an int.")
        if self.steps < 0:
            raise InterfaceValidationError(f"steps must be >= 0, got {self.steps!r}.")

    def reset(self) -> None:
        """Clear interpolation state."""

        self._last_output = None
        self._transition_start = None
        self._target_action = None
        self._transition_phase = 0
        self._transition_total = 0

    def __call__(self, action: Action, frame: Frame) -> Action:
        """Return one smoothed action toward the latest target."""

        del frame

        normalized = as_action(action)
        validate_action(normalized)

        if self.steps == 0:
            self._last_output = _clone_action(normalized)
            self._target_action = _clone_action(normalized)
            return _clone_action(normalized)

        if self._last_output is None:
            self._last_output = _clone_action(normalized)
            self._target_action = _clone_action(normalized)
            return _clone_action(normalized)

        if not _compatible_action(self._last_output, normalized):
            self.reset()
            self._last_output = _clone_action(normalized)
            self._target_action = _clone_action(normalized)
            return _clone_action(normalized)

        if self._target_action is None or not _same_target(
            self._target_action,
            normalized,
        ):
            self._transition_start = _clone_action(self._last_output)
            self._target_action = _clone_action(normalized)
            self._transition_phase = 1
            self._transition_total = self.steps + 1
        elif self._transition_phase < self._transition_total:
            self._transition_phase += 1

        if self._transition_start is None or self._target_action is None:
            self._last_output = _clone_action(normalized)
            return _clone_action(normalized)

        ratio = min(self._transition_phase / self._transition_total, 1.0)
        output = _blend_action(self._transition_start, self._target_action, ratio)
        self._last_output = _clone_action(output)

        if ratio >= 1.0:
            self._transition_start = _clone_action(self._target_action)

        return output


__all__ = ["ActionEnsembler", "ActionInterpolator"]
