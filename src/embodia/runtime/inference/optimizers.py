"""Action-level inference optimizers."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
import math

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ..checks import validate_action
from .common import as_action, validate_positive_number


@dataclass(slots=True)
class ActionEnsembler:
    """Smooth actions by averaging a short window of recent predictions."""

    window_size: int = 4
    weights: Sequence[float] | None = None
    average_gripper: bool = True
    _history: deque[Action] = field(init=False, repr=False)
    _validated_weights: tuple[float, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate configuration and initialize internal history."""

        if self.window_size <= 0:
            raise InterfaceValidationError(
                f"window_size must be > 0, got {self.window_size!r}."
            )

        self._history = deque(maxlen=self.window_size)

        if self.weights is None:
            return

        if len(self.weights) != self.window_size:
            raise InterfaceValidationError(
                "weights must have the same length as window_size; "
                f"got {len(self.weights)} and {self.window_size}."
            )

        validated = tuple(
            validate_positive_number(weight, f"weights[{index}]")
            for index, weight in enumerate(self.weights)
        )
        self._validated_weights = validated

    def reset(self) -> None:
        """Clear ensemble history."""

        self._history.clear()

    def _is_compatible(self, left: Action, right: Action) -> bool:
        """Return whether two actions can be averaged together."""

        return (
            left.mode == right.mode
            and left.ref_frame == right.ref_frame
            and len(left.value) == len(right.value)
            and math.isclose(left.dt, right.dt, rel_tol=0.0, abs_tol=1e-12)
        )

    def _active_weights(self, size: int) -> tuple[float, ...]:
        """Return the weights aligned with the current history length."""

        if self._validated_weights is None:
            return (1.0,) * size
        return self._validated_weights[-size:]

    def __call__(self, action: Action, frame: Frame) -> Action:
        """Return an ensembled action."""

        del frame

        normalized = as_action(action)
        validate_action(normalized)

        if self._history and not self._is_compatible(self._history[-1], normalized):
            self.reset()

        self._history.append(normalized)
        weights = self._active_weights(len(self._history))
        weight_sum = sum(weights)

        value = [
            sum(
                item.value[index] * weights[item_index]
                for item_index, item in enumerate(self._history)
            )
            / weight_sum
            for index in range(len(normalized.value))
        ]

        gripper = normalized.gripper
        if (
            self.average_gripper
            and gripper is not None
            and all(item.gripper is not None for item in self._history)
        ):
            gripper = (
                sum(
                    float(item.gripper) * weights[item_index]
                    for item_index, item in enumerate(self._history)
                )
                / weight_sum
            )

        return Action(
            mode=normalized.mode,
            value=value,
            gripper=gripper,
            ref_frame=normalized.ref_frame,
            dt=normalized.dt,
        )


__all__ = ["ActionEnsembler"]
