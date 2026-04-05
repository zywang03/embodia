"""Helpers for the inference side of embodia's unified runtime data flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.schema import Action, Frame
from ..core.transform import coerce_action, coerce_frame
from .checks import validate_action, validate_frame


@dataclass(slots=True)
class StepResult:
    """Result of one normalized robot -> model -> robot runtime step."""

    frame: Frame
    action: Action


def run_step(
    robot: object,
    model: object,
    *,
    frame: Frame | Mapping[str, Any] | None = None,
    execute_action: bool = True,
    reset_model: bool = False,
) -> StepResult:
    """Run one normalized data-flow step.

    The flow is:

    1. observe a frame from the robot, unless a frame is provided
    2. normalize and validate the frame
    3. step the model with the normalized frame
    4. normalize and validate the action
    5. optionally execute the action on the robot
    """

    if reset_model:
        model.reset()

    raw_frame = robot.observe() if frame is None else frame
    normalized_frame = coerce_frame(raw_frame)
    validate_frame(normalized_frame)

    raw_action = model.step(normalized_frame)
    normalized_action = coerce_action(raw_action)
    validate_action(normalized_action)

    if execute_action:
        robot.act(normalized_action)

    return StepResult(frame=normalized_frame, action=normalized_action)


__all__ = ["StepResult", "run_step"]
