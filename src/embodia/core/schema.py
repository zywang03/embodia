"""Minimal shared data schema for robot/model interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ActionMode = Literal["ee_delta", "joint_position", "joint_velocity"]
KNOWN_ACTION_MODES: tuple[str, ...] = (
    "ee_delta",
    "joint_position",
    "joint_velocity",
)


@dataclass(slots=True)
class Frame:
    """A single robot observation frame."""

    timestamp_ns: int
    images: dict[str, Any]
    state: dict[str, Any]
    task: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class Action:
    """A minimal control command produced by a model and consumed by a robot.

    ``ref_frame`` names the reference coordinate frame for actions such as
    end-effector deltas.
    """

    mode: ActionMode
    value: list[float]
    gripper: float | None = None
    ref_frame: str | None = None
    dt: float = 0.1


@dataclass(slots=True)
class RobotSpec:
    """Describes a robot's observable inputs and accepted action modes."""

    name: str
    action_modes: list[str]
    image_keys: list[str]
    state_keys: list[str]


@dataclass(slots=True)
class ModelSpec:
    """Describes a model's required inputs and produced action mode."""

    name: str
    required_image_keys: list[str]
    required_state_keys: list[str]
    output_action_mode: str


__all__ = [
    "Action",
    "ActionMode",
    "Frame",
    "KNOWN_ACTION_MODES",
    "ModelSpec",
    "RobotSpec",
]
