"""Minimal shared schema for robot/model runtime interaction.

This module intentionally stays lightweight. It standardizes the data objects
that sit between robot adapters and model adapters without turning embodia into
an all-encompassing robotics framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KNOWN_CONTROL_GROUP_KINDS: tuple[str, ...] = (
    "arm",
    "gripper",
    "hand",
    "suction",
    "base",
    "custom",
)


@dataclass(slots=True)
class Frame:
    """One standardized observation frame."""

    timestamp_ns: int
    images: dict[str, Any]
    state: dict[str, Any]
    task: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    sequence_id: int | None = None


@dataclass(slots=True)
class Command:
    """One command for one control group.

    Examples:

    - ``target="left_arm", mode="ee_delta", value=[...]``
    - ``target="left_gripper", mode="scalar_position", value=[0.8]``
    - ``target="right_hand", mode="joint_position", value=[...]``
    """

    target: str
    mode: str
    value: list[float]
    ref_frame: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Action:
    """One control step containing one or more commands."""

    commands: list[Command]
    dt: float = 0.1
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def single(
        cls,
        *,
        target: str,
        mode: str,
        value: list[float],
        dt: float = 0.1,
        ref_frame: str | None = None,
        command_meta: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Action:
        """Build an action containing exactly one command."""

        return cls(
            commands=[
                Command(
                    target=target,
                    mode=mode,
                    value=list(value),
                    ref_frame=ref_frame,
                    meta={} if command_meta is None else dict(command_meta),
                )
            ],
            dt=dt,
            meta={} if meta is None else dict(meta),
        )

    def get_command(self, target: str) -> Command | None:
        """Return the command for ``target`` when present."""

        for command in self.commands:
            if command.target == target:
                return command
        return None


@dataclass(slots=True)
class ControlGroupSpec:
    """Description of one robot control group."""

    name: str
    kind: str
    dof: int
    action_modes: list[str]
    state_keys: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def supports_mode(self, mode: str) -> bool:
        """Return whether the group accepts ``mode``."""

        return mode in self.action_modes


@dataclass(slots=True)
class RobotSpec:
    """Description of one robot embodiment."""

    name: str
    image_keys: list[str]
    groups: list[ControlGroupSpec]
    task_keys: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def get_group(self, name: str) -> ControlGroupSpec | None:
        """Return one group by name when present."""

        for group in self.groups:
            if group.name == name:
                return group
        return None

    def all_action_modes(self) -> list[str]:
        """Return the unique action modes supported across all groups."""

        seen: set[str] = set()
        result: list[str] = []
        for group in self.groups:
            for mode in group.action_modes:
                if mode not in seen:
                    seen.add(mode)
                    result.append(mode)
        return result

    def all_state_keys(self) -> list[str]:
        """Return the unique state keys exposed across all groups."""

        seen: set[str] = set()
        result: list[str] = []
        for group in self.groups:
            for key in group.state_keys:
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        return result


@dataclass(slots=True)
class ModelOutputSpec:
    """Description of one model output command slot."""

    target: str
    mode: str
    dim: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelSpec:
    """Description of one model's required inputs and emitted commands."""

    name: str
    required_image_keys: list[str]
    required_state_keys: list[str]
    required_task_keys: list[str]
    outputs: list[ModelOutputSpec]
    meta: dict[str, Any] = field(default_factory=dict)

    def get_output(self, target: str) -> ModelOutputSpec | None:
        """Return one output spec by target when present."""

        for output in self.outputs:
            if output.target == target:
                return output
        return None


__all__ = [
    "Action",
    "Command",
    "ControlGroupSpec",
    "Frame",
    "KNOWN_CONTROL_GROUP_KINDS",
    "ModelOutputSpec",
    "ModelSpec",
    "RobotSpec",
]
