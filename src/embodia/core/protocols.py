"""Structural protocol definitions for third-party compatibility."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import Action, Frame, PolicySpec, RobotSpec


@runtime_checkable
class RobotProtocol(Protocol):
    """Structural interface for runtime-compatible robot implementations."""

    def get_spec(self) -> RobotSpec:
        """Return a static description of robot capabilities."""

    def observe(self) -> Frame:
        """Return the latest observation frame."""

    def act(self, action: Action) -> None:
        """Execute an action on the robot."""

    def reset(self) -> Frame:
        """Reset the robot/environment and return the first observation."""


@runtime_checkable
class PolicyProtocol(Protocol):
    """Structural interface for runtime-compatible policy implementations."""

    def get_spec(self) -> PolicySpec:
        """Return a static description of policy requirements."""

    def reset(self) -> None:
        """Reset any internal policy state."""

    def infer(self, frame: Frame) -> Action:
        """Consume one frame and produce one action."""


@runtime_checkable
class ChunkPolicyProtocol(Protocol):
    """Structural interface for chunk-producing policy implementations."""

    def get_spec(self) -> PolicySpec:
        """Return a static description of policy requirements."""

    def reset(self) -> None:
        """Reset any internal policy state."""

    def infer_chunk(self, frame: Frame, request: object) -> list[Action]:
        """Consume one frame and produce one action chunk."""


__all__ = ["ChunkPolicyProtocol", "PolicyProtocol", "RobotProtocol"]
