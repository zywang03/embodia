"""Structural protocol definitions for third-party compatibility."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import Action, Frame, ModelSpec, RobotSpec


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
class ModelProtocol(Protocol):
    """Structural interface for runtime-compatible model implementations."""

    def get_spec(self) -> ModelSpec:
        """Return a static description of model requirements."""

    def reset(self) -> None:
        """Reset any internal model state."""

    def infer(self, frame: Frame) -> Action:
        """Consume one frame and produce one action."""


@runtime_checkable
class ChunkModelProtocol(Protocol):
    """Structural interface for chunk-producing model implementations."""

    def get_spec(self) -> ModelSpec:
        """Return a static description of model requirements."""

    def reset(self) -> None:
        """Reset any internal model state."""

    def infer_chunk(self, frame: Frame, request: object) -> list[Action]:
        """Consume one frame and produce one action chunk."""


__all__ = ["ChunkModelProtocol", "ModelProtocol", "RobotProtocol"]
