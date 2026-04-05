"""Recommended abstract base classes for official implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import Action, Frame, ModelSpec, RobotSpec


class RobotBase(ABC):
    """Recommended base class for official robot implementations."""

    @abstractmethod
    def get_spec(self) -> RobotSpec:
        """Return the robot specification."""

    @abstractmethod
    def observe(self) -> Frame:
        """Return the latest observation frame."""

    @abstractmethod
    def act(self, action: Action) -> None:
        """Execute an action on the robot."""

    @abstractmethod
    def reset(self) -> Frame:
        """Reset the robot/environment and return the first observation."""


class ModelBase(ABC):
    """Recommended base class for official model implementations."""

    @abstractmethod
    def get_spec(self) -> ModelSpec:
        """Return the model specification."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal model state."""

    @abstractmethod
    def step(self, frame: Frame) -> Action:
        """Consume one frame and produce one action."""


__all__ = ["ModelBase", "RobotBase"]
