"""Private coercion helpers shared across runtime modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...core.schema import Action, Frame
from ...core.transform import coerce_action, coerce_frame


def as_frame(value: Frame | Mapping[str, Any]) -> Frame:
    """Return a frame object without copying when already standardized."""

    if isinstance(value, Frame):
        return value
    return coerce_frame(value)


def as_action(value: Action | Mapping[str, Any]) -> Action:
    """Return an action object without copying when already standardized."""

    if isinstance(value, Action):
        return value
    return coerce_action(value)


def maybe_as_action(value: object) -> Action | None:
    """Return one action when a value looks action-like, else ``None``.

    This lets embodia treat ``robot.act(...)`` / ``send_command(...)`` return
    values as the final executed action without forcing every robot backend to
    return one.
    """

    if isinstance(value, Action):
        return value
    if isinstance(value, Mapping):
        return coerce_action(value)
    return None


__all__ = ["as_action", "as_frame", "maybe_as_action"]
