"""Private type guards shared across inferaxis internals."""

from __future__ import annotations

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from ..core.transform import coerce_action, coerce_frame


def as_frame(value: object) -> Frame:
    """Return a standardized frame or raise a strict runtime boundary error."""

    try:
        return coerce_frame(value)
    except InterfaceValidationError as exc:
        raise TypeError(str(exc)) from exc


def as_frame_fast(value: object) -> Frame:
    """Return one Frame without re-coercing already standardized instances."""

    if isinstance(value, Frame):
        return value
    return as_frame(value)


def as_action(value: object) -> Action:
    """Return a standardized action or raise a strict runtime boundary error."""

    try:
        return coerce_action(value)
    except InterfaceValidationError as exc:
        raise TypeError(str(exc)) from exc


def as_action_fast(value: object) -> Action:
    """Return one Action without re-coercing already standardized instances."""

    if isinstance(value, Action):
        return value
    return as_action(value)


def maybe_as_action(value: object) -> Action | None:
    """Return one action when a value is action-like, else ``None``."""

    if value is None:
        return None
    try:
        return as_action(value)
    except TypeError:
        return None


def maybe_as_action_fast(value: object) -> Action | None:
    """Return one Action while fast-pathing already standardized instances."""

    if value is None:
        return None
    if isinstance(value, Action):
        return value
    return maybe_as_action(value)


__all__ = [
    "as_action",
    "as_action_fast",
    "as_frame",
    "as_frame_fast",
    "maybe_as_action",
    "maybe_as_action_fast",
]
