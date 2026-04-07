"""Shared helpers for inference-time runtime modules."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ...core.transform import coerce_action, coerce_frame


def reset_if_possible(component: object) -> None:
    """Reset a stateful runtime component when it exposes ``reset()``."""

    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()


def validate_positive_number(value: object, field_name: str) -> float:
    """Validate one finite positive real number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterfaceValidationError(
            f"{field_name} must be a real number, got {type(value).__name__}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise InterfaceValidationError(f"{field_name} must be finite, got {value!r}.")
    if number <= 0.0:
        raise InterfaceValidationError(f"{field_name} must be > 0, got {value!r}.")
    return number


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


__all__ = ["as_action", "as_frame", "reset_if_possible", "validate_positive_number"]
