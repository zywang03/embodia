"""Small shared helpers used across inferaxis internals."""

from __future__ import annotations

import math

from ..core.errors import InterfaceValidationError
from .coerce import as_action, as_frame


def reset_if_possible(component: object) -> None:
    """Reset a stateful component when it exposes ``reset()``."""

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


__all__ = ["as_action", "as_frame", "reset_if_possible", "validate_positive_number"]
