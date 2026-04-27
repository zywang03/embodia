"""Validation strategy helpers for realtime inference runtime paths."""

from __future__ import annotations

from enum import StrEnum

from ...core.errors import InterfaceValidationError


class ValidationMode(StrEnum):
    """Named validation strategies for runtime and scheduler hot paths."""

    ALWAYS = "always"
    STARTUP = "startup"
    OFF = "off"


def resolve_validation_mode(
    *,
    validation: ValidationMode | str | None,
    field_name: str,
) -> ValidationMode:
    """Resolve the current validation strategy."""

    if validation is None:
        return ValidationMode.STARTUP
    try:
        return ValidationMode(str(validation))
    except ValueError as exc:
        raise InterfaceValidationError(
            f"{field_name}.validation must be 'always', 'startup', or "
            f"'off', got {validation!r}."
        ) from exc
