"""Validation strategy helpers for realtime inference runtime paths."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from ...core.errors import InterfaceValidationError


UNSET_VALIDATION: Final = object()


class ValidationMode(StrEnum):
    """Named validation strategies for runtime and scheduler hot paths."""

    ALWAYS = "always"
    STARTUP = "startup"
    OFF = "off"


def resolve_validation_mode(
    *,
    validation: ValidationMode | str | None,
    startup_validation_only: object,
    field_name: str,
) -> tuple[ValidationMode, bool]:
    """Resolve current validation strategy plus the normalized legacy flag."""

    legacy_provided = startup_validation_only is not UNSET_VALIDATION
    legacy_value: bool | None = None
    if legacy_provided:
        if not isinstance(startup_validation_only, bool):
            raise InterfaceValidationError(
                f"{field_name}.startup_validation_only must be a bool."
            )
        legacy_value = startup_validation_only

    resolved_validation: ValidationMode | None = None
    if validation is not None:
        try:
            resolved_validation = ValidationMode(str(validation))
        except ValueError as exc:
            raise InterfaceValidationError(
                f"{field_name}.validation must be 'always', 'startup', or "
                f"'off', got {validation!r}."
            ) from exc

    if resolved_validation is None:
        if not legacy_provided:
            resolved_validation = ValidationMode.STARTUP
        elif legacy_value:
            resolved_validation = ValidationMode.STARTUP
        else:
            resolved_validation = ValidationMode.ALWAYS
    elif legacy_provided:
        legacy_validation = (
            ValidationMode.STARTUP if legacy_value else ValidationMode.ALWAYS
        )
        if resolved_validation is not legacy_validation:
            raise InterfaceValidationError(
                f"{field_name}.validation conflicts with "
                f"{field_name}.startup_validation_only."
            )

    return resolved_validation, resolved_validation is ValidationMode.STARTUP
