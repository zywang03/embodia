"""Shared exception types for embodia."""

from __future__ import annotations


class InterfaceValidationError(ValueError):
    """Raised when an object does not satisfy the runtime interface contract."""


__all__ = ["InterfaceValidationError"]
