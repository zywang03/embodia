"""Internal helpers for inference-time overlap blending."""

from __future__ import annotations

from collections.abc import Sequence

from ...core.errors import InterfaceValidationError

BlendWeight = float | tuple[float, float]


def _normalize_blend_weight(value: object, *, field_name: str) -> BlendWeight:
    """Normalize one scalar or ``(low, high)`` blend-weight config."""

    if isinstance(value, bool):
        raise InterfaceValidationError(
            f"{field_name} must be a real number in [0, 1] or a pair "
            "(low, high) with 0 <= low <= high <= 1."
        )

    if isinstance(value, (int, float)):
        weight = float(value)
        if not 0.0 <= weight <= 1.0:
            raise InterfaceValidationError(
                f"{field_name} must be in [0, 1], got {weight!r}."
            )
        return weight

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if len(items) != 2:
            raise InterfaceValidationError(
                f"{field_name} must contain exactly two values when given as a "
                f"schedule, got {len(items)}."
            )
        low, high = (
            _normalize_blend_weight(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(items)
        )
        assert isinstance(low, float)
        assert isinstance(high, float)
        if low > high:
            raise InterfaceValidationError(
                f"{field_name} must satisfy low <= high, got {(low, high)!r}."
            )
        return (low, high)

    raise InterfaceValidationError(
        f"{field_name} must be a real number in [0, 1] or a pair "
        "(low, high) with 0 <= low <= high <= 1."
    )


__all__ = ["BlendWeight"]
