"""Shared rendering helpers for profiling reports."""

from __future__ import annotations


def _format_profile_value(value: float) -> str:
    """Format one numeric profile value compactly for SVG labels."""

    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.4g}"


__all__ = ["_format_profile_value"]
