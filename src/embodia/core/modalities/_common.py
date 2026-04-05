"""Shared helpers for modality-specific mapping configuration."""

from __future__ import annotations

from collections.abc import Mapping


def resolve_string_mapping(owner: object, attr_name: str) -> Mapping[str, str]:
    """Resolve ``dict[str, str]``-like class configuration."""

    cls = owner if isinstance(owner, type) else type(owner)
    value = getattr(cls, attr_name, {})
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{cls.__name__}.{attr_name} must be a mapping, "
            f"got {type(value).__name__}."
        )

    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(
                f"{cls.__name__}.{attr_name} must be a mapping[str, str], "
                f"got key {key!r} and value {item!r}."
            )
    return value


__all__ = ["resolve_string_mapping"]
