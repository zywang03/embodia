"""Shared helpers for modality-specific mapping configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


MODALITY_MAPS_ATTR = "MODALITY_MAPS"


@dataclass(frozen=True, slots=True)
class ModalityToken:
    """A stable key that identifies one embodia modality mapping."""

    name: str
    legacy_attr: str | None = None

    def __str__(self) -> str:
        return self.name


IMAGE_KEYS = ModalityToken(
    "images",
    legacy_attr="IMAGE_KEY_MAP",
)
STATE_KEYS = ModalityToken(
    "state",
    legacy_attr="STATE_KEY_MAP",
)
ACTION_MODES = ModalityToken(
    "action_modes",
    legacy_attr="ACTION_MODE_MAP",
)

KNOWN_MODALITIES = (IMAGE_KEYS, STATE_KEYS, ACTION_MODES)
_KNOWN_MODALITY_BY_NAME = {token.name: token for token in KNOWN_MODALITIES}


def _ensure_mapping(
    cls: type[object],
    value: object,
    field_name: str,
) -> Mapping[object, object]:
    """Validate that a value is mapping-like."""

    if not isinstance(value, Mapping):
        raise TypeError(
            f"{cls.__name__}.{field_name} must be a mapping, "
            f"got {type(value).__name__}."
        )
    return value


def _ensure_string_mapping(
    cls: type[object],
    value: object,
    field_name: str,
) -> Mapping[str, str]:
    """Validate a ``mapping[str, str]`` configuration block."""

    mapping = _ensure_mapping(cls, value, field_name)

    for key, item in mapping.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(
                f"{cls.__name__}.{field_name} must be a mapping[str, str], "
                f"got key {key!r} and value {item!r}."
            )
    return mapping


def resolve_string_mapping(
    owner: object,
    attr_name: str,
) -> Mapping[str, str]:
    """Resolve a generic ``mapping[str, str]`` class attribute."""

    cls = owner if isinstance(owner, type) else type(owner)
    return _ensure_string_mapping(cls, getattr(cls, attr_name, {}), attr_name)


def _modality_lookup_keys(modality: ModalityToken | str) -> tuple[object, ...]:
    """Return accepted lookup keys for one modality."""

    if isinstance(modality, ModalityToken):
        return (modality, modality.name)

    known = _KNOWN_MODALITY_BY_NAME.get(modality)
    if known is None:
        return (modality,)
    return (known, known.name)


def _modality_field_label(candidate: object) -> str:
    """Return a friendly field label for error messages."""

    if isinstance(candidate, ModalityToken):
        return repr(candidate.name)
    return repr(candidate)


def resolve_modality_mapping(
    owner: object,
    modality: ModalityToken | str,
) -> Mapping[str, str]:
    """Resolve one modality remapping table from ``MODALITY_MAPS``.

    Preferred public form:

    ``MODALITY_MAPS = {IMAGE_KEYS: {...}, STATE_KEYS: {...}, ACTION_MODES: {...}}``

    String keys such as ``"images"`` and legacy top-level attributes such as
    ``IMAGE_KEY_MAP`` remain supported for compatibility.
    """

    cls = owner if isinstance(owner, type) else type(owner)
    modality_maps = getattr(cls, MODALITY_MAPS_ATTR, None)
    if modality_maps is not None:
        validated_maps = _ensure_mapping(cls, modality_maps, MODALITY_MAPS_ATTR)
        for candidate in _modality_lookup_keys(modality):
            if candidate in validated_maps:
                return _ensure_string_mapping(
                    cls,
                    validated_maps[candidate],
                    f"{MODALITY_MAPS_ATTR}[{_modality_field_label(candidate)}]",
                )

    token = (
        modality
        if isinstance(modality, ModalityToken)
        else _KNOWN_MODALITY_BY_NAME.get(modality)
    )
    if token is not None and token.legacy_attr is not None:
        return resolve_string_mapping(cls, token.legacy_attr)

    return {}


__all__ = [
    "ACTION_MODES",
    "IMAGE_KEYS",
    "KNOWN_MODALITIES",
    "MODALITY_MAPS_ATTR",
    "ModalityToken",
    "STATE_KEYS",
    "resolve_modality_mapping",
    "resolve_string_mapping",
]
