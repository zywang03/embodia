"""Private config-validation helpers shared by embodia mixins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import InterfaceValidationError
from ..modalities._common import (
    COMMAND_KINDS,
    KNOWN_MODALITIES,
    ModalityToken,
    resolve_modality_mapping,
    resolve_string_mapping,
)


def copy_string_mapping(
    value: Mapping[str, str],
    *,
    field_name: str,
    allowed_keys: set[str] | None = None,
) -> dict[str, str]:
    """Validate and copy one ``mapping[str, str]`` configuration block."""

    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"{field_name} must be a mapping[str, str], got "
            f"{type(value).__name__}."
        )

    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise InterfaceValidationError(
                f"{field_name} must be a mapping[str, str], got key "
                f"{key!r} and value {item!r}."
            )
        if not key.strip() or not item.strip():
            raise InterfaceValidationError(
                f"{field_name} keys and values must be non-empty strings."
            )
        if allowed_keys is not None and key not in allowed_keys:
            expected = ", ".join(repr(name) for name in sorted(allowed_keys))
            raise InterfaceValidationError(
                f"{field_name} contains unsupported key {key!r}. "
                f"Expected one of: {expected}."
            )
        result[key] = item
    return result


def normalize_modality_name(key: ModalityToken | str) -> str:
    """Return the stable string name for one modality key."""

    if isinstance(key, ModalityToken):
        name = key.name
    elif isinstance(key, str):
        if not key.strip():
            raise InterfaceValidationError(
                "modality_maps keys must be non-empty strings."
            )
        name = key
    else:
        raise InterfaceValidationError(
            "modality_maps keys must be strings or embodia modality tokens, "
            f"got {type(key).__name__}."
        )

    known_names = {token.name for token in KNOWN_MODALITIES}
    if name not in known_names:
        expected = ", ".join(repr(token.name) for token in KNOWN_MODALITIES)
        raise InterfaceValidationError(
            f"modality_maps contains unsupported modality {name!r}. "
            f"Expected one of: {expected}."
        )
    return name


def copy_modality_maps(
    value: Mapping[ModalityToken | str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Validate and normalize runtime modality remapping config."""

    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"modality_maps must be a mapping, got {type(value).__name__}."
        )

    result: dict[str, dict[str, str]] = {}
    for key, item in value.items():
        normalized_name = normalize_modality_name(key)
        result[normalized_name] = copy_string_mapping(
            item,
            field_name=f"modality_maps[{normalized_name!r}]",
        )
    return result


def validate_method_aliases_config(
    cls: type[object],
    value: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Validate one instance-level method alias config mapping."""

    if value is None:
        return None
    return copy_string_mapping(
        value,
        field_name="method_aliases",
        allowed_keys=set(getattr(cls, "_METHOD_ALIAS_ATTRS", {})),
    )


def validate_modality_maps_config(
    value: Mapping[ModalityToken | str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]] | None:
    """Validate one instance-level modality remapping config mapping."""

    if value is None:
        return None
    return copy_modality_maps(value)


def validate_declared_runtime_interface_config(cls: type[object]) -> None:
    """Validate class-level mixin config before instance construction."""

    try:
        copy_string_mapping(
            resolve_string_mapping(cls, "METHOD_ALIASES"),
            field_name=f"{cls.__name__}.METHOD_ALIASES",
            allowed_keys=set(getattr(cls, "_METHOD_ALIAS_ATTRS", {})),
        )
        declared_modality_maps = getattr(cls, "MODALITY_MAPS", {})
        if declared_modality_maps is not None:
            copy_modality_maps(declared_modality_maps)
        for token in KNOWN_MODALITIES:
            legacy_attr = token.legacy_attr
            if legacy_attr is None or not hasattr(cls, legacy_attr):
                continue
            copy_string_mapping(
                getattr(cls, legacy_attr),
                field_name=f"{cls.__name__}.{legacy_attr}",
            )
    except TypeError as exc:
        raise InterfaceValidationError(str(exc)) from exc


def effective_modality_map(
    cls: type[object],
    modality: ModalityToken | str,
    *,
    modality_maps: Mapping[str, Mapping[str, str]] | None = None,
) -> Mapping[str, str]:
    """Return one active modality mapping without constructing an instance."""

    normalized_name = normalize_modality_name(modality)
    if modality_maps is not None and normalized_name in modality_maps:
        return modality_maps[normalized_name]
    try:
        return resolve_modality_mapping(cls, normalized_name)
    except TypeError as exc:
        raise InterfaceValidationError(str(exc)) from exc


def effective_command_kind_map(
    cls: type[object],
    *,
    modality_maps: Mapping[str, Mapping[str, str]] | None = None,
) -> Mapping[str, str]:
    """Resolve the active command-kind mapping."""

    return effective_modality_map(
        cls,
        COMMAND_KINDS,
        modality_maps=modality_maps,
    )


def validate_yaml_section_keys(
    loaded: Mapping[str, Any],
    *,
    allowed_keys: set[str],
    config_label: str,
) -> None:
    """Validate top-level keys inside one loaded YAML config block."""

    unknown = sorted(key for key in loaded if key not in allowed_keys)
    if not unknown:
        return
    expected = ", ".join(repr(name) for name in sorted(allowed_keys))
    found = ", ".join(repr(name) for name in unknown)
    raise InterfaceValidationError(
        f"{config_label} contains unsupported field(s) {found}. "
        f"Expected only: {expected}."
    )


def configure_runtime_interface(
    owner: object,
    *,
    method_aliases: Mapping[str, str] | None = None,
    modality_maps: Mapping[ModalityToken | str, Mapping[str, str]] | None = None,
) -> None:
    """Attach validated instance-level interface config."""

    if method_aliases is not None:
        setattr(
            owner,
            "_embodia_method_aliases",
            copy_string_mapping(
                method_aliases,
                field_name="method_aliases",
            ),
        )
    if modality_maps is not None:
        setattr(owner, "_embodia_modality_maps", copy_modality_maps(modality_maps))


__all__ = [
    "configure_runtime_interface",
    "copy_modality_maps",
    "copy_string_mapping",
    "effective_command_kind_map",
    "effective_modality_map",
    "normalize_modality_name",
    "validate_declared_runtime_interface_config",
    "validate_method_aliases_config",
    "validate_modality_maps_config",
    "validate_yaml_section_keys",
]
