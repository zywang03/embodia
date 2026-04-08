"""Private method-resolution helpers shared by embodia mixins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...modalities import images, meta, state, task
from ...modalities._common import COMMAND_KINDS, CONTROL_TARGETS, ModalityToken
from ...modalities._common import resolve_modality_mapping, resolve_string_mapping
from .config import normalize_modality_name


def get_cached_class_value(owner: object, cache_name: str) -> Any | None:
    """Return a class-level cached value when available."""

    return type(owner).__dict__.get(cache_name)


def set_cached_class_value(owner: object, cache_name: str, value: Any) -> Any:
    """Store one class-level cached value and return it."""

    setattr(type(owner), cache_name, value)
    return value


def resolve_impl(
    owner: object,
    public_name: str,
    fallback_name: str,
    *,
    super_anchor: type[object],
    method_alias_attrs: Mapping[str, str],
) -> Any:
    """Resolve the wrapped implementation for one public embodia method."""

    alias_name = resolve_method_alias(
        owner,
        public_name,
        attr_name=method_alias_attrs.get(public_name),
    )

    if alias_name != public_name:
        local_aliased = resolve_local_method(owner, alias_name)
        if callable(local_aliased):
            return local_aliased

        inherited_aliased = getattr(super(super_anchor, owner), alias_name, None)
        if callable(inherited_aliased):
            return inherited_aliased

    local_public = resolve_local_method(owner, public_name)
    if callable(local_public):
        return local_public

    inherited = getattr(super(super_anchor, owner), public_name, None)
    if callable(inherited):
        return inherited

    fallback = getattr(owner, fallback_name, None)
    if callable(fallback):
        return fallback

    raise AttributeError(
        f"{type(owner).__name__} requires either {public_name}() from another "
        f"base class or a local {fallback_name}() implementation."
    )


def resolve_optional_impl(
    owner: object,
    public_name: str,
    fallback_name: str,
    *,
    super_anchor: type[object],
    method_alias_attrs: Mapping[str, str],
) -> Any | None:
    """Resolve an optional wrapped implementation when present."""

    alias_name = resolve_method_alias(
        owner,
        public_name,
        attr_name=method_alias_attrs.get(public_name),
    )

    if alias_name != public_name:
        local_aliased = resolve_local_method(owner, alias_name)
        if callable(local_aliased):
            return local_aliased

        inherited_aliased = getattr(super(super_anchor, owner), alias_name, None)
        if callable(inherited_aliased):
            return inherited_aliased

        raise AttributeError(
            f"{type(owner).__name__} configured optional alias {public_name!r} "
            f"-> {alias_name!r}, but {alias_name}() was not found."
        )

    local_public = resolve_local_method(owner, public_name)
    if callable(local_public):
        return local_public

    inherited = getattr(super(super_anchor, owner), public_name, None)
    if callable(inherited):
        return inherited

    fallback = getattr(owner, fallback_name, None)
    if callable(fallback):
        return fallback

    return None


def resolve_mapping(owner: object, attr_name: str) -> Mapping[str, str]:
    """Resolve one declarative mapping attribute."""

    if attr_name == "METHOD_ALIASES":
        instance_value = getattr(owner, "_embodia_method_aliases", None)
        if instance_value is not None:
            return instance_value
    return resolve_string_mapping(owner, attr_name)


def get_modality_map(owner: object, modality: ModalityToken | str) -> Mapping[str, str]:
    """Resolve one declared modality remapping table."""

    instance_value = getattr(owner, "_embodia_modality_maps", None)
    if instance_value is not None:
        resolved = instance_value.get(normalize_modality_name(modality))
        if resolved is not None:
            return resolved
    return resolve_modality_mapping(owner, modality)


def resolve_local_method(owner: object, method_name: str) -> Any | None:
    """Resolve a method defined directly on the integration class."""

    descriptor = type(owner).__dict__.get(method_name)
    if descriptor is None:
        return None

    candidate = (
        descriptor.__get__(owner, type(owner))
        if hasattr(descriptor, "__get__")
        else descriptor
    )
    if not callable(candidate):
        raise TypeError(
            f"{type(owner).__name__}.{method_name} exists but is not callable."
        )
    return candidate


def resolve_method_alias(
    owner: object,
    public_name: str,
    *,
    attr_name: str | None = None,
) -> str:
    """Resolve an alternative wrapped method name."""

    alias_map = resolve_mapping(owner, "METHOD_ALIASES")
    alias_name = alias_map.get(public_name)
    if alias_name is None and attr_name is not None:
        alias_name = getattr(type(owner), attr_name, public_name)

    if alias_name is None:
        return public_name
    if not isinstance(alias_name, str):
        raise TypeError(
            f"{type(owner).__name__} alias for {public_name!r} must be a string, "
            f"got {type(alias_name).__name__}."
        )
    if not alias_name.strip():
        raise TypeError(
            f"{type(owner).__name__} alias for {public_name!r} must not be empty."
        )
    return alias_name


def get_image_key_map(owner: object) -> Mapping[str, str]:
    """Map native image keys to embodia-standard image keys."""

    return get_modality_map(owner, images.IMAGE_KEYS)


def get_state_key_map(owner: object) -> Mapping[str, str]:
    """Map native state keys to embodia-standard state keys."""

    return get_modality_map(owner, state.STATE_KEYS)


def get_control_target_map(owner: object) -> Mapping[str, str]:
    """Map native component names to embodia-standard target names."""

    return get_modality_map(owner, CONTROL_TARGETS)


def get_command_kind_map(owner: object) -> Mapping[str, str]:
    """Map native command kinds to embodia-standard command kinds."""

    return get_modality_map(owner, COMMAND_KINDS)


def get_task_key_map(owner: object) -> Mapping[str, str]:
    """Map native task keys to embodia-standard task keys."""

    return get_modality_map(owner, task.TASK_KEYS)


def get_meta_key_map(owner: object) -> Mapping[str, str]:
    """Map native meta keys to embodia-standard meta keys."""

    return get_modality_map(owner, meta.META_KEYS)


__all__ = [
    "get_cached_class_value",
    "get_command_kind_map",
    "get_control_target_map",
    "get_image_key_map",
    "get_meta_key_map",
    "get_modality_map",
    "get_state_key_map",
    "get_task_key_map",
    "resolve_impl",
    "resolve_local_method",
    "resolve_mapping",
    "resolve_method_alias",
    "resolve_optional_impl",
    "set_cached_class_value",
]
