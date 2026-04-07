"""Shared mixin helpers for embodia robot/model integrations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...runtime.checks import (
    validate_action as _validate_action,
    validate_frame as _validate_frame,
)
from ..errors import InterfaceValidationError
from ..modalities import images, meta, state, task
from ..modalities._common import (
    COMMAND_KINDS,
    CONTROL_TARGETS,
    KNOWN_MODALITIES,
    ModalityToken,
    resolve_modality_mapping,
    resolve_string_mapping,
)
from ..schema import Action, Frame
from ..transform import action_to_dict, frame_to_dict, remap_action, remap_frame


class _CommonInterfaceMixin:
    """Shared transform and validation helpers."""

    MODALITY_MAPS: Mapping[ModalityToken | str, Mapping[str, str]] = {}
    METHOD_ALIASES: Mapping[str, str] = {}
    _METHOD_ALIAS_ATTRS: Mapping[str, str] = {}

    @classmethod
    def _copy_string_mapping_cls(
        cls,
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

    def _copy_string_mapping(
        self,
        value: Mapping[str, str],
        *,
        field_name: str,
    ) -> dict[str, str]:
        """Validate and copy one ``mapping[str, str]`` configuration block."""

        return type(self)._copy_string_mapping_cls(value, field_name=field_name)

    @classmethod
    def _normalize_modality_name_cls(cls, key: ModalityToken | str) -> str:
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

    def _normalize_modality_name(self, key: ModalityToken | str) -> str:
        """Return the stable string name for one modality key."""

        return type(self)._normalize_modality_name_cls(key)

    @classmethod
    def _copy_modality_maps_cls(
        cls,
        value: Mapping[ModalityToken | str, Mapping[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Validate and normalize runtime modality remapping config."""

        if not isinstance(value, Mapping):
            raise InterfaceValidationError(
                f"modality_maps must be a mapping, got {type(value).__name__}."
            )

        result: dict[str, dict[str, str]] = {}
        for key, item in value.items():
            normalized_name = cls._normalize_modality_name_cls(key)
            result[normalized_name] = cls._copy_string_mapping_cls(
                item,
                field_name=f"modality_maps[{normalized_name!r}]",
            )
        return result

    def _copy_modality_maps(
        self,
        value: Mapping[ModalityToken | str, Mapping[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Validate and normalize runtime modality remapping config."""

        return type(self)._copy_modality_maps_cls(value)

    @classmethod
    def _validate_method_aliases_config(
        cls,
        value: Mapping[str, str] | None,
    ) -> dict[str, str] | None:
        """Validate one instance-level method alias config mapping."""

        if value is None:
            return None
        return cls._copy_string_mapping_cls(
            value,
            field_name="method_aliases",
            allowed_keys=set(cls._METHOD_ALIAS_ATTRS),
        )

    @classmethod
    def _validate_modality_maps_config(
        cls,
        value: Mapping[ModalityToken | str, Mapping[str, str]] | None,
    ) -> dict[str, dict[str, str]] | None:
        """Validate one instance-level modality remapping config mapping."""

        if value is None:
            return None
        return cls._copy_modality_maps_cls(value)

    @classmethod
    def _validate_declared_runtime_interface_config(cls) -> None:
        """Validate class-level mixin config before instance construction."""

        try:
            cls._copy_string_mapping_cls(
                resolve_string_mapping(cls, "METHOD_ALIASES"),
                field_name=f"{cls.__name__}.METHOD_ALIASES",
                allowed_keys=set(cls._METHOD_ALIAS_ATTRS),
            )
            declared_modality_maps = getattr(cls, "MODALITY_MAPS", {})
            if declared_modality_maps is not None:
                cls._copy_modality_maps_cls(declared_modality_maps)
            for token in KNOWN_MODALITIES:
                legacy_attr = token.legacy_attr
                if legacy_attr is None or not hasattr(cls, legacy_attr):
                    continue
                cls._copy_string_mapping_cls(
                    getattr(cls, legacy_attr),
                    field_name=f"{cls.__name__}.{legacy_attr}",
                )
        except TypeError as exc:
            raise InterfaceValidationError(str(exc)) from exc

    @classmethod
    def _effective_modality_map(
        cls,
        modality: ModalityToken | str,
        *,
        modality_maps: Mapping[str, Mapping[str, str]] | None = None,
    ) -> Mapping[str, str]:
        """Return the active modality mapping without constructing an instance."""

        normalized_name = cls._normalize_modality_name_cls(modality)
        if modality_maps is not None and normalized_name in modality_maps:
            return modality_maps[normalized_name]
        try:
            return resolve_modality_mapping(cls, normalized_name)
        except TypeError as exc:
            raise InterfaceValidationError(str(exc)) from exc

    @classmethod
    def _effective_command_kind_map(
        cls,
        *,
        modality_maps: Mapping[str, Mapping[str, str]] | None = None,
    ) -> Mapping[str, str]:
        """Resolve the active command-kind mapping."""

        return cls._effective_modality_map(
            COMMAND_KINDS,
            modality_maps=modality_maps,
        )

    @classmethod
    def _validate_yaml_section_keys(
        cls,
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

    def _configure_runtime_interface(
        self,
        *,
        method_aliases: Mapping[str, str] | None = None,
        modality_maps: Mapping[ModalityToken | str, Mapping[str, str]] | None = None,
    ) -> None:
        """Attach validated instance-level interface config."""

        if method_aliases is not None:
            self._embodia_method_aliases = self._copy_string_mapping(
                method_aliases,
                field_name="method_aliases",
            )
        if modality_maps is not None:
            self._embodia_modality_maps = self._copy_modality_maps(modality_maps)

    def _get_cached_class_value(self, cache_name: str) -> Any | None:
        """Return a class-level cached value when available."""

        return type(self).__dict__.get(cache_name)

    def _set_cached_class_value(self, cache_name: str, value: Any) -> Any:
        """Store one class-level cached value and return it."""

        setattr(type(self), cache_name, value)
        return value

    def _resolve_impl(self, public_name: str, fallback_name: str) -> Any:
        """Resolve the wrapped implementation for a public embodia method."""

        alias_name = self._resolve_method_alias(
            public_name,
            type(self)._METHOD_ALIAS_ATTRS.get(public_name),
        )

        if alias_name != public_name:
            local_aliased = self._resolve_local_method(alias_name)
            if callable(local_aliased):
                return local_aliased

            inherited_aliased = getattr(super(), alias_name, None)
            if callable(inherited_aliased):
                return inherited_aliased

        inherited = getattr(super(), public_name, None)
        if callable(inherited):
            return inherited

        fallback = getattr(self, fallback_name, None)
        if callable(fallback):
            return fallback

        raise AttributeError(
            f"{type(self).__name__} requires either {public_name}() from another "
            f"base class or a local {fallback_name}() implementation."
        )

    def _resolve_optional_impl(self, public_name: str, fallback_name: str) -> Any | None:
        """Resolve an optional wrapped implementation when present.

        Unlike :meth:`_resolve_impl`, this returns ``None`` when neither an
        aliased method, inherited public method, nor local fallback exists.
        If an alias was explicitly configured but cannot be resolved, it raises
        ``AttributeError`` so configuration mistakes still fail loudly.
        """

        alias_name = self._resolve_method_alias(
            public_name,
            type(self)._METHOD_ALIAS_ATTRS.get(public_name),
        )

        if alias_name != public_name:
            local_aliased = self._resolve_local_method(alias_name)
            if callable(local_aliased):
                return local_aliased

            inherited_aliased = getattr(super(), alias_name, None)
            if callable(inherited_aliased):
                return inherited_aliased

            raise AttributeError(
                f"{type(self).__name__} configured optional alias {public_name!r} "
                f"-> {alias_name!r}, but {alias_name}() was not found."
            )

        inherited = getattr(super(), public_name, None)
        if callable(inherited):
            return inherited

        fallback = getattr(self, fallback_name, None)
        if callable(fallback):
            return fallback

        return None

    def _resolve_mapping(self, attr_name: str) -> Mapping[str, str]:
        """Resolve a declarative mapping attribute."""

        if attr_name == "METHOD_ALIASES":
            instance_value = getattr(self, "_embodia_method_aliases", None)
            if instance_value is not None:
                return instance_value
        return resolve_string_mapping(self, attr_name)

    def get_modality_map(self, modality: ModalityToken | str) -> Mapping[str, str]:
        """Resolve one declared modality remapping table."""

        instance_value = getattr(self, "_embodia_modality_maps", None)
        if instance_value is not None:
            resolved = instance_value.get(self._normalize_modality_name(modality))
            if resolved is not None:
                return resolved
        return resolve_modality_mapping(self, modality)

    def _resolve_local_method(self, method_name: str) -> Any | None:
        """Resolve a method defined directly on the integration class."""

        descriptor = type(self).__dict__.get(method_name)
        if descriptor is None:
            return None

        candidate = (
            descriptor.__get__(self, type(self))
            if hasattr(descriptor, "__get__")
            else descriptor
        )
        if not callable(candidate):
            raise TypeError(
                f"{type(self).__name__}.{method_name} exists but is not callable."
            )
        return candidate

    def _resolve_method_alias(
        self,
        public_name: str,
        attr_name: str | None = None,
    ) -> str:
        """Resolve an alternative wrapped method name."""

        alias_map = self._resolve_mapping("METHOD_ALIASES")
        alias_name = alias_map.get(public_name)
        if alias_name is None and attr_name is not None:
            alias_name = getattr(type(self), attr_name, public_name)

        if alias_name is None:
            return public_name
        if not isinstance(alias_name, str):
            raise TypeError(
                f"{type(self).__name__} alias for {public_name!r} must be a string, "
                f"got {type(alias_name).__name__}."
            )
        if not alias_name.strip():
            raise TypeError(
                f"{type(self).__name__} alias for {public_name!r} must not be empty."
            )
        return alias_name

    def get_image_key_map(self) -> Mapping[str, str]:
        """Map native image keys to embodia-standard image keys."""

        return self.get_modality_map(images.IMAGE_KEYS)

    def get_state_key_map(self) -> Mapping[str, str]:
        """Map native state keys to embodia-standard state keys."""

        return self.get_modality_map(state.STATE_KEYS)

    def get_control_target_map(self) -> Mapping[str, str]:
        """Map native control-group names to embodia-standard target names."""

        return self.get_modality_map(CONTROL_TARGETS)

    def get_command_kind_map(self) -> Mapping[str, str]:
        """Map native command kinds to embodia-standard command kinds."""

        return self.get_modality_map(COMMAND_KINDS)

    def get_task_key_map(self) -> Mapping[str, str]:
        """Map native task keys to embodia-standard task keys."""

        return self.get_modality_map(task.TASK_KEYS)

    def get_meta_key_map(self) -> Mapping[str, str]:
        """Map native meta keys to embodia-standard meta keys."""

        return self.get_modality_map(meta.META_KEYS)

    def normalize_frame(self, frame: Frame | Mapping[str, Any]) -> Frame:
        """Transform a frame-like value into :class:`Frame`."""

        return remap_frame(
            frame,
            image_key_map=self.get_image_key_map(),
            state_key_map=self.get_state_key_map(),
            task_key_map=self.get_task_key_map(),
            meta_key_map=self.get_meta_key_map(),
        )

    def transform_frame(self, frame: Frame | Mapping[str, Any]) -> Frame:
        """Alias for :meth:`normalize_frame`."""

        return self.normalize_frame(frame)

    def validate_frame(self, frame: Frame | Mapping[str, Any]) -> Frame:
        """Normalize and validate a frame-like value."""

        normalized = self.normalize_frame(frame)
        _validate_frame(normalized)
        return normalized

    def frame_to_dict(self, frame: Frame | Mapping[str, Any]) -> dict[str, Any]:
        """Export a frame-like value into a plain dictionary."""

        return frame_to_dict(frame)

    def normalize_action(self, action: Action | Mapping[str, Any]) -> Action:
        """Transform an action-like value into :class:`Action`."""

        return remap_action(
            action,
            target_map=self.get_control_target_map(),
            kind_map=self.get_command_kind_map(),
        )

    def transform_action(self, action: Action | Mapping[str, Any]) -> Action:
        """Alias for :meth:`normalize_action`."""

        return self.normalize_action(action)

    def validate_action(self, action: Action | Mapping[str, Any]) -> Action:
        """Normalize and validate an action-like value."""

        normalized = self.normalize_action(action)
        _validate_action(normalized)
        return normalized

    def action_to_dict(self, action: Action | Mapping[str, Any]) -> dict[str, Any]:
        """Export an action-like value into a plain dictionary."""

        return action_to_dict(action)


__all__ = ["_CommonInterfaceMixin"]
