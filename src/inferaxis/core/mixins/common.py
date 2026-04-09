"""Shared frame/action normalization helpers for inferaxis mixins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...runtime.checks import (
    validate_action as _validate_action,
    validate_frame as _validate_frame,
)
from ..modalities._common import ModalityToken
from ..schema import Action, Frame
from ..transform import action_to_dict, frame_to_dict, remap_action, remap_frame
from .shared import config as config_utils
from .shared import resolution as resolution_utils


class _CommonInterfaceMixin:
    """Shared transform, config, and resolution helpers for inferaxis mixins."""

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

        return config_utils.copy_string_mapping(
            value,
            field_name=field_name,
            allowed_keys=allowed_keys,
        )

    def _copy_string_mapping(
        self,
        value: Mapping[str, str],
        *,
        field_name: str,
    ) -> dict[str, str]:
        """Validate and copy one ``mapping[str, str]`` configuration block."""

        return config_utils.copy_string_mapping(value, field_name=field_name)

    @classmethod
    def _normalize_modality_name_cls(cls, key: ModalityToken | str) -> str:
        """Return the stable string name for one modality key."""

        del cls
        return config_utils.normalize_modality_name(key)

    def _normalize_modality_name(self, key: ModalityToken | str) -> str:
        """Return the stable string name for one modality key."""

        return config_utils.normalize_modality_name(key)

    @classmethod
    def _copy_modality_maps_cls(
        cls,
        value: Mapping[ModalityToken | str, Mapping[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Validate and normalize runtime modality remapping config."""

        del cls
        return config_utils.copy_modality_maps(value)

    def _copy_modality_maps(
        self,
        value: Mapping[ModalityToken | str, Mapping[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Validate and normalize runtime modality remapping config."""

        return config_utils.copy_modality_maps(value)

    @classmethod
    def _validate_method_aliases_config(
        cls,
        value: Mapping[str, str] | None,
    ) -> dict[str, str] | None:
        """Validate one instance-level method alias config mapping."""

        return config_utils.validate_method_aliases_config(cls, value)

    @classmethod
    def _validate_modality_maps_config(
        cls,
        value: Mapping[ModalityToken | str, Mapping[str, str]] | None,
    ) -> dict[str, dict[str, str]] | None:
        """Validate one instance-level modality remapping config mapping."""

        return config_utils.validate_modality_maps_config(value)

    @classmethod
    def _validate_declared_runtime_interface_config(cls) -> None:
        """Validate class-level mixin config before instance construction."""

        config_utils.validate_declared_runtime_interface_config(cls)

    @classmethod
    def _effective_modality_map(
        cls,
        modality: ModalityToken | str,
        *,
        modality_maps: Mapping[str, Mapping[str, str]] | None = None,
    ) -> Mapping[str, str]:
        """Return the active modality mapping without constructing an instance."""

        return config_utils.effective_modality_map(
            cls,
            modality,
            modality_maps=modality_maps,
        )

    @classmethod
    def _effective_command_kind_map(
        cls,
        *,
        modality_maps: Mapping[str, Mapping[str, str]] | None = None,
    ) -> Mapping[str, str]:
        """Resolve the active command-kind mapping."""

        return config_utils.effective_command_kind_map(
            cls,
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

        del cls
        config_utils.validate_yaml_section_keys(
            loaded,
            allowed_keys=allowed_keys,
            config_label=config_label,
        )

    def _configure_runtime_interface(
        self,
        *,
        method_aliases: Mapping[str, str] | None = None,
        modality_maps: Mapping[ModalityToken | str, Mapping[str, str]] | None = None,
    ) -> None:
        """Attach validated instance-level interface config."""

        config_utils.configure_runtime_interface(
            self,
            method_aliases=method_aliases,
            modality_maps=modality_maps,
        )

    def _get_cached_class_value(self, cache_name: str) -> Any | None:
        """Return a class-level cached value when available."""

        return resolution_utils.get_cached_class_value(self, cache_name)

    def _set_cached_class_value(self, cache_name: str, value: Any) -> Any:
        """Store one class-level cached value and return it."""

        return resolution_utils.set_cached_class_value(self, cache_name, value)

    def _resolve_impl(self, public_name: str, fallback_name: str) -> Any:
        """Resolve the wrapped implementation for a public inferaxis method."""

        return resolution_utils.resolve_impl(
            self,
            public_name,
            fallback_name,
            super_anchor=_CommonInterfaceMixin,
            method_alias_attrs=type(self)._METHOD_ALIAS_ATTRS,
        )

    def _resolve_optional_impl(self, public_name: str, fallback_name: str) -> Any | None:
        """Resolve an optional wrapped implementation when present."""

        return resolution_utils.resolve_optional_impl(
            self,
            public_name,
            fallback_name,
            super_anchor=_CommonInterfaceMixin,
            method_alias_attrs=type(self)._METHOD_ALIAS_ATTRS,
        )

    def _resolve_mapping(self, attr_name: str) -> Mapping[str, str]:
        """Resolve a declarative mapping attribute."""

        return resolution_utils.resolve_mapping(self, attr_name)

    def get_modality_map(self, modality: ModalityToken | str) -> Mapping[str, str]:
        """Resolve one declared modality remapping table."""

        return resolution_utils.get_modality_map(self, modality)

    def _resolve_local_method(self, method_name: str) -> Any | None:
        """Resolve a method defined directly on the integration class."""

        return resolution_utils.resolve_local_method(self, method_name)

    def _resolve_method_alias(
        self,
        public_name: str,
        attr_name: str | None = None,
    ) -> str:
        """Resolve an alternative wrapped method name."""

        return resolution_utils.resolve_method_alias(
            self,
            public_name,
            attr_name=attr_name,
        )

    def get_image_key_map(self) -> Mapping[str, str]:
        """Map native image keys to inferaxis-standard image keys."""

        return resolution_utils.get_image_key_map(self)

    def get_state_key_map(self) -> Mapping[str, str]:
        """Map native state keys to inferaxis-standard state keys."""

        return resolution_utils.get_state_key_map(self)

    def get_control_target_map(self) -> Mapping[str, str]:
        """Map native component names to inferaxis-standard target names."""

        return resolution_utils.get_control_target_map(self)

    def get_command_kind_map(self) -> Mapping[str, str]:
        """Map native command names to inferaxis-standard command names."""

        return resolution_utils.get_command_kind_map(self)

    def get_task_key_map(self) -> Mapping[str, str]:
        """Map native task keys to inferaxis-standard task keys."""

        return resolution_utils.get_task_key_map(self)

    def get_meta_key_map(self) -> Mapping[str, str]:
        """Map native meta keys to inferaxis-standard meta keys."""

        return resolution_utils.get_meta_key_map(self)

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
            command_map=self.get_command_kind_map(),
        )

    def transform_action(self, action: Action | Mapping[str, Any]) -> Action:
        """Alias for :meth:`normalize_action`."""

        return self.normalize_action(action)

    def validate_action(self, action: Action | Mapping[str, Any]) -> Action:
        """Normalize and validate an action-like value."""

        normalized = self.normalize_action(action)
        _validate_action(normalized)
        return normalized

    def action_to_dict(
        self,
        action: Action | Mapping[str, Any],
        *,
        compact: bool = True,
        commands_as_mapping: bool = True,
    ) -> dict[str, Any]:
        """Export an action-like value into a plain dictionary."""

        return action_to_dict(
            action,
            compact=compact,
            commands_as_mapping=commands_as_mapping,
        )


__all__ = ["_CommonInterfaceMixin"]
