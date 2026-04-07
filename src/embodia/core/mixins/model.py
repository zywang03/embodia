"""Model-facing mixin for embodia runtime integration."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any, Self

from ..config_io import (
    expand_component_yaml_interface_config,
    load_component_yaml_config,
)
from ...runtime.checks import (
    ensure_action_matches_model_spec as _ensure_action_matches_model_spec,
    validate_model_spec as _validate_model_spec,
)
from ..errors import InterfaceValidationError
from ..modalities import action_modes, images, meta, state, task
from ..modalities._common import CONTROL_TARGETS
from ..schema import Action, Frame, ModelSpec
from ..transform import invert_mapping, model_spec_to_dict, remap_frame, remap_model_spec
from .common import _CommonInterfaceMixin


class ModelMixin(_CommonInterfaceMixin):
    """Mixin that wraps a model implementation with embodia behavior."""

    MODEL_SPEC: ModelSpec | Mapping[str, Any] | None = None
    GET_SPEC_METHOD = "get_spec"
    RESET_METHOD = "reset"
    STEP_METHOD = "step"
    STEP_CHUNK_METHOD: str | None = None
    PLAN_METHOD: str | None = None

    _METHOD_ALIAS_ATTRS: dict[str, str] = {
        "get_spec": "GET_SPEC_METHOD",
        "reset": "RESET_METHOD",
        "step": "STEP_METHOD",
        "step_chunk": "STEP_CHUNK_METHOD",
        "plan": "PLAN_METHOD",
    }
    _YAML_CONFIG_KEYS = {
        "interface",
        "method_aliases",
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Catch the common MRO mistake early."""

        super().__init_subclass__(**kwargs)
        direct_bases = cls.__bases__
        if ModelMixin in direct_bases and direct_bases[0] is not ModelMixin:
            raise TypeError(
                f"{cls.__name__} must list ModelMixin as its first base class, "
                f"for example: class {cls.__name__}(ModelMixin, VendorModel): ..."
            )

    @classmethod
    def from_config(
        cls,
        *args: Any,
        model_spec: ModelSpec | Mapping[str, Any] | None = None,
        method_aliases: Mapping[str, str] | None = None,
        modality_maps: Mapping[object, Mapping[str, str]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Instantiate one model from validated embodia runtime config."""

        cls._validate_declared_runtime_interface_config()
        validated_method_aliases = cls._validate_method_aliases_config(method_aliases)
        validated_modality_maps = cls._validate_modality_maps_config(modality_maps)
        validated_model_spec = (
            cls._validate_model_spec_config(
                model_spec,
                modality_maps=validated_modality_maps,
            )
            if model_spec is not None
            else None
        )

        model = cls(*args, **kwargs)
        model._configure_runtime_interface(
            method_aliases=validated_method_aliases,
            modality_maps=validated_modality_maps,
        )
        if validated_model_spec is not None:
            model._embodia_runtime_model_spec = validated_model_spec
        return model

    @classmethod
    def from_yaml(
        cls,
        path: str | PathLike[str],
        **overrides: Any,
    ) -> Self:
        """Instantiate one model from a YAML config file.

        The file may either contain a top-level ``model:`` section or be a
        direct model-only config mapping.
        """

        loaded = load_component_yaml_config(path, component="model")
        cls._validate_yaml_section_keys(
            loaded,
            allowed_keys=cls._YAML_CONFIG_KEYS,
            config_label=f"model YAML config at {path}",
        )
        loaded = expand_component_yaml_interface_config(
            loaded,
            component="model",
            path=path,
        )
        loaded.update(overrides)
        return cls.from_config(**loaded)

    @classmethod
    def _validate_model_spec_config(
        cls,
        spec: ModelSpec | Mapping[str, Any],
        *,
        modality_maps: Mapping[str, Mapping[str, str]] | None = None,
    ) -> ModelSpec:
        """Validate one model spec before instance construction."""

        normalized = remap_model_spec(
            spec,
            image_key_map=cls._effective_modality_map(
                images.IMAGE_KEYS,
                modality_maps=modality_maps,
            ),
            target_map=cls._effective_modality_map(
                CONTROL_TARGETS,
                modality_maps=modality_maps,
            ),
            state_key_map=cls._effective_modality_map(
                state.STATE_KEYS,
                modality_maps=modality_maps,
            ),
            task_key_map=cls._effective_modality_map(
                task.TASK_KEYS,
                modality_maps=modality_maps,
            ),
            action_mode_map=cls._effective_modality_map(
                action_modes.ACTION_MODES,
                modality_maps=modality_maps,
            ),
        )
        _validate_model_spec(normalized)
        return normalized

    def normalize_spec(self, spec: ModelSpec | Mapping[str, Any]) -> ModelSpec:
        """Transform a model spec-like value into :class:`ModelSpec`."""

        return remap_model_spec(
            spec,
            image_key_map=self.get_image_key_map(),
            target_map=self.get_control_target_map(),
            state_key_map=self.get_state_key_map(),
            task_key_map=self.get_task_key_map(),
            action_mode_map=self.get_action_mode_map(),
        )

    def transform_spec(self, spec: ModelSpec | Mapping[str, Any]) -> ModelSpec:
        """Alias for :meth:`normalize_spec`."""

        return self.normalize_spec(spec)

    def validate_spec(self, spec: ModelSpec | Mapping[str, Any]) -> ModelSpec:
        """Normalize and validate a model spec-like value."""

        normalized = self.normalize_spec(spec)
        _validate_model_spec(normalized)
        return normalized

    def spec_to_dict(self, spec: ModelSpec | Mapping[str, Any]) -> dict[str, Any]:
        """Export a model spec-like value into a plain dictionary."""

        return model_spec_to_dict(spec)

    def _get_runtime_spec(self) -> ModelSpec:
        """Return the model spec with caching for declarative class specs."""

        instance_cached = getattr(self, "_embodia_runtime_model_spec", None)
        if instance_cached is not None:
            return instance_cached

        if type(self).MODEL_SPEC is None:
            raw_get_spec = self._resolve_impl("get_spec", "_get_spec_impl")
            return self.validate_spec(raw_get_spec())

        cached = self._get_cached_class_value("_EMBODIA_CACHED_MODEL_SPEC")
        if cached is not None:
            return cached
        return self._set_cached_class_value(
            "_EMBODIA_CACHED_MODEL_SPEC",
            self.validate_spec(type(self).MODEL_SPEC),
        )

    def ensure_frame_satisfies_spec(
        self,
        frame: Frame | Mapping[str, Any],
        spec: ModelSpec | Mapping[str, Any] | None = None,
    ) -> Frame:
        """Ensure frame keys satisfy the model spec."""

        normalized_frame = self.validate_frame(frame)
        normalized_spec = self.get_spec() if spec is None else self.validate_spec(spec)
        images.ensure_frame_keys(
            normalized_frame,
            owner_label="model",
            required_keys=normalized_spec.required_image_keys,
        )
        state.ensure_frame_keys(
            normalized_frame,
            owner_label="model",
            required_keys=normalized_spec.required_state_keys,
        )
        task.ensure_frame_keys(
            normalized_frame,
            owner_label="model",
            required_keys=normalized_spec.required_task_keys,
        )
        return normalized_frame

    def ensure_output_matches_spec(
        self,
        action: Action | Mapping[str, Any],
        spec: ModelSpec | Mapping[str, Any] | None = None,
    ) -> Action:
        """Ensure action mode matches the model spec."""

        normalized_action = self.validate_action(action)
        normalized_spec = self.get_spec() if spec is None else self.validate_spec(spec)
        _ensure_action_matches_model_spec(normalized_action, normalized_spec)
        return normalized_action

    def to_native_frame(self, frame: Frame) -> Any:
        """Convert a normalized frame into the wrapped implementation's format."""

        return remap_frame(
            frame,
            image_key_map=invert_mapping(
                self.get_image_key_map(),
                "ModelMixin image key mapping",
            ),
            state_key_map=invert_mapping(
                self.get_state_key_map(),
                "ModelMixin state key mapping",
            ),
            task_key_map=invert_mapping(
                self.get_task_key_map(),
                "ModelMixin task key mapping",
            ),
            meta_key_map=invert_mapping(
                self.get_meta_key_map(),
                "ModelMixin meta key mapping",
            ),
        )

    def get_spec(self) -> ModelSpec:
        """Return a normalized, validated model spec from the wrapped class."""

        return self._get_runtime_spec()

    def reset(self) -> None:
        """Forward reset and enforce the embodia return contract."""

        raw_reset = self._resolve_impl("reset", "_reset_impl")
        result = raw_reset()
        if result is not None:
            raise InterfaceValidationError(
                f"model reset() must return None, got {type(result).__name__}."
            )

    def step(self, frame: Frame | Mapping[str, Any]) -> Action:
        """Normalize inputs and outputs around the wrapped model step()."""

        raw_step = self._resolve_impl("step", "_step_impl")
        normalized_frame = self.validate_frame(frame)
        spec = self._get_runtime_spec()
        images.ensure_frame_keys(
            normalized_frame,
            owner_label="model",
            required_keys=spec.required_image_keys,
        )
        state.ensure_frame_keys(
            normalized_frame,
            owner_label="model",
            required_keys=spec.required_state_keys,
        )
        task.ensure_frame_keys(
            normalized_frame,
            owner_label="model",
            required_keys=spec.required_task_keys,
        )
        raw_action = raw_step(self.to_native_frame(normalized_frame))
        normalized_action = self.validate_action(raw_action)
        _ensure_action_matches_model_spec(normalized_action, spec)
        return normalized_action

__all__ = ["ModelMixin"]
