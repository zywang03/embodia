"""Model-facing mixin for embodia runtime integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from typing import Any, Self

from ..config_io import load_component_yaml_config
from ...runtime.checks import validate_model_spec as _validate_model_spec
from ..errors import InterfaceValidationError
from ..modalities import action_modes, images, state
from ..schema import Action, Frame, ModelSpec
from ..transform import invert_mapping, model_spec_to_dict, remap_frame, remap_model_spec
from .common import _CommonInterfaceMixin


class ModelMixin(_CommonInterfaceMixin):
    """Mixin that wraps a model implementation with embodia behavior."""

    MODEL_SPEC: ModelSpec | Mapping[str, Any] | None = None
    GET_SPEC_METHOD = "get_spec"
    RESET_METHOD = "reset"
    STEP_METHOD = "step"

    _METHOD_ALIAS_ATTRS: dict[str, str] = {
        "get_spec": "GET_SPEC_METHOD",
        "reset": "RESET_METHOD",
        "step": "STEP_METHOD",
    }
    _YAML_CONFIG_KEYS = {
        "init",
        "model_spec",
        "method_aliases",
        "modality_maps",
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
        init_config = loaded.pop("init", {})
        if not isinstance(init_config, Mapping):
            raise InterfaceValidationError(
                "YAML field 'init' must be a mapping when provided."
            )

        merged: dict[str, Any] = dict(init_config)
        merged.update(loaded)
        merged.update(overrides)
        return cls.from_config(**merged)

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
            state_key_map=cls._effective_modality_map(
                state.STATE_KEYS,
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
            state_key_map=self.get_state_key_map(),
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
        return normalized_frame

    def ensure_output_matches_spec(
        self,
        action: Action | Mapping[str, Any],
        spec: ModelSpec | Mapping[str, Any] | None = None,
    ) -> Action:
        """Ensure action mode matches the model spec."""

        normalized_action = self.validate_action(action)
        normalized_spec = self.get_spec() if spec is None else self.validate_spec(spec)
        action_modes.ensure_model_output(
            normalized_action,
            output_mode=normalized_spec.output_action_mode,
            model_name=normalized_spec.name,
        )
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
        raw_action = raw_step(self.to_native_frame(normalized_frame))
        normalized_action = self.validate_action(raw_action)
        action_modes.ensure_model_output(
            normalized_action,
            output_mode=spec.output_action_mode,
            model_name=spec.name,
        )
        return normalized_action

    def build_openpi_policy_adapter(
        self,
        *,
        obs_to_frame: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]],
        action_plan_provider: Callable[[object, Frame], Any] | None = None,
        response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None,
        server_metadata: Mapping[str, Any] | None = None,
        reset_model_on_connect: bool = False,
    ) -> Any:
        """Build an OpenPI-compatible adapter around this model."""

        from ...contrib import openpi_remote as em_openpi_remote

        return em_openpi_remote.EmbodiaModelPolicyAdapter(
            self,
            obs_to_frame=obs_to_frame,
            action_plan_provider=action_plan_provider,
            response_builder=response_builder,
            server_metadata=server_metadata,
            reset_model_on_connect=reset_model_on_connect,
        )

    def build_openpi_policy_server(
        self,
        *,
        obs_to_frame: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]],
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_server_timing: bool = True,
        action_plan_provider: Callable[[object, Frame], Any] | None = None,
        response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None,
        server_metadata: Mapping[str, Any] | None = None,
        reset_model_on_connect: bool = False,
    ) -> Any:
        """Build an OpenPI-compatible websocket server around this model."""

        from ...contrib import openpi_remote as em_openpi_remote

        adapter = self.build_openpi_policy_adapter(
            obs_to_frame=obs_to_frame,
            action_plan_provider=action_plan_provider,
            response_builder=response_builder,
            server_metadata=server_metadata,
            reset_model_on_connect=reset_model_on_connect,
        )
        return em_openpi_remote.WebsocketPolicyServer(
            adapter,
            host=host,
            port=port,
            metadata=metadata if metadata is not None else adapter.get_server_metadata(),
            api_key=api_key,
            include_server_timing=include_server_timing,
        )

    def serve_openpi_policy(
        self,
        *,
        obs_to_frame: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]],
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_server_timing: bool = True,
        action_plan_provider: Callable[[object, Frame], Any] | None = None,
        response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None,
        server_metadata: Mapping[str, Any] | None = None,
        reset_model_on_connect: bool = False,
    ) -> None:
        """Serve this model through an OpenPI-compatible websocket API."""

        server = self.build_openpi_policy_server(
            obs_to_frame=obs_to_frame,
            host=host,
            port=port,
            api_key=api_key,
            metadata=metadata,
            include_server_timing=include_server_timing,
            action_plan_provider=action_plan_provider,
            response_builder=response_builder,
            server_metadata=server_metadata,
            reset_model_on_connect=reset_model_on_connect,
        )
        server.serve_forever()


__all__ = ["ModelMixin"]
