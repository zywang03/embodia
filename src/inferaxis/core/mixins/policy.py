"""Policy-facing mixin for inferaxis runtime integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from types import SimpleNamespace
from typing import Any, Self

from ..config_io import (
    expand_component_yaml_config,
    load_component_yaml_config,
)
from ...runtime.checks import (
    ensure_action_matches_policy_spec as _ensure_action_matches_policy_spec,
    validate_policy_spec as _validate_policy_spec,
)
from ..errors import InterfaceValidationError
from ..modalities import images, meta, state, task
from ..modalities._common import CONTROL_TARGETS
from ..schema import Action, Frame, PolicySpec
from ..transform import invert_mapping, policy_spec_to_dict, remap_frame, remap_policy_spec
from .common import _CommonInterfaceMixin


class PolicyMixin(_CommonInterfaceMixin):
    """Mixin that wraps a policy implementation with inferaxis behavior."""

    POLICY_SPEC: PolicySpec | Mapping[str, Any] | None = None
    GET_SPEC_METHOD = "get_spec"
    RESET_METHOD = "reset"
    INFER_METHOD = "infer"
    INFER_CHUNK_METHOD: str | None = "infer_chunk"
    PLAN_METHOD: str | None = "plan"

    _METHOD_ALIAS_ATTRS: dict[str, str] = {
        "get_spec": "GET_SPEC_METHOD",
        "reset": "RESET_METHOD",
        "infer": "INFER_METHOD",
        "infer_chunk": "INFER_CHUNK_METHOD",
        "plan": "PLAN_METHOD",
    }
    _YAML_CONFIG_KEYS = {
        "schema",
        "name",
        "method_aliases",
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Catch the common MRO mistake early."""

        super().__init_subclass__(**kwargs)
        direct_bases = cls.__bases__
        if PolicyMixin in direct_bases and direct_bases[0] is not PolicyMixin:
            raise TypeError(
                f"{cls.__name__} must list PolicyMixin as its first base class, "
                f"for example: class {cls.__name__}(PolicyMixin, VendorPolicy): ..."
            )

    @classmethod
    def from_config(
        cls,
        *args: Any,
        policy_spec: PolicySpec | Mapping[str, Any] | None = None,
        method_aliases: Mapping[str, str] | None = None,
        modality_maps: Mapping[object, Mapping[str, str]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Instantiate one policy from validated inferaxis runtime config."""

        cls._validate_declared_runtime_interface_config()
        validated_method_aliases = cls._validate_method_aliases_config(method_aliases)
        validated_modality_maps = cls._validate_modality_maps_config(modality_maps)
        validated_policy_spec = (
            cls._validate_policy_spec_config(
                policy_spec,
                modality_maps=validated_modality_maps,
            )
            if policy_spec is not None
            else None
        )

        policy = cls(*args, **kwargs)
        policy._configure_runtime_interface(
            method_aliases=validated_method_aliases,
            modality_maps=validated_modality_maps,
        )
        if validated_policy_spec is not None:
            policy._inferaxis_runtime_policy_spec = validated_policy_spec
        return policy

    @classmethod
    def from_yaml(
        cls,
        path: str | PathLike[str],
        **overrides: Any,
    ) -> Self:
        """Instantiate one policy from a YAML config file.

        The file may either contain a top-level ``policy:`` section or be a
        direct policy-only config mapping.
        """

        loaded = load_component_yaml_config(path, component="policy")
        cls._validate_yaml_section_keys(
            loaded,
            allowed_keys=cls._YAML_CONFIG_KEYS,
            config_label=f"policy YAML config at {path}",
        )
        loaded = expand_component_yaml_config(
            loaded,
            component="policy",
            path=path,
        )
        loaded.update(overrides)
        return cls.from_config(**loaded)

    @classmethod
    def _validate_policy_spec_config(
        cls,
        spec: PolicySpec | Mapping[str, Any],
        *,
        modality_maps: Mapping[str, Mapping[str, str]] | None = None,
    ) -> PolicySpec:
        """Validate one policy spec before instance construction."""

        normalized = remap_policy_spec(
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
            command_map=cls._effective_command_kind_map(
                modality_maps=modality_maps
            ),
        )
        _validate_policy_spec(normalized)
        return normalized

    def normalize_spec(self, spec: PolicySpec | Mapping[str, Any]) -> PolicySpec:
        """Transform a policy spec-like value into :class:`PolicySpec`."""

        return remap_policy_spec(
            spec,
            image_key_map=self.get_image_key_map(),
            target_map=self.get_control_target_map(),
            state_key_map=self.get_state_key_map(),
            task_key_map=self.get_task_key_map(),
            command_map=self.get_command_kind_map(),
        )

    def transform_spec(self, spec: PolicySpec | Mapping[str, Any]) -> PolicySpec:
        """Alias for :meth:`normalize_spec`."""

        return self.normalize_spec(spec)

    def validate_spec(self, spec: PolicySpec | Mapping[str, Any]) -> PolicySpec:
        """Normalize and validate a policy spec-like value."""

        normalized = self.normalize_spec(spec)
        _validate_policy_spec(normalized)
        return normalized

    def spec_to_dict(self, spec: PolicySpec | Mapping[str, Any]) -> dict[str, Any]:
        """Export a policy spec-like value into a plain dictionary."""

        return policy_spec_to_dict(spec)

    def _get_runtime_spec(self) -> PolicySpec:
        """Return the policy spec with caching for declarative class specs."""

        instance_cached = getattr(self, "_inferaxis_runtime_policy_spec", None)
        if instance_cached is not None:
            return instance_cached

        if type(self).POLICY_SPEC is None:
            raw_get_spec = self._resolve_impl("get_spec", "_get_spec_impl")
            return self.validate_spec(raw_get_spec())

        cached = self._get_cached_class_value("_EMBODIA_CACHED_POLICY_SPEC")
        if cached is not None:
            return cached
        return self._set_cached_class_value(
            "_EMBODIA_CACHED_POLICY_SPEC",
            self.validate_spec(type(self).POLICY_SPEC),
        )

    def ensure_frame_satisfies_spec(
        self,
        frame: Frame | Mapping[str, Any],
        spec: PolicySpec | Mapping[str, Any] | None = None,
    ) -> Frame:
        """Ensure frame keys satisfy the policy spec."""

        normalized_frame = self.validate_frame(frame)
        normalized_spec = (
            self.inferaxis_get_spec() if spec is None else self.validate_spec(spec)
        )
        images.ensure_frame_keys(
            normalized_frame,
            owner_label="policy",
            required_keys=normalized_spec.required_image_keys,
        )
        state.ensure_frame_keys(
            normalized_frame,
            owner_label="policy",
            required_keys=normalized_spec.required_state_keys,
        )
        task.ensure_frame_keys(
            normalized_frame,
            owner_label="policy",
            required_keys=normalized_spec.required_task_keys,
        )
        return normalized_frame

    def ensure_output_matches_spec(
        self,
        action: Action | Mapping[str, Any],
        spec: PolicySpec | Mapping[str, Any] | None = None,
    ) -> Action:
        """Ensure emitted commands and dims match the policy spec."""

        normalized_action = self.validate_action(action)
        normalized_spec = (
            self.inferaxis_get_spec() if spec is None else self.validate_spec(spec)
        )
        _ensure_action_matches_policy_spec(normalized_action, normalized_spec)
        return normalized_action

    def to_native_frame(self, frame: Frame) -> Any:
        """Convert a normalized frame into the wrapped implementation's format."""

        return remap_frame(
            frame,
            image_key_map=invert_mapping(
                self.get_image_key_map(),
                "PolicyMixin image key mapping",
            ),
            state_key_map=invert_mapping(
                self.get_state_key_map(),
                "PolicyMixin state key mapping",
            ),
            task_key_map=invert_mapping(
                self.get_task_key_map(),
                "PolicyMixin task key mapping",
            ),
            meta_key_map=invert_mapping(
                self.get_meta_key_map(),
                "PolicyMixin meta key mapping",
            ),
        )

    def inferaxis_get_spec(self) -> PolicySpec:
        """Return the normalized policy spec used internally by inferaxis."""

        return self._get_runtime_spec()

    def get_spec(self) -> PolicySpec:
        """Backward-compatible alias for :meth:`inferaxis_get_spec`."""

        return self.inferaxis_get_spec()

    def inferaxis_reset(self) -> None:
        """Forward reset and ignore the wrapped policy's return value."""

        raw_reset = self._resolve_impl("reset", "_reset_impl")
        raw_reset()

    def reset(self) -> None:
        """Backward-compatible alias for :meth:`inferaxis_reset`."""

        self.inferaxis_reset()

    @staticmethod
    def _single_step_chunk_request() -> object:
        """Build one minimal request object for chunk-to-step fallback."""

        return SimpleNamespace(
            request_step=0,
            request_time_s=0.0,
            history_start=0,
            history_end=0,
            active_chunk_length=0,
            remaining_steps=0,
            overlap_steps=0,
            latency_steps=0,
            request_trigger_steps=0,
            plan_start_step=0,
            history_actions=[],
        )

    def _validate_policy_input_frame(
        self,
        frame: Frame | Mapping[str, Any],
    ) -> tuple[Frame, PolicySpec]:
        """Normalize one policy input frame and ensure it satisfies the spec."""

        normalized_frame = self.validate_frame(frame)
        spec = self._get_runtime_spec()
        images.ensure_frame_keys(
            normalized_frame,
            owner_label="policy",
            required_keys=spec.required_image_keys,
        )
        state.ensure_frame_keys(
            normalized_frame,
            owner_label="policy",
            required_keys=spec.required_state_keys,
        )
        task.ensure_frame_keys(
            normalized_frame,
            owner_label="policy",
            required_keys=spec.required_task_keys,
        )
        return normalized_frame, spec

    def _coerce_action_plan(
        self,
        raw_plan: Action | Mapping[str, Any] | Sequence[Action | Mapping[str, Any]],
        *,
        spec: PolicySpec,
    ) -> list[Action]:
        """Normalize one step result or chunk result into validated actions."""

        if isinstance(raw_plan, (Action, Mapping)):
            items = [raw_plan]
        elif isinstance(raw_plan, (str, bytes)) or not isinstance(raw_plan, Sequence):
            raise InterfaceValidationError(
                "policy infer_chunk() must return one action-like object or a "
                f"sequence of action-like objects, got {type(raw_plan).__name__}."
            )
        else:
            items = list(raw_plan)

        if not items:
            raise InterfaceValidationError(
                "policy infer_chunk() must not return an empty chunk."
            )

        actions: list[Action] = []
        for index, item in enumerate(items):
            action = self.validate_action(item)
            try:
                _ensure_action_matches_policy_spec(action, spec)
            except InterfaceValidationError as exc:
                raise InterfaceValidationError(
                    f"invalid action at chunk index {index}: {exc}"
                ) from exc
            actions.append(action)
        return actions

    def inferaxis_infer(self, frame: Frame | Mapping[str, Any]) -> Action:
        """Normalize inputs and outputs around one wrapped single-step inference."""

        normalized_frame, spec = self._validate_policy_input_frame(frame)
        raw_infer = self._resolve_optional_impl("infer", "_infer_impl")

        if callable(raw_infer):
            raw_action = raw_infer(self.to_native_frame(normalized_frame))
            normalized_action = self.validate_action(raw_action)
            _ensure_action_matches_policy_spec(normalized_action, spec)
            return normalized_action

        raw_step = self._resolve_optional_impl("step", "_step_impl")
        if callable(raw_step):
            raw_action = raw_step(self.to_native_frame(normalized_frame))
            normalized_action = self.validate_action(raw_action)
            _ensure_action_matches_policy_spec(normalized_action, spec)
            return normalized_action

        raw_infer_chunk = self._resolve_optional_impl("infer_chunk", "_infer_chunk_impl")
        if callable(raw_infer_chunk):
            return self.inferaxis_infer_chunk(
                normalized_frame,
                self._single_step_chunk_request(),
            )[0]

        raw_step_chunk = self._resolve_optional_impl("step_chunk", "_step_chunk_impl")
        if callable(raw_step_chunk):
            return self.inferaxis_infer_chunk(
                normalized_frame,
                self._single_step_chunk_request(),
            )[0]

        raise InterfaceValidationError(
            f"{type(self).__name__} must expose infer(frame) or infer_chunk(frame, request)."
        )

    def step(self, frame: Frame | Mapping[str, Any]) -> Action:
        """Backward-compatible alias for :meth:`inferaxis_infer`."""

        return self.inferaxis_infer(frame)

    def inferaxis_infer_chunk(
        self,
        frame: Frame | Mapping[str, Any],
        request: object,
    ) -> list[Action]:
        """Normalize inputs and outputs around one chunk-producing inference."""

        normalized_frame, spec = self._validate_policy_input_frame(frame)
        raw_infer_chunk = self._resolve_optional_impl("infer_chunk", "_infer_chunk_impl")

        if callable(raw_infer_chunk):
            raw_plan = raw_infer_chunk(self.to_native_frame(normalized_frame), request)
            return self._coerce_action_plan(raw_plan, spec=spec)

        raw_step_chunk = self._resolve_optional_impl("step_chunk", "_step_chunk_impl")
        if callable(raw_step_chunk):
            raw_plan = raw_step_chunk(self.to_native_frame(normalized_frame), request)
            return self._coerce_action_plan(raw_plan, spec=spec)

        raw_infer = self._resolve_optional_impl("infer", "_infer_impl")
        if callable(raw_infer):
            raw_action = raw_infer(self.to_native_frame(normalized_frame))
            normalized_action = self.validate_action(raw_action)
            _ensure_action_matches_policy_spec(normalized_action, spec)
            return [normalized_action]

        raw_step = self._resolve_optional_impl("step", "_step_impl")
        if callable(raw_step):
            raw_action = raw_step(self.to_native_frame(normalized_frame))
            normalized_action = self.validate_action(raw_action)
            _ensure_action_matches_policy_spec(normalized_action, spec)
            return [normalized_action]

        raise InterfaceValidationError(
            f"{type(self).__name__} must expose infer_chunk(frame, request) or infer(frame)."
        )

    def step_chunk(
        self,
        frame: Frame | Mapping[str, Any],
        request: object,
    ) -> list[Action]:
        """Backward-compatible alias for :meth:`inferaxis_infer_chunk`."""

        return self.inferaxis_infer_chunk(frame, request)

    def inferaxis_plan(self, frame: Frame | Mapping[str, Any]) -> list[Action]:
        """Return one normalized action plan when the wrapped policy exposes one."""

        normalized_frame, spec = self._validate_policy_input_frame(frame)
        raw_plan = self._resolve_optional_impl("plan", "_plan_impl")
        if callable(raw_plan):
            return self._coerce_action_plan(
                raw_plan(self.to_native_frame(normalized_frame)),
                spec=spec,
            )
        return self.inferaxis_infer_chunk(
            normalized_frame,
            self._single_step_chunk_request(),
        )

__all__ = ["PolicyMixin"]
