"""Robot-facing mixin for inferaxis runtime integration."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any, Self

from ..config_io import (
    expand_component_yaml_config,
    load_component_yaml_config,
)
from ...runtime.checks import (
    ensure_action_supported_by_robot as _ensure_action_supported_by_robot,
    validate_robot_spec as _validate_robot_spec,
)
from ...runtime.shared.sequence import attach_runtime_frame_metadata
from ..errors import InterfaceValidationError
from ..modalities import images, state
from ..modalities._common import CONTROL_TARGETS
from ..schema import Action, Frame, RobotSpec
from ..transform import invert_mapping, remap_action, remap_robot_spec, robot_spec_to_dict
from .common import _CommonInterfaceMixin


class RobotMixin(_CommonInterfaceMixin):
    """Mixin that wraps a robot implementation with inferaxis behavior."""

    ROBOT_SPEC: RobotSpec | Mapping[str, Any] | None = None
    GET_SPEC_METHOD = "get_spec"
    OBSERVE_METHOD = "observe"
    ACT_METHOD = "act"
    RESET_METHOD = "reset"

    _METHOD_ALIAS_ATTRS: dict[str, str] = {
        "get_spec": "GET_SPEC_METHOD",
        "observe": "OBSERVE_METHOD",
        "act": "ACT_METHOD",
        "reset": "RESET_METHOD",
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
        if RobotMixin in direct_bases and direct_bases[0] is not RobotMixin:
            raise TypeError(
                f"{cls.__name__} must list RobotMixin as its first base class, "
                f"for example: class {cls.__name__}(RobotMixin, VendorRobot): ..."
            )

    @classmethod
    def from_config(
        cls,
        *args: Any,
        robot_spec: RobotSpec | Mapping[str, Any] | None = None,
        method_aliases: Mapping[str, str] | None = None,
        modality_maps: Mapping[object, Mapping[str, str]] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Instantiate one robot from validated inferaxis runtime config."""

        cls._validate_declared_runtime_interface_config()
        validated_method_aliases = cls._validate_method_aliases_config(method_aliases)
        validated_modality_maps = cls._validate_modality_maps_config(modality_maps)
        validated_robot_spec = (
            cls._validate_robot_spec_config(
                robot_spec,
                modality_maps=validated_modality_maps,
            )
            if robot_spec is not None
            else None
        )

        robot = cls(*args, **kwargs)
        robot._configure_runtime_interface(
            method_aliases=validated_method_aliases,
            modality_maps=validated_modality_maps,
        )
        if validated_robot_spec is not None:
            robot._inferaxis_runtime_robot_spec = validated_robot_spec
        return robot

    @classmethod
    def from_yaml(
        cls,
        path: str | PathLike[str],
        **overrides: Any,
    ) -> Self:
        """Instantiate one robot from a YAML config file.

        The file may either contain a top-level ``robot:`` section or be a
        direct robot-only config mapping.
        """

        loaded = load_component_yaml_config(path, component="robot")
        cls._validate_yaml_section_keys(
            loaded,
            allowed_keys=cls._YAML_CONFIG_KEYS,
            config_label=f"robot YAML config at {path}",
        )
        loaded = expand_component_yaml_config(
            loaded,
            component="robot",
            path=path,
        )
        loaded.update(overrides)
        return cls.from_config(**loaded)

    @classmethod
    def _validate_robot_spec_config(
        cls,
        spec: RobotSpec | Mapping[str, Any],
        *,
        modality_maps: Mapping[str, Mapping[str, str]] | None = None,
    ) -> RobotSpec:
        """Validate one robot spec before instance construction."""

        normalized = remap_robot_spec(
            spec,
            image_key_map=cls._effective_modality_map(
                images.IMAGE_KEYS,
                modality_maps=modality_maps,
            ),
            target_map=cls._effective_modality_map(
                CONTROL_TARGETS,
                modality_maps=modality_maps,
            ),
            command_map=cls._effective_command_kind_map(
                modality_maps=modality_maps
            ),
        )
        _validate_robot_spec(normalized)
        return normalized

    def normalize_spec(self, spec: RobotSpec | Mapping[str, Any]) -> RobotSpec:
        """Transform a robot spec-like value into :class:`RobotSpec`."""

        return remap_robot_spec(
            spec,
            image_key_map=self.get_image_key_map(),
            target_map=self.get_control_target_map(),
            command_map=self.get_command_kind_map(),
        )

    def transform_spec(self, spec: RobotSpec | Mapping[str, Any]) -> RobotSpec:
        """Alias for :meth:`normalize_spec`."""

        return self.normalize_spec(spec)

    def validate_spec(self, spec: RobotSpec | Mapping[str, Any]) -> RobotSpec:
        """Normalize and validate a robot spec-like value."""

        normalized = self.normalize_spec(spec)
        _validate_robot_spec(normalized)
        return normalized

    def spec_to_dict(self, spec: RobotSpec | Mapping[str, Any]) -> dict[str, Any]:
        """Export a robot spec-like value into a plain dictionary."""

        return robot_spec_to_dict(spec)

    def _get_runtime_spec(self) -> RobotSpec:
        """Return the robot spec with caching for declarative class specs."""

        instance_cached = getattr(self, "_inferaxis_runtime_robot_spec", None)
        if instance_cached is not None:
            return instance_cached

        if type(self).ROBOT_SPEC is None:
            raw_get_spec = self._resolve_impl("get_spec", "_get_spec_impl")
            return self.validate_spec(raw_get_spec())

        cached = self._get_cached_class_value("_EMBODIA_CACHED_ROBOT_SPEC")
        if cached is not None:
            return cached
        return self._set_cached_class_value(
            "_EMBODIA_CACHED_ROBOT_SPEC",
            self.validate_spec(type(self).ROBOT_SPEC),
        )

    def ensure_frame_matches_spec(
        self,
        frame: Frame | Mapping[str, Any],
        spec: RobotSpec | Mapping[str, Any] | None = None,
    ) -> Frame:
        """Ensure frame keys satisfy the robot spec."""

        normalized_frame = self.validate_frame(frame)
        normalized_spec = (
            self.inferaxis_get_spec() if spec is None else self.validate_spec(spec)
        )
        images.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=normalized_spec.image_keys,
        )
        state.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=normalized_spec.all_state_keys(),
        )
        return normalized_frame

    def ensure_action_supported(
        self,
        action: Action | Mapping[str, Any],
        spec: RobotSpec | Mapping[str, Any] | None = None,
    ) -> Action:
        """Ensure commands and dims are supported by the robot spec."""

        normalized_action = self.validate_action(action)
        normalized_spec = (
            self.inferaxis_get_spec() if spec is None else self.validate_spec(spec)
        )
        _ensure_action_supported_by_robot(normalized_action, normalized_spec)
        return normalized_action

    def to_native_action(self, action: Action) -> Any:
        """Convert a normalized action into the wrapped implementation's format."""

        return remap_action(
            action,
            target_map=invert_mapping(
                self.get_control_target_map(),
                "RobotMixin control target mapping",
            ),
            command_map=invert_mapping(
                self.get_command_kind_map(),
                "RobotMixin command mapping",
            ),
        )

    def inferaxis_get_spec(self) -> RobotSpec:
        """Return the normalized robot spec used internally by inferaxis."""

        return self._get_runtime_spec()

    def get_spec(self) -> RobotSpec:
        """Backward-compatible alias for :meth:`inferaxis_get_spec`."""

        return self.inferaxis_get_spec()

    def inferaxis_observe(self) -> Frame:
        """Return the normalized frame used internally by inferaxis."""

        raw_observe = self._resolve_impl("observe", "_observe_impl")
        frame = attach_runtime_frame_metadata(
            self.validate_frame(raw_observe()),
            owner=self,
        )
        spec = self.inferaxis_get_spec()
        images.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.image_keys,
        )
        state.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.all_state_keys(),
        )
        return frame

    def observe(self) -> Frame:
        """Backward-compatible alias for :meth:`inferaxis_observe`."""

        return self.inferaxis_observe()

    def inferaxis_act(self, action: Action | Mapping[str, Any]) -> Action:
        """Normalize, execute, and return the final action used by the robot.

        When the wrapped native ``act`` / ``send_command`` method returns an
        action-like value, inferaxis treats that as the robot-confirmed action
        and normalizes it back into the shared schema. When it returns
        ``None`` or some non-action status object, inferaxis falls back to the
        requested normalized action.
        """

        raw_act = self._resolve_impl("act", "_act_impl")
        normalized_action = self.validate_action(action)
        spec = self.inferaxis_get_spec()
        _ensure_action_supported_by_robot(normalized_action, spec)
        raw_result = raw_act(self.to_native_action(normalized_action))
        if not isinstance(raw_result, (Action, Mapping)):
            return normalized_action

        executed_action = self.validate_action(raw_result)
        _ensure_action_supported_by_robot(executed_action, spec)
        return executed_action

    def act(self, action: Action | Mapping[str, Any]) -> Action:
        """Backward-compatible alias for :meth:`inferaxis_act`."""

        return self.inferaxis_act(action)

    def inferaxis_reset(self) -> Frame:
        """Return the normalized reset frame used internally by inferaxis."""

        raw_reset = self._resolve_impl("reset", "_reset_impl")
        frame = attach_runtime_frame_metadata(
            self.validate_frame(raw_reset()),
            owner=self,
            reset=True,
        )
        spec = self.inferaxis_get_spec()
        images.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.image_keys,
        )
        state.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.all_state_keys(),
        )
        return frame

    def reset(self) -> Frame:
        """Backward-compatible alias for :meth:`inferaxis_reset`."""

        return self.inferaxis_reset()


__all__ = ["RobotMixin"]
