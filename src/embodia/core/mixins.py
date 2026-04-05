"""Reusable mixins that add normalization and validation behavior."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..runtime.checks import (
    validate_action as _validate_action,
    validate_frame as _validate_frame,
    validate_model_spec as _validate_model_spec,
    validate_robot_spec as _validate_robot_spec,
)
from .modalities._common import resolve_string_mapping
from .modalities import action_modes, images, state
from .errors import InterfaceValidationError
from .schema import Action, Frame, ModelSpec, RobotSpec
from .transform import (
    action_to_dict,
    frame_to_dict,
    invert_mapping,
    model_spec_to_dict,
    remap_action,
    remap_frame,
    remap_model_spec,
    remap_robot_spec,
    robot_spec_to_dict,
)


class _CommonInterfaceMixin:
    """Shared transform and validation helpers."""

    METHOD_ALIASES: Mapping[str, str] = {}
    _METHOD_ALIAS_ATTRS: Mapping[str, str] = {}

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

    def _resolve_mapping(self, attr_name: str) -> Mapping[str, str]:
        """Resolve a declarative mapping attribute."""

        return resolve_string_mapping(self, attr_name)

    def _resolve_local_method(self, method_name: str) -> Any | None:
        """Resolve a method defined directly on the integration class.

        This supports the common "edit the existing outer class in place"
        workflow:

        ``class VendorRobot(RobotMixin): ...``

        where native methods such as ``capture()`` or ``infer()`` live on the
        same class body rather than a separate parent class.
        """

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
        """Resolve an alternative wrapped method name.

        ``METHOD_ALIASES`` is the preferred declarative form. Per-method class
        attributes such as ``OBSERVE_METHOD`` remain supported for clarity and
        backward compatibility.
        """

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

        return images.get_key_map(self)

    def get_state_key_map(self) -> Mapping[str, str]:
        """Map native state keys to embodia-standard state keys."""

        return state.get_key_map(self)

    def get_action_mode_map(self) -> Mapping[str, str]:
        """Map native action modes to embodia-standard action modes."""

        return action_modes.get_mode_map(self)

    def normalize_frame(self, frame: Frame | Mapping[str, Any]) -> Frame:
        """Transform a frame-like value into :class:`Frame`."""

        return remap_frame(
            frame,
            image_key_map=self.get_image_key_map(),
            state_key_map=self.get_state_key_map(),
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
            mode_map=self.get_action_mode_map(),
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

class RobotMixin(_CommonInterfaceMixin):
    """Mixin that wraps a robot implementation with embodia behavior.

    Supported integration styles:

    1. Existing class:
       ``class MyRobot(RobotMixin, VendorRobot): ...``
    2. Fresh implementation:
       define ``_get_spec_impl()``, ``_observe_impl()``, ``_act_impl()``,
       and ``_reset_impl()`` directly on the subclass.

    Minimal-intrusion configuration is usually done with class attributes:

    ``ROBOT_SPEC``, ``METHOD_ALIASES``, ``IMAGE_KEY_MAP``, ``STATE_KEY_MAP``,
    ``ACTION_MODE_MAP``.

    Fine-grained alias attributes such as ``OBSERVE_METHOD`` remain available
    when a project prefers explicit per-method declarations.

    Important:
    ``RobotMixin`` must be the leftmost direct base in the integration class,
    for example ``class MyRobot(RobotMixin, VendorRobot): ...``. This mixin
    relies on method resolution order to intercept runtime calls first.
    """

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Catch the common MRO mistake early.

        ``RobotMixin`` must be the outermost mixin layer so its normalization
        and validation wrappers are not bypassed by earlier bases.
        """

        super().__init_subclass__(**kwargs)
        direct_bases = cls.__bases__
        if RobotMixin in direct_bases and direct_bases[0] is not RobotMixin:
            raise TypeError(
                f"{cls.__name__} must list RobotMixin as its first base class, "
                f"for example: class {cls.__name__}(RobotMixin, VendorRobot): ..."
            )

    def normalize_spec(self, spec: RobotSpec | Mapping[str, Any]) -> RobotSpec:
        """Transform a robot spec-like value into :class:`RobotSpec`."""

        return remap_robot_spec(
            spec,
            image_key_map=self.get_image_key_map(),
            state_key_map=self.get_state_key_map(),
            action_mode_map=self.get_action_mode_map(),
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

    def ensure_frame_matches_spec(
        self,
        frame: Frame | Mapping[str, Any],
        spec: RobotSpec | Mapping[str, Any] | None = None,
    ) -> Frame:
        """Ensure frame keys satisfy the robot spec."""

        normalized_frame = self.validate_frame(frame)
        normalized_spec = self.get_spec() if spec is None else self.validate_spec(spec)
        images.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=normalized_spec.image_keys,
        )
        state.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=normalized_spec.state_keys,
        )
        return normalized_frame

    def ensure_action_supported(
        self,
        action: Action | Mapping[str, Any],
        spec: RobotSpec | Mapping[str, Any] | None = None,
    ) -> Action:
        """Ensure the action mode is supported by the robot spec."""

        normalized_action = self.validate_action(action)
        normalized_spec = self.get_spec() if spec is None else self.validate_spec(spec)
        action_modes.ensure_supported(
            normalized_action,
            normalized_spec.action_modes,
            owner_label="robot",
            owner_name=normalized_spec.name,
        )
        return normalized_action

    def to_native_action(self, action: Action) -> Any:
        """Convert a normalized action into the wrapped implementation's format."""

        return remap_action(
            action,
            mode_map=invert_mapping(
                self.get_action_mode_map(),
                "RobotMixin action mode mapping",
            ),
        )

    def get_spec(self) -> RobotSpec:
        """Return a normalized, validated robot spec from the wrapped class."""

        if type(self).ROBOT_SPEC is not None:
            return self.validate_spec(type(self).ROBOT_SPEC)

        raw_get_spec = self._resolve_impl("get_spec", "_get_spec_impl")
        return self.validate_spec(raw_get_spec())

    def observe(self) -> Frame:
        """Return a normalized, validated frame from the wrapped class."""

        raw_observe = self._resolve_impl("observe", "_observe_impl")
        frame = self.validate_frame(raw_observe())
        return self.ensure_frame_matches_spec(frame)

    def act(self, action: Action | Mapping[str, Any]) -> None:
        """Normalize and validate an action before forwarding it."""

        raw_act = self._resolve_impl("act", "_act_impl")
        normalized_action = self.ensure_action_supported(action)
        raw_act(self.to_native_action(normalized_action))

    def reset(self) -> Frame:
        """Return a normalized, validated reset frame from the wrapped class."""

        raw_reset = self._resolve_impl("reset", "_reset_impl")
        frame = self.validate_frame(raw_reset())
        return self.ensure_frame_matches_spec(frame)


class ModelMixin(_CommonInterfaceMixin):
    """Mixin that wraps a model implementation with embodia behavior.

    Supported integration styles:

    1. Existing class:
       ``class MyModel(ModelMixin, VendorModel): ...``
    2. Fresh implementation:
       define ``_get_spec_impl()``, ``_reset_impl()``, and ``_step_impl()``
       directly on the subclass.

    Minimal-intrusion configuration is usually done with class attributes:

    ``MODEL_SPEC``, ``METHOD_ALIASES``, ``IMAGE_KEY_MAP``, ``STATE_KEY_MAP``,
    ``ACTION_MODE_MAP``.

    Fine-grained alias attributes such as ``STEP_METHOD`` remain available
    when a project prefers explicit per-method declarations.

    Important:
    ``ModelMixin`` must be the leftmost direct base in the integration class,
    for example ``class MyModel(ModelMixin, VendorModel): ...``. This mixin
    relies on method resolution order to intercept runtime calls first.
    """

    MODEL_SPEC: ModelSpec | Mapping[str, Any] | None = None
    GET_SPEC_METHOD = "get_spec"
    RESET_METHOD = "reset"
    STEP_METHOD = "step"

    _METHOD_ALIAS_ATTRS: dict[str, str] = {
        "get_spec": "GET_SPEC_METHOD",
        "reset": "RESET_METHOD",
        "step": "STEP_METHOD",
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Catch the common MRO mistake early.

        ``ModelMixin`` must be the outermost mixin layer so its normalization
        and validation wrappers are not bypassed by earlier bases.
        """

        super().__init_subclass__(**kwargs)
        direct_bases = cls.__bases__
        if ModelMixin in direct_bases and direct_bases[0] is not ModelMixin:
            raise TypeError(
                f"{cls.__name__} must list ModelMixin as its first base class, "
                f"for example: class {cls.__name__}(ModelMixin, VendorModel): ..."
            )

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

        if type(self).MODEL_SPEC is not None:
            return self.validate_spec(type(self).MODEL_SPEC)

        raw_get_spec = self._resolve_impl("get_spec", "_get_spec_impl")
        return self.validate_spec(raw_get_spec())

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
        normalized_frame = self.ensure_frame_satisfies_spec(frame)
        raw_action = raw_step(self.to_native_frame(normalized_frame))
        return self.ensure_output_matches_spec(raw_action)


__all__ = ["ModelMixin", "RobotMixin"]
