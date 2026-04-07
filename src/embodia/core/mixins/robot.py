"""Robot-facing mixin for embodia runtime integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
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
from ..errors import InterfaceValidationError
from ..modalities import images, state, task
from ..modalities._common import CONTROL_TARGETS
from ..schema import Action, Frame, RobotSpec
from ..transform import invert_mapping, remap_action, remap_robot_spec, robot_spec_to_dict
from .common import _CommonInterfaceMixin


class RobotMixin(_CommonInterfaceMixin):
    """Mixin that wraps a robot implementation with embodia behavior."""

    ROBOT_SPEC: RobotSpec | Mapping[str, Any] | None = None
    REMOTE_POLICY_ENABLED = False
    REMOTE_POLICY_REQUEST_METHOD: str | None = None
    REMOTE_POLICY_RESPONSE_METHOD: str | None = None
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
        "remote_policy",
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
        remote_policy: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Instantiate one robot from validated embodia runtime config."""

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
        validated_remote_policy = (
            cls._validate_remote_policy_config(remote_policy)
            if remote_policy is not None
            else None
        )

        robot = cls(*args, **kwargs)
        robot._configure_runtime_interface(
            method_aliases=validated_method_aliases,
            modality_maps=validated_modality_maps,
        )
        if validated_robot_spec is not None:
            robot._embodia_runtime_robot_spec = validated_robot_spec
        if validated_remote_policy is None:
            return robot
        robot.configure_remote_policy(**validated_remote_policy)
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
            state_key_map=cls._effective_modality_map(
                state.STATE_KEYS,
                modality_maps=modality_maps,
            ),
            task_key_map=cls._effective_modality_map(
                task.TASK_KEYS,
                modality_maps=modality_maps,
            ),
            command_kind_map=cls._effective_command_kind_map(
                modality_maps=modality_maps
            ),
        )
        _validate_robot_spec(normalized)
        return normalized

    @classmethod
    def _validate_remote_policy_config(
        cls,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate remote-policy config before instance construction."""

        if not isinstance(value, Mapping):
            raise InterfaceValidationError(
                "remote_policy must be a mapping of "
                "RobotMixin.configure_remote_policy(...) arguments."
            )

        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InterfaceValidationError(
                    "remote_policy must use string keys, got "
                    f"{key!r} of type {type(key).__name__}."
                )
            copied[key] = item

        try:
            inspect.signature(cls.configure_remote_policy).bind_partial(None, **copied)
        except TypeError as exc:
            raise InterfaceValidationError(
                "remote_policy contains unsupported keys for "
                "RobotMixin.configure_remote_policy(...)."
            ) from exc

        raw_runner = copied.get("runner")
        if raw_runner is not None:
            cls._validate_remote_policy_runner(raw_runner)

        raw_request_builder = copied.get("request_builder")
        if raw_request_builder is not None and not callable(raw_request_builder):
            raise InterfaceValidationError(
                "remote_policy.request_builder must be callable when provided."
            )

        raw_response_to_action = copied.get("response_to_action")
        if raw_response_to_action is not None and not callable(raw_response_to_action):
            raise InterfaceValidationError(
                "remote_policy.response_to_action must be callable when provided."
            )

        raw_enabled = copied.get("enabled")
        if raw_enabled is not None and not isinstance(raw_enabled, bool):
            raise InterfaceValidationError(
                "remote_policy.enabled must be a bool when provided."
            )

        return copied

    @classmethod
    def _validate_remote_policy_runner(cls, runner: object) -> object:
        """Validate one remote-policy runner by duck typing."""

        infer = getattr(runner, "infer", None)
        if not callable(infer):
            raise InterfaceValidationError(
                "remote_policy.runner must expose infer(request)."
            )
        return runner

    def uses_remote_policy(self) -> bool:
        """Return whether this robot instance is configured for remote policy use."""

        return self.has_remote_policy()

    def embodia_has_remote_policy(self) -> bool:
        """Internal embodia-prefixed alias for remote-policy availability."""

        return self.has_remote_policy()

    def has_remote_policy(self) -> bool:
        """Return whether a remote policy backend has been configured."""

        runner = getattr(self, "_embodia_remote_policy_runner", None)
        enabled = getattr(
            self,
            "_embodia_remote_policy_enabled",
            bool(type(self).REMOTE_POLICY_ENABLED),
        )
        return bool(enabled) and runner is not None

    def close_remote_policy(self) -> None:
        """Close the hidden remote-policy backend when present."""

        runner = getattr(self, "_embodia_remote_policy_runner", None)
        if runner is None:
            return

        close = getattr(runner, "close", None)
        if callable(close):
            close()

    def configure_remote_policy(
        self,
        *,
        runner: object | None = None,
        request_builder: Callable[[Frame], Mapping[str, Any]] | None = None,
        response_to_action: Callable[[object], Action | Mapping[str, Any]] | None = None,
        enabled: bool = True,
    ) -> None:
        """Configure one generic remote-policy backend for this robot."""

        if not isinstance(enabled, bool):
            raise InterfaceValidationError("enabled must be a bool.")

        if runner is not None:
            self._embodia_remote_policy_runner = self._validate_remote_policy_runner(
                runner
            )
        elif enabled:
            raise InterfaceValidationError(
                "configure_remote_policy() requires runner=... when enabled=True."
            )

        if request_builder is not None:
            if not callable(request_builder):
                raise InterfaceValidationError(
                    "request_builder must be callable when provided."
                )
            self._embodia_remote_policy_request_builder = request_builder

        if response_to_action is not None:
            if not callable(response_to_action):
                raise InterfaceValidationError(
                    "response_to_action must be callable when provided."
                )
            self._embodia_remote_policy_response_to_action = response_to_action

        self._embodia_remote_policy_enabled = enabled

    def _resolve_remote_policy_request_builder(self) -> Any:
        """Resolve the method that converts a frame into a remote request payload."""

        configured = getattr(self, "_embodia_remote_policy_request_builder", None)
        if callable(configured):
            return configured

        method_name = type(self).REMOTE_POLICY_REQUEST_METHOD
        if method_name is None and callable(
            getattr(self, "_build_remote_policy_request", None)
        ):
            method_name = "_build_remote_policy_request"

        if method_name is None:
            raise InterfaceValidationError(
                f"{type(self).__name__} uses remote policy mode, but does not "
                "declare REMOTE_POLICY_REQUEST_METHOD or define "
                "_build_remote_policy_request()."
            )
        method = getattr(self, method_name, None)
        if not callable(method):
            raise InterfaceValidationError(
                f"{type(self).__name__}.{method_name} must be callable for remote "
                "policy mode."
            )
        return method

    def _resolve_remote_policy_response_to_action(self) -> Any:
        """Resolve the method that converts one remote response into an action."""

        configured = getattr(self, "_embodia_remote_policy_response_to_action", None)
        if callable(configured):
            return configured

        method_name = type(self).REMOTE_POLICY_RESPONSE_METHOD
        if method_name is None and callable(
            getattr(self, "_parse_remote_policy_action", None)
        ):
            method_name = "_parse_remote_policy_action"

        if method_name is None:
            raise InterfaceValidationError(
                f"{type(self).__name__} uses remote policy mode, but does not "
                "declare REMOTE_POLICY_RESPONSE_METHOD or define "
                "_parse_remote_policy_action()."
            )
        method = getattr(self, method_name, None)
        if not callable(method):
            raise InterfaceValidationError(
                f"{type(self).__name__}.{method_name} must be callable for remote "
                "policy mode."
            )
        return method

    def embodia_request_remote_policy_action(
        self,
        frame: Frame | Mapping[str, Any] | None = None,
    ) -> Action:
        """Internal helper used by embodia runtime for remote policy actions."""

        runner = getattr(self, "_embodia_remote_policy_runner", None)
        if runner is None or not self.has_remote_policy():
            raise InterfaceValidationError(
                "Remote policy access is disabled for this robot instance. "
                "Configure it with configure_remote_policy(...)."
            )

        normalized_frame = (
            self.embodia_observe() if frame is None else self.validate_frame(frame)
        )
        spec = self.embodia_get_spec()
        images.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=spec.image_keys,
        )
        state.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=spec.all_state_keys(),
        )
        task.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=spec.task_keys,
        )
        build_request = self._resolve_remote_policy_request_builder()
        policy_output = runner.infer(build_request(normalized_frame))
        setattr(self, "last_policy_output", policy_output)

        parse_action = self._resolve_remote_policy_response_to_action()
        action = self.validate_action(parse_action(policy_output))
        return self.ensure_action_supported(action)

    def request_remote_policy_action(
        self,
        frame: Frame | Mapping[str, Any] | None = None,
    ) -> Action:
        """Backward-compatible alias for embodia's internal remote action path."""

        return self.embodia_request_remote_policy_action(frame)

    def _request_remote_policy_action(
        self,
        frame: Frame | Mapping[str, Any] | None = None,
    ) -> Action:
        """Backward-compatible alias kept for older internal embodia call sites."""

        return self.embodia_request_remote_policy_action(frame)

    def normalize_spec(self, spec: RobotSpec | Mapping[str, Any]) -> RobotSpec:
        """Transform a robot spec-like value into :class:`RobotSpec`."""

        return remap_robot_spec(
            spec,
            image_key_map=self.get_image_key_map(),
            target_map=self.get_control_target_map(),
            state_key_map=self.get_state_key_map(),
            task_key_map=self.get_task_key_map(),
            command_kind_map=self.get_command_kind_map(),
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

        instance_cached = getattr(self, "_embodia_runtime_robot_spec", None)
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
            self.embodia_get_spec() if spec is None else self.validate_spec(spec)
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
        task.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=normalized_spec.task_keys,
        )
        return normalized_frame

    def ensure_action_supported(
        self,
        action: Action | Mapping[str, Any],
        spec: RobotSpec | Mapping[str, Any] | None = None,
    ) -> Action:
        """Ensure command kinds and dims are supported by the robot spec."""

        normalized_action = self.validate_action(action)
        normalized_spec = (
            self.embodia_get_spec() if spec is None else self.validate_spec(spec)
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
            kind_map=invert_mapping(
                self.get_command_kind_map(),
                "RobotMixin command kind mapping",
            ),
        )

    def embodia_get_spec(self) -> RobotSpec:
        """Return the normalized robot spec used internally by embodia."""

        return self._get_runtime_spec()

    def get_spec(self) -> RobotSpec:
        """Backward-compatible alias for :meth:`embodia_get_spec`."""

        return self.embodia_get_spec()

    def embodia_observe(self) -> Frame:
        """Return the normalized frame used internally by embodia."""

        raw_observe = self._resolve_impl("observe", "_observe_impl")
        frame = self.validate_frame(raw_observe())
        spec = self.embodia_get_spec()
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
        task.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.task_keys,
        )
        return frame

    def observe(self) -> Frame:
        """Backward-compatible alias for :meth:`embodia_observe`."""

        return self.embodia_observe()

    def embodia_act(self, action: Action | Mapping[str, Any]) -> None:
        """Normalize and validate an action before forwarding it."""

        raw_act = self._resolve_impl("act", "_act_impl")
        normalized_action = self.validate_action(action)
        spec = self.embodia_get_spec()
        _ensure_action_supported_by_robot(normalized_action, spec)
        raw_act(self.to_native_action(normalized_action))

    def act(self, action: Action | Mapping[str, Any]) -> None:
        """Backward-compatible alias for :meth:`embodia_act`."""

        self.embodia_act(action)

    def embodia_reset(self) -> Frame:
        """Return the normalized reset frame used internally by embodia."""

        raw_reset = self._resolve_impl("reset", "_reset_impl")
        frame = self.validate_frame(raw_reset())
        spec = self.embodia_get_spec()
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
        task.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.task_keys,
        )
        return frame

    def reset(self) -> Frame:
        """Backward-compatible alias for :meth:`embodia_reset`."""

        return self.embodia_reset()


__all__ = ["RobotMixin"]
