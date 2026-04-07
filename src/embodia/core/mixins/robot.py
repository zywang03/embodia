"""Robot-facing mixin for embodia runtime integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from numbers import Real
from os import PathLike
from typing import Any, Self

from ..config_io import (
    expand_component_yaml_interface_config,
    load_component_yaml_config,
)
from ...runtime.checks import validate_robot_spec as _validate_robot_spec
from ..errors import InterfaceValidationError
from ..modalities import action_modes, images, state
from ..schema import Action, Frame, RobotSpec
from ..transform import invert_mapping, remap_action, remap_robot_spec, robot_spec_to_dict
from .common import _CommonInterfaceMixin


class RobotMixin(_CommonInterfaceMixin):
    """Mixin that wraps a robot implementation with embodia behavior."""

    ROBOT_SPEC: RobotSpec | Mapping[str, Any] | None = None
    REMOTE_POLICY_ENABLED = False
    REMOTE_POLICY_HOST = "localhost"
    REMOTE_POLICY_PORT = 8000
    REMOTE_POLICY_API_KEY: str | None = None
    REMOTE_POLICY_RETRY_INTERVAL_S = 5.0
    REMOTE_POLICY_CONNECT_TIMEOUT_S: float | None = None
    REMOTE_POLICY_ADDITIONAL_HEADERS: Mapping[str, str] | None = None
    REMOTE_POLICY_CONNECT_IMMEDIATELY = False
    REMOTE_POLICY_WAIT_FOR_SERVER = True
    REMOTE_POLICY_OBS_METHOD: str | None = None
    REMOTE_POLICY_ACTION_MODE: str | None = None
    REMOTE_POLICY_DT: float | str | None = None
    REMOTE_POLICY_REF_FRAME: str | None = None
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
        "init",
        "interface",
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
        loaded = expand_component_yaml_interface_config(
            loaded,
            component="robot",
            path=path,
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
            state_key_map=cls._effective_modality_map(
                state.STATE_KEYS,
                modality_maps=modality_maps,
            ),
            action_mode_map=cls._effective_modality_map(
                action_modes.ACTION_MODES,
                modality_maps=modality_maps,
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

        def ensure_optional_non_empty_string(
            field_name: str,
            *,
            allow_none: bool = True,
        ) -> None:
            raw = copied.get(field_name)
            if raw is None and allow_none:
                return
            if not isinstance(raw, str) or not raw.strip():
                raise InterfaceValidationError(
                    f"remote_policy.{field_name} must be a non-empty string."
                )

        def ensure_optional_positive_float(field_name: str) -> None:
            raw = copied.get(field_name)
            if raw is None:
                return
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise InterfaceValidationError(
                    f"remote_policy.{field_name} must be a real number, got "
                    f"{type(raw).__name__}."
                )
            if float(raw) <= 0.0:
                raise InterfaceValidationError(
                    f"remote_policy.{field_name} must be > 0, got {raw!r}."
                )

        def ensure_optional_bool(field_name: str) -> None:
            raw = copied.get(field_name)
            if raw is None:
                return
            if not isinstance(raw, bool):
                raise InterfaceValidationError(
                    f"remote_policy.{field_name} must be a bool, got "
                    f"{type(raw).__name__}."
                )

        ensure_optional_non_empty_string("host")
        raw_port = copied.get("port")
        if raw_port is not None and (
            isinstance(raw_port, bool) or not isinstance(raw_port, int)
        ):
            raise InterfaceValidationError(
                f"remote_policy.port must be an int, got {type(raw_port).__name__}."
            )

        raw_api_key = copied.get("api_key")
        if raw_api_key is not None and not isinstance(raw_api_key, str):
            raise InterfaceValidationError(
                "remote_policy.api_key must be a string or None, got "
                f"{type(raw_api_key).__name__}."
            )

        ensure_optional_positive_float("retry_interval_s")
        ensure_optional_positive_float("connect_timeout_s")
        ensure_optional_positive_float("dt")
        ensure_optional_bool("connect_immediately")
        ensure_optional_bool("wait_for_server")
        ensure_optional_bool("enabled")
        ensure_optional_non_empty_string("action_mode")
        ensure_optional_non_empty_string("ref_frame")

        raw_headers = copied.get("additional_headers")
        if raw_headers is not None:
            cls._copy_string_mapping_cls(
                raw_headers,
                field_name="remote_policy.additional_headers",
            )

        raw_obs_builder = copied.get("obs_builder")
        if raw_obs_builder is not None and not callable(raw_obs_builder):
            raise InterfaceValidationError(
                "remote_policy.obs_builder must be callable when provided."
            )

        return copied

    def _import_openpi_remote(self) -> Any:
        """Import the optional OpenPI remote helpers lazily."""

        from ...contrib import openpi_remote as em_openpi_remote

        return em_openpi_remote

    def _coerce_optional_bool(self, value: object, field_name: str) -> bool:
        """Validate an optional boolean-like configuration field."""

        if not isinstance(value, bool):
            raise InterfaceValidationError(
                f"{field_name} must be a bool, got {type(value).__name__}."
            )
        return value

    def _coerce_optional_positive_float(self, value: object, field_name: str) -> float:
        """Validate a finite positive real number."""

        if isinstance(value, bool) or not isinstance(value, Real):
            raise InterfaceValidationError(
                f"{field_name} must be a real number, got {type(value).__name__}."
            )
        number = float(value)
        if number <= 0.0:
            raise InterfaceValidationError(f"{field_name} must be > 0, got {value!r}.")
        return number

    def _embodia_init_remote_policy(
        self,
        *,
        use_remote_policy: object | None = None,
        remote_policy_host: object | None = None,
        remote_policy_port: object | None = None,
        remote_policy_api_key: object | None = None,
        remote_policy_retry_interval_s: object | None = None,
        remote_policy_connect_timeout_s: object | None = None,
        remote_policy_additional_headers: object | None = None,
        remote_policy_connect_immediately: object | None = None,
        remote_policy_wait_for_server: object | None = None,
    ) -> None:
        """Initialize the hidden robot-side remote-policy backend."""

        enabled = type(self).REMOTE_POLICY_ENABLED
        if use_remote_policy is not None:
            enabled = self._coerce_optional_bool(
                use_remote_policy,
                "use_remote_policy",
            )

        host = type(self).REMOTE_POLICY_HOST
        if remote_policy_host is not None:
            if not isinstance(remote_policy_host, str) or not remote_policy_host.strip():
                raise InterfaceValidationError(
                    "remote_policy_host must be a non-empty string."
                )
            host = remote_policy_host

        port = type(self).REMOTE_POLICY_PORT
        if remote_policy_port is not None:
            if isinstance(remote_policy_port, bool) or not isinstance(
                remote_policy_port, int
            ):
                raise InterfaceValidationError("remote_policy_port must be an int.")
            port = remote_policy_port

        api_key = type(self).REMOTE_POLICY_API_KEY
        if remote_policy_api_key is not None:
            if remote_policy_api_key is not None and not isinstance(
                remote_policy_api_key, str
            ):
                raise InterfaceValidationError(
                    "remote_policy_api_key must be a string or None."
                )
            api_key = remote_policy_api_key

        retry_interval_s = type(self).REMOTE_POLICY_RETRY_INTERVAL_S
        if remote_policy_retry_interval_s is not None:
            retry_interval_s = self._coerce_optional_positive_float(
                remote_policy_retry_interval_s,
                "remote_policy_retry_interval_s",
            )

        connect_timeout_s = type(self).REMOTE_POLICY_CONNECT_TIMEOUT_S
        if remote_policy_connect_timeout_s is not None:
            connect_timeout_s = self._coerce_optional_positive_float(
                remote_policy_connect_timeout_s,
                "remote_policy_connect_timeout_s",
            )

        additional_headers = type(self).REMOTE_POLICY_ADDITIONAL_HEADERS
        if remote_policy_additional_headers is not None:
            if not isinstance(remote_policy_additional_headers, Mapping):
                raise InterfaceValidationError(
                    "remote_policy_additional_headers must be a mapping when "
                    "provided."
                )
            additional_headers = dict(remote_policy_additional_headers)

        connect_immediately = type(self).REMOTE_POLICY_CONNECT_IMMEDIATELY
        if remote_policy_connect_immediately is not None:
            connect_immediately = self._coerce_optional_bool(
                remote_policy_connect_immediately,
                "remote_policy_connect_immediately",
            )

        wait_for_server = type(self).REMOTE_POLICY_WAIT_FOR_SERVER
        if remote_policy_wait_for_server is not None:
            wait_for_server = self._coerce_optional_bool(
                remote_policy_wait_for_server,
                "remote_policy_wait_for_server",
            )

        em_openpi_remote = self._import_openpi_remote()
        self._embodia_remote_policy = em_openpi_remote.RemotePolicyRunner(
            enabled=enabled,
            host=host,
            port=port,
            api_key=api_key,
            retry_interval_s=retry_interval_s,
            connect_timeout_s=connect_timeout_s,
            additional_headers=additional_headers,
            connect_immediately=connect_immediately,
            wait_for_server=wait_for_server,
        )

    def uses_remote_policy(self) -> bool:
        """Return whether this robot instance is configured for remote policy use."""

        runner = getattr(self, "_embodia_remote_policy", None)
        if runner is None:
            self._embodia_init_remote_policy()
            runner = getattr(self, "_embodia_remote_policy")
        return bool(getattr(runner, "enabled", False))

    def has_remote_policy(self) -> bool:
        """Return whether remote policy is configured without initializing it."""

        runner = getattr(self, "_embodia_remote_policy", None)
        if runner is not None:
            return bool(getattr(runner, "enabled", False))
        return bool(type(self).REMOTE_POLICY_ENABLED)

    def close_remote_policy(self) -> None:
        """Close the hidden remote-policy backend when present."""

        runner = getattr(self, "_embodia_remote_policy", None)
        if runner is not None:
            runner.close()

    def configure_remote_policy(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        retry_interval_s: float | None = None,
        connect_timeout_s: float | None = None,
        additional_headers: Mapping[str, str] | None = None,
        connect_immediately: bool | None = None,
        wait_for_server: bool | None = None,
        obs_builder: Callable[[Frame], Mapping[str, Any]] | None = None,
        action_mode: str | None = None,
        dt: float | None = None,
        ref_frame: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Configure robot-side remote policy access without touching the class body."""

        if obs_builder is not None and not callable(obs_builder):
            raise InterfaceValidationError("obs_builder must be callable when provided.")
        if action_mode is not None:
            if not isinstance(action_mode, str) or not action_mode.strip():
                raise InterfaceValidationError(
                    "action_mode must be a non-empty string when provided."
                )
            self._embodia_remote_policy_action_mode = action_mode
        if dt is not None:
            self._embodia_remote_policy_dt = self._coerce_optional_positive_float(
                dt,
                "remote_policy_dt",
            )
        if ref_frame is not None:
            if not isinstance(ref_frame, str) or not ref_frame.strip():
                raise InterfaceValidationError(
                    "ref_frame must be a non-empty string when provided."
                )
            self._embodia_remote_policy_ref_frame = ref_frame
        if obs_builder is not None:
            self._embodia_remote_policy_obs_builder = obs_builder

        self._embodia_init_remote_policy(
            use_remote_policy=enabled,
            remote_policy_host=host,
            remote_policy_port=port,
            remote_policy_api_key=api_key,
            remote_policy_retry_interval_s=retry_interval_s,
            remote_policy_connect_timeout_s=connect_timeout_s,
            remote_policy_additional_headers=additional_headers,
            remote_policy_connect_immediately=connect_immediately,
            remote_policy_wait_for_server=wait_for_server,
        )

    def _resolve_remote_policy_obs_builder(self) -> Any:
        """Resolve the method that converts a frame into remote observation payload."""

        configured = getattr(self, "_embodia_remote_policy_obs_builder", None)
        if callable(configured):
            return configured

        method_name = type(self).REMOTE_POLICY_OBS_METHOD
        if method_name is None and callable(getattr(self, "_frame_to_openpi_obs", None)):
            method_name = "_frame_to_openpi_obs"

        if method_name is None:
            raise InterfaceValidationError(
                f"{type(self).__name__} uses remote policy mode, but does not "
                "declare REMOTE_POLICY_OBS_METHOD or define _frame_to_openpi_obs()."
            )
        method = getattr(self, method_name, None)
        if not callable(method):
            raise InterfaceValidationError(
                f"{type(self).__name__}.{method_name} must be callable for remote "
                "policy mode."
            )
        return method

    def remote_policy_action_mode(self) -> str:
        """Return the action mode expected from the remote policy."""

        configured = getattr(self, "_embodia_remote_policy_action_mode", None)
        if configured is not None:
            return configured

        configured = type(self).REMOTE_POLICY_ACTION_MODE
        if configured is not None:
            if not isinstance(configured, str) or not configured.strip():
                raise InterfaceValidationError(
                    "REMOTE_POLICY_ACTION_MODE must be a non-empty string when "
                    "provided."
                )
            return configured

        spec = self.get_spec()
        if len(spec.action_modes) == 1:
            return spec.action_modes[0]
        raise InterfaceValidationError(
            f"{type(self).__name__} supports multiple action modes "
            f"{spec.action_modes!r}; set REMOTE_POLICY_ACTION_MODE explicitly."
        )

    def remote_policy_action_dt(self) -> float:
        """Return the action dt used when converting remote policy responses."""

        configured = getattr(self, "_embodia_remote_policy_dt", None)
        if configured is not None:
            return self._coerce_optional_positive_float(
                configured,
                "remote_policy_dt",
            )

        config = type(self).REMOTE_POLICY_DT
        if config is None:
            value = 0.1
        elif isinstance(config, str):
            value = getattr(self, config)
        else:
            value = config
        return self._coerce_optional_positive_float(value, "remote_policy_dt")

    def remote_policy_ref_frame(self) -> str | None:
        """Return the reference frame used for remote policy actions."""

        configured = getattr(self, "_embodia_remote_policy_ref_frame", None)
        if configured is not None:
            return configured

        value = type(self).REMOTE_POLICY_REF_FRAME
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise InterfaceValidationError(
                "REMOTE_POLICY_REF_FRAME must be a non-empty string when provided."
            )
        return value

    def _request_remote_policy_action(
        self,
        frame: Frame | Mapping[str, Any] | None = None,
    ) -> Action:
        """Internal helper used by embodia runtime for remote policy actions."""

        runner = getattr(self, "_embodia_remote_policy", None)
        if runner is None:
            self._embodia_init_remote_policy()
            runner = getattr(self, "_embodia_remote_policy")

        if not getattr(runner, "enabled", False):
            raise InterfaceValidationError(
                "Remote policy access is disabled for this robot instance. "
                "Enable it with use_remote_policy=True."
            )

        normalized_frame = self.observe() if frame is None else self.validate_frame(frame)
        spec = self._get_runtime_spec()
        images.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=spec.image_keys,
        )
        state.ensure_frame_keys(
            normalized_frame,
            owner_label="robot",
            required_keys=spec.state_keys,
        )
        build_obs = self._resolve_remote_policy_obs_builder()
        policy_output = runner.infer(build_obs(normalized_frame))
        setattr(self, "last_policy_output", policy_output)

        em_openpi_remote = self._import_openpi_remote()
        action = em_openpi_remote.openpi_first_action(
            policy_output,
            mode=self.remote_policy_action_mode(),
            dt=self.remote_policy_action_dt(),
            ref_frame=self.remote_policy_ref_frame(),
        )
        return self.ensure_action_supported(action)

    def request_remote_policy_action(
        self,
        frame: Frame | Mapping[str, Any] | None = None,
    ) -> Action:
        """Backward-compatible alias for embodia's internal remote action path."""

        return self._request_remote_policy_action(frame)

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

        return self._get_runtime_spec()

    def observe(self) -> Frame:
        """Return a normalized, validated frame from the wrapped class."""

        raw_observe = self._resolve_impl("observe", "_observe_impl")
        frame = self.validate_frame(raw_observe())
        spec = self._get_runtime_spec()
        images.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.image_keys,
        )
        state.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.state_keys,
        )
        return frame

    def act(self, action: Action | Mapping[str, Any]) -> None:
        """Normalize and validate an action before forwarding it."""

        raw_act = self._resolve_impl("act", "_act_impl")
        normalized_action = self.validate_action(action)
        spec = self._get_runtime_spec()
        action_modes.ensure_supported(
            normalized_action,
            spec.action_modes,
            owner_label="robot",
            owner_name=spec.name,
        )
        raw_act(self.to_native_action(normalized_action))

    def reset(self) -> Frame:
        """Return a normalized, validated reset frame from the wrapped class."""

        raw_reset = self._resolve_impl("reset", "_reset_impl")
        frame = self.validate_frame(raw_reset())
        spec = self._get_runtime_spec()
        images.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.image_keys,
        )
        state.ensure_frame_keys(
            frame,
            owner_label="robot",
            required_keys=spec.state_keys,
        )
        return frame


__all__ = ["RobotMixin"]
