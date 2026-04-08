"""Optional remote-policy helpers.

This module keeps embodia's core package free of websocket and msgpack
dependencies, while still making it easy to talk to a remote policy server
from local Python code.

Payload conversion lives in ``embodia.contrib.remote_transform`` so this
module can stay focused on transport, client/server lifecycle, and serving.

The websocket client intentionally follows one compact policy websocket
protocol:

- connect over websocket
- receive one metadata message immediately after connect
- send observation dictionaries encoded with msgpack
- receive inference results as msgpack dictionaries
- expose ``infer()``, ``reset()``, and ``get_server_metadata()``

The remote server can send NumPy arrays in a custom msgpack encoding. embodia's
core runtime is numpy-based, so this module decodes that payload back into
``numpy.ndarray`` objects before it re-enters the shared schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import http
import importlib
import importlib.util
import time
import traceback
from typing import Any

from ..core.arraylike import (
    numpy_ndarray_type,
    torch_tensor_type,
)
from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from ..core.transform import (
    coerce_action,
    coerce_frame,
    coerce_policy_spec,
    frame_to_dict,
    policy_spec_to_dict,
)
from ..runtime.shared.dispatch import (
    POLICY_GET_SPEC_METHODS,
    POLICY_INFER_METHODS,
    POLICY_RESET_METHODS,
    format_method_options,
    resolve_callable_method,
)
from ..runtime.checks import validate_action, validate_frame
from .remote_transform import (
    RemoteTransform,
    _coerce_embodia_action_plan,
    actions_to_action_plan,
    first_action_from_response,
    response_from_action_plan,
)


def _has_top_level_module(module_name: str) -> bool:
    """Return whether a top-level module can be discovered."""

    return importlib.util.find_spec(module_name) is not None


def _has_nested_module(module_name: str) -> bool:
    """Return whether a nested module can be imported."""

    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def is_remote_available() -> bool:
    """Return whether the minimal websocket/msgpack client deps are available."""

    return _has_top_level_module("msgpack") and _has_nested_module(
        "websockets.sync.client"
    )


def require_remote() -> None:
    """Raise a clear error if the minimal remote-policy deps are missing."""

    if not is_remote_available():
        raise InterfaceValidationError(
            "Remote policy support requires the optional 'msgpack' and "
            "'websockets' packages. Install them with "
            "`pip install 'embodia[remote]'`."
        )


def _import_msgpack() -> Any:
    """Import ``msgpack`` lazily."""

    try:
        return importlib.import_module("msgpack")
    except ImportError as exc:
        raise InterfaceValidationError(
            "Remote policy support requires the optional 'msgpack' "
            "package. Install it with `pip install 'embodia[remote]'`."
        ) from exc


def _import_websocket_client() -> Any:
    """Import ``websockets.sync.client`` lazily."""

    try:
        return importlib.import_module("websockets.sync.client")
    except ImportError as exc:
        raise InterfaceValidationError(
            "Remote policy support requires the optional 'websockets' "
            "package with sync client support. Install it with "
            "`pip install 'embodia[remote]'`."
        ) from exc


def _import_websocket_server() -> Any:
    """Import ``websockets.sync.server`` lazily."""

    try:
        return importlib.import_module("websockets.sync.server")
    except ImportError as exc:
        raise InterfaceValidationError(
            "Remote policy serving requires the optional 'websockets' "
            "package with sync server support. Install it with "
            "`pip install 'embodia[remote]'`."
        ) from exc


def _import_numpy() -> Any:
    """Import ``numpy`` lazily for ndarray payloads."""

    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise InterfaceValidationError(
            "This remote message contains NumPy array payloads encoded with the "
            "expected msgpack format, but numpy is not available in the current "
            "environment."
        ) from exc


class RemoteMsgpackCodec:
    """Msgpack codec for embodia's remote ndarray wire format."""

    def __init__(self) -> None:
        self._packer: Any | None = None

    def _pack_array(self, obj: object) -> object:
        """Encode optional ndarray/tensor values in the remote msgpack format."""

        numpy_type = numpy_ndarray_type()
        if numpy_type is not None and isinstance(obj, numpy_type):
            if obj.dtype.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported dtype: {obj.dtype}")
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }

        if numpy_type is not None:
            numpy_module = importlib.import_module("numpy")
            if isinstance(obj, numpy_module.generic):
                return {
                    b"__npgeneric__": True,
                    b"data": obj.item(),
                    b"dtype": obj.dtype.str,
                }

        torch_type = torch_tensor_type()
        if torch_type is not None and isinstance(obj, torch_type):
            tensor = obj.detach() if callable(getattr(obj, "detach", None)) else obj
            device = getattr(getattr(tensor, "device", None), "type", None)
            if device not in (None, "cpu"):
                cpu = getattr(tensor, "cpu", None)
                if not callable(cpu):
                    raise InterfaceValidationError(
                        "torch tensor payload is not on CPU and does not expose "
                        "cpu()."
                    )
                tensor = cpu()

            to_numpy = getattr(tensor, "numpy", None)
            if callable(to_numpy) and numpy_type is not None:
                return self._pack_array(to_numpy())

            tolist = getattr(tensor, "tolist", None)
            if callable(tolist):
                return tolist()

        return obj

    def _unpack_array(self, obj: dict[Any, Any]) -> Any:
        """Decode ndarray markers from one msgpack object hook."""

        if b"__ndarray__" in obj:
            numpy_module = _import_numpy()
            return numpy_module.ndarray(
                buffer=obj[b"data"],
                dtype=numpy_module.dtype(obj[b"dtype"]),
                shape=obj[b"shape"],
            )

        if b"__npgeneric__" in obj:
            numpy_module = _import_numpy()
            return numpy_module.dtype(obj[b"dtype"]).type(obj[b"data"])

        return obj

    def _ensure_packer(self) -> Any:
        """Build and cache a msgpack packer."""

        if self._packer is None:
            msgpack = _import_msgpack()
            self._packer = msgpack.Packer(default=self._pack_array)
        return self._packer

    def packb(self, obj: object) -> bytes:
        """Serialize one remote request payload."""

        return self._ensure_packer().pack(obj)

    def unpackb(self, data: bytes) -> Any:
        """Deserialize one remote response payload."""

        msgpack = _import_msgpack()
        return msgpack.unpackb(data, object_hook=self._unpack_array)


def _build_websocket_uri(host: str, port: int | None) -> str:
    """Build a websocket URI for the current remote client."""

    if host.startswith("ws"):
        uri = host
    else:
        uri = f"ws://{host}"

    if port is not None:
        uri += f":{port}"
    return uri


def _connect_websocket(uri: str, headers: Mapping[str, str] | None) -> Any:
    """Open one websocket connection."""

    websocket_client = _import_websocket_client()
    return websocket_client.connect(
        uri,
        compression=None,
        max_size=None,
        additional_headers=headers,
    )


def _is_connection_closed_error(exc: Exception) -> bool:
    """Return whether an exception represents a closed websocket."""

    if type(exc).__name__ == "ConnectionClosed":
        return True

    try:
        websocket_exceptions = importlib.import_module("websockets.exceptions")
    except ImportError:
        return False

    connection_closed = getattr(websocket_exceptions, "ConnectionClosed", None)
    return (
        isinstance(connection_closed, type)
        and isinstance(exc, connection_closed)
    )


def _close_with_internal_error(websocket: object, reason: str) -> None:
    """Best-effort close for server-side internal errors."""

    close = getattr(websocket, "close", None)
    if not callable(close):
        return

    try:
        close(code=1011, reason=reason)
    except TypeError:
        close()


def _request_header(request: object, header_name: str) -> str | None:
    """Return one HTTP header from a websocket request-like object."""

    headers = getattr(request, "headers", None)
    if headers is None:
        return None

    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(header_name)
        return None if value is None else str(value)

    if isinstance(headers, Mapping):
        value = headers.get(header_name)
        return None if value is None else str(value)

    return None


class EmbodiaPolicyAdapter:
    """Expose an embodia-style policy through the remote websocket protocol.

    This adapter keeps the policy itself simple: the policy still only receives a
    normalized frame and returns one action. Chunking, alternate response
    shapes, or native observation conversion stay outside the policy and can be
    swapped in here.
    """

    def __init__(
        self,
        policy: object,
        *,
        obs_to_frame: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]] | None = None,
        action_plan_provider: Callable[[object, Frame], Any] | None = None,
        response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None,
        server_metadata: Mapping[str, Any] | None = None,
        reset_policy_on_connect: bool = False,
        transform: RemoteTransform | None = None,
    ) -> None:
        self.policy = policy
        if transform is not None and (
            obs_to_frame is not None or response_builder is not None
        ):
            raise InterfaceValidationError(
                "EmbodiaPolicyAdapter accepts either transform=... or "
                "obs_to_frame=/response_builder=..., not both."
            )

        self._transform = transform
        if transform is not None:
            build_frame = getattr(transform, "build_frame", None)
            transform_response_builder = getattr(
                transform,
                "response_from_action_plan",
                None,
            )
            if not callable(build_frame) or not callable(transform_response_builder):
                raise InterfaceValidationError(
                    "transform must expose build_frame(obs) and "
                    "response_from_action_plan(plan, frame=...)."
                )
            self._obs_to_frame = build_frame
            self._response_builder = (
                lambda actions, frame: transform_response_builder(
                    actions,
                    frame=frame,
                )
            )
        else:
            self._obs_to_frame = obs_to_frame or (lambda obs: coerce_frame(obs))
            self._response_builder = (
                response_builder
                or (lambda actions, frame: response_from_action_plan(actions))
            )
        self._action_plan_provider = action_plan_provider
        self._server_metadata = (
            dict(server_metadata) if server_metadata is not None else {}
        )
        self._reset_policy_on_connect = reset_policy_on_connect

    def _coerce_frame(self, obs: Mapping[str, Any]) -> Frame:
        """Convert one raw remote observation into a validated frame."""

        if not isinstance(obs, Mapping):
            raise InterfaceValidationError(
                f"remote observation must be a mapping, got {type(obs).__name__}."
            )
        frame = coerce_frame(self._obs_to_frame(obs))
        validate_frame(frame)
        return frame

    def get_server_metadata(self) -> dict[str, Any]:
        """Return default metadata for the websocket handshake."""

        metadata = dict(self._server_metadata)
        embodia_meta = metadata.setdefault("embodia", {})
        if not isinstance(embodia_meta, dict):
            raise InterfaceValidationError(
                "server_metadata['embodia'] must be a mapping when provided."
            )

        get_spec, _ = resolve_callable_method(self.policy, POLICY_GET_SPEC_METHODS)
        if callable(get_spec):
            embodia_meta.setdefault("policy_spec", policy_spec_to_dict(get_spec()))

        return metadata

    def on_connect(self) -> None:
        """Optional lifecycle hook used by :class:`WebsocketPolicyServer`."""

        if self._reset_policy_on_connect:
            self.reset()

    def reset(self) -> None:
        """Reset the wrapped policy when it exposes ``reset()``."""

        reset, _ = resolve_callable_method(self.policy, POLICY_RESET_METHODS)
        if callable(reset):
            reset()

    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Convert remote observations, run the policy, and build a response."""

        frame = self._coerce_frame(obs)
        if self._action_plan_provider is not None:
            raw_plan = self._action_plan_provider(self.policy, frame)
        else:
            infer, _ = resolve_callable_method(self.policy, POLICY_INFER_METHODS)
            if not callable(infer):
                raise InterfaceValidationError(
                    f"{type(self.policy).__name__} must expose "
                    f"{format_method_options(POLICY_INFER_METHODS)} to be "
                    "served through EmbodiaPolicyAdapter."
                )
            raw_plan = infer(frame)

        actions = _coerce_embodia_action_plan(raw_plan)
        response = self._response_builder(actions, frame)
        if not isinstance(response, Mapping):
            raise InterfaceValidationError(
                "response_builder must return a mapping, got "
                f"{type(response).__name__}."
            )
        return dict(response)


def build_policy_adapter(
    policy: object,
    *,
    obs_to_frame: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]] | None = None,
    action_plan_provider: Callable[[object, Frame], Any] | None = None,
    response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None,
    server_metadata: Mapping[str, Any] | None = None,
    reset_policy_on_connect: bool = False,
    transform: RemoteTransform | None = None,
) -> EmbodiaPolicyAdapter:
    """Build one remote adapter around a policy-like object."""

    if transform is not None and (
        obs_to_frame is not None or response_builder is not None
    ):
        raise InterfaceValidationError(
            "build_policy_adapter() accepts either transform=... or "
            "obs_to_frame=/response_builder=..., not both."
        )

    return EmbodiaPolicyAdapter(
        policy,
        obs_to_frame=obs_to_frame,
        action_plan_provider=action_plan_provider,
        response_builder=response_builder,
        server_metadata=server_metadata,
        reset_policy_on_connect=reset_policy_on_connect,
        transform=transform,
    )


def build_policy_server(
    policy: object,
    *,
    obs_to_frame: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]] | None = None,
    host: str = "0.0.0.0",
    port: int | None = None,
    api_key: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_server_timing: bool = True,
    action_plan_provider: Callable[[object, Frame], Any] | None = None,
    response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None,
    server_metadata: Mapping[str, Any] | None = None,
    reset_policy_on_connect: bool = False,
    transform: RemoteTransform | None = None,
) -> "WebsocketPolicyServer":
    """Build one websocket server around a policy-like object."""

    adapter = build_policy_adapter(
        policy,
        obs_to_frame=obs_to_frame,
        action_plan_provider=action_plan_provider,
        response_builder=response_builder,
        server_metadata=server_metadata,
        reset_policy_on_connect=reset_policy_on_connect,
        transform=transform,
    )
    return WebsocketPolicyServer(
        adapter,
        host=host,
        port=port,
        metadata=metadata if metadata is not None else adapter.get_server_metadata(),
        api_key=api_key,
        include_server_timing=include_server_timing,
    )


def serve_policy(
    policy: object,
    *,
    obs_to_frame: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]] | None = None,
    host: str = "0.0.0.0",
    port: int | None = None,
    api_key: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_server_timing: bool = True,
    action_plan_provider: Callable[[object, Frame], Any] | None = None,
    response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None,
    server_metadata: Mapping[str, Any] | None = None,
    reset_policy_on_connect: bool = False,
    transform: RemoteTransform | None = None,
) -> None:
    """Serve one policy-like object through embodia's remote websocket API."""

    server = build_policy_server(
        policy,
        obs_to_frame=obs_to_frame,
        host=host,
        port=port,
        api_key=api_key,
        metadata=metadata,
        include_server_timing=include_server_timing,
        action_plan_provider=action_plan_provider,
        response_builder=response_builder,
        server_metadata=server_metadata,
        reset_policy_on_connect=reset_policy_on_connect,
        transform=transform,
    )
    server.serve_forever()


class RemotePolicy:
    """Policy-like source that talks to embodia's remote policy server.

    This keeps `robot` local-only. The remote deployment boundary lives on the
    action-source side, so users can pass this object straight into
    ``run_step(robot, source=...)`` or ``InferenceRuntime.step(...)``.
    """

    def __init__(
        self,
        *,
        runner: object | None = None,
        host: str = "localhost",
        port: int | None = None,
        api_key: str | None = None,
        retry_interval_s: float = 5.0,
        connect_timeout_s: float | None = None,
        additional_headers: Mapping[str, str] | None = None,
        connect_immediately: bool = False,
        wait_for_server: bool = True,
        frame_to_obs: Callable[[Frame], Mapping[str, Any]] | None = None,
        response_to_action: Callable[[object], Action | Mapping[str, Any]] | None = None,
        action_target: str | None = None,
        command_kind: str | None = None,
        ref_frame: str | None = None,
        policy_spec: object | None = None,
        transform: RemoteTransform | None = None,
        openpi: bool = False,
        openpi_transform: object | None = None,
        enabled: bool = True,
    ) -> None:
        if not openpi and openpi_transform is not None:
            raise InterfaceValidationError(
                "RemotePolicy(openpi_transform=...) requires openpi=True."
            )
        if openpi and (
            transform is not None
            or frame_to_obs is not None
            or response_to_action is not None
            or command_kind is not None
            or ref_frame is not None
            or action_target is not None
        ):
            raise InterfaceValidationError(
                "RemotePolicy(openpi=True) only accepts connection parameters "
                "and optional policy_spec=/openpi_transform=.... Do not combine it with "
                "transform=..., frame_to_obs=..., response_to_action=..., "
                "action_target=..., command_kind=..., or ref_frame=...."
            )
        if transform is not None and (
            frame_to_obs is not None
            or response_to_action is not None
            or command_kind is not None
            or ref_frame is not None
            or action_target is not None
        ):
            raise InterfaceValidationError(
                "RemotePolicy accepts either transform=... or "
                "frame_to_obs=/response_to_action=/action_target=/command_kind=/ref_frame, "
                "not both."
            )

        self._runner = (
            runner
            if runner is not None
            else RemotePolicyRunner(
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
        )
        infer = getattr(self._runner, "infer", None)
        if not callable(infer):
            raise InterfaceValidationError(
                "RemotePolicy runner must expose infer(obs)."
            )

        self._policy_spec = (
            None if policy_spec is None else coerce_policy_spec(policy_spec)
        )
        self._action_target = action_target
        self._command_kind = command_kind
        self._ref_frame = ref_frame
        self._openpi = openpi
        self._openpi_adapter: object | None = None

        if openpi:
            from . import openpi as em_openpi

            if openpi_transform is not None:
                self._openpi_adapter = openpi_transform
            else:
                self._openpi_adapter = em_openpi.OpenPITransform()
            build_obs = getattr(self._openpi_adapter, "build_obs", None)
            action_plan_from_response = getattr(
                self._openpi_adapter,
                "action_plan_from_response",
                None,
            )
            first_action_from_response = getattr(
                self._openpi_adapter,
                "first_action_from_response",
                None,
            )
            if (
                not callable(build_obs)
                or not callable(action_plan_from_response)
                or not callable(first_action_from_response)
            ):
                raise InterfaceValidationError(
                    "RemotePolicy(openpi=True, openpi_transform=...) requires "
                    "an object exposing build_obs(frame), "
                    "action_plan_from_response(response), and "
                    "first_action_from_response(response)."
                )
            self._frame_to_obs = self._openpi_adapter.build_obs
            self._response_to_action_plan = self._openpi_adapter.action_plan_from_response
            self._response_to_action = self._openpi_adapter.first_action_from_response
        elif transform is not None:
            self._frame_to_obs = transform.build_obs
            self._response_to_action_plan = transform.action_plan_from_response
            self._response_to_action = transform.first_action_from_response
        else:
            self._frame_to_obs = frame_to_obs or frame_to_dict
            if response_to_action is not None:
                self._response_to_action = response_to_action
                self._response_to_action_plan = (
                    lambda response, _response_to_action=response_to_action: [
                        coerce_action(_response_to_action(response))
                    ]
                )
            else:
                self._response_to_action_plan = self._default_response_to_action_plan
                self._response_to_action = self._default_response_to_action

    def _default_response_to_action(self, response: object) -> Action:
        """Convert one remote response into a validated embodia action."""

        return self._default_response_to_action_plan(response)[0]

    def _default_response_to_action_plan(self, response: object) -> list[Action]:
        """Convert one remote response into a validated embodia action plan."""

        if isinstance(response, Mapping):
            try:
                action = coerce_action(response)
                validate_action(action)
                return [action]
            except InterfaceValidationError:
                pass

        target, command_kind, ref_frame = self._resolve_action_shape(response)
        plan = actions_to_action_plan(
            response,
            target=target,
            kind=command_kind,
            ref_frame=ref_frame,
        )
        for action in plan:
            validate_action(action)
        return plan

    def _resolve_action_shape(
        self,
        response: object,
    ) -> tuple[str, str, str | None]:
        """Resolve target/kind hints for one remote action response."""

        target = self._action_target
        command_kind = self._command_kind
        ref_frame = self._ref_frame

        if isinstance(response, Mapping):
            embodia_meta = response.get("embodia")
            if isinstance(embodia_meta, Mapping):
                if target is None:
                    raw_target = embodia_meta.get("action_target")
                    if isinstance(raw_target, str) and raw_target.strip():
                        target = raw_target
                if command_kind is None:
                    raw_kind = embodia_meta.get("action_kind")
                    if isinstance(raw_kind, str) and raw_kind.strip():
                        command_kind = raw_kind
                if ref_frame is None:
                    raw_ref_frame = embodia_meta.get("action_ref_frame")
                    if raw_ref_frame is not None:
                        if (
                            not isinstance(raw_ref_frame, str)
                            or not raw_ref_frame.strip()
                        ):
                            raise InterfaceValidationError(
                                "remote response embodia.action_ref_frame "
                                "must be a non-empty string when provided."
                            )
                        ref_frame = raw_ref_frame

        if target is None or command_kind is None:
            try:
                spec = self.get_spec()
            except InterfaceValidationError:
                spec = None
            if spec is not None:
                outputs = getattr(spec, "outputs", None)
                if isinstance(outputs, list) and len(outputs) == 1:
                    output = outputs[0]
                    if target is None:
                        target = output.target
                    if command_kind is None:
                        command_kind = output.command_kind
                    if ref_frame is None:
                        output_meta = getattr(output, "meta", None)
                        if isinstance(output_meta, Mapping):
                            raw_ref_frame = output_meta.get("ref_frame")
                            if isinstance(raw_ref_frame, str) and raw_ref_frame.strip():
                                ref_frame = raw_ref_frame

        if target is None and command_kind is not None:
            target = "arm"

        if target is None or command_kind is None:
            raise InterfaceValidationError(
                "RemotePolicy could not infer how to decode the remote action "
                "payload. Return embodia action metadata from the server, "
                "expose a single-output policy_spec through remote metadata, "
                "or pass response_to_action=... for a custom wire format."
            )
        return target, command_kind, ref_frame

    def infer(self, frame: Frame | Mapping[str, Any]) -> Action:
        """Run one remote inference step and return a normalized action."""

        normalized_frame = coerce_frame(frame)
        validate_frame(normalized_frame)

        obs = self._frame_to_obs(normalized_frame)
        if not isinstance(obs, Mapping):
            raise InterfaceValidationError(
                "RemotePolicy frame_to_obs must return a mapping, got "
                f"{type(obs).__name__}."
            )

        response = getattr(self._runner, "infer")(dict(obs))
        plan = self._response_to_action_plan(response)
        if not isinstance(plan, list) or not plan:
            raise InterfaceValidationError(
                "RemotePolicy response decoder must return a non-empty list[Action]."
            )
        action = coerce_action(plan[0])
        validate_action(action)
        return action

    def infer_chunk(
        self,
        frame: Frame | Mapping[str, Any],
        request: object,
    ) -> list[Action]:
        """Run one remote inference step and return the full action chunk."""

        del request
        normalized_frame = coerce_frame(frame)
        validate_frame(normalized_frame)

        obs = self._frame_to_obs(normalized_frame)
        if not isinstance(obs, Mapping):
            raise InterfaceValidationError(
                "RemotePolicy frame_to_obs must return a mapping, got "
                f"{type(obs).__name__}."
            )

        response = getattr(self._runner, "infer")(dict(obs))
        plan = self._response_to_action_plan(response)
        if not isinstance(plan, list) or not plan:
            raise InterfaceValidationError(
                "RemotePolicy response decoder must return a non-empty list[Action]."
            )
        for index, action in enumerate(plan):
            coerced = coerce_action(action)
            validate_action(coerced)
            plan[index] = coerced
        return plan

    def embodia_bind_robot(self, robot: object) -> None:
        """Optional internal hook used to configure source-side adapters."""

        if self._openpi_adapter is None:
            return

        bind_robot = getattr(self._openpi_adapter, "bind_robot", None)
        if callable(bind_robot):
            bind_robot(robot)
        if self._policy_spec is None:
            get_policy_spec = getattr(self._openpi_adapter, "get_policy_spec", None)
            if callable(get_policy_spec):
                self._policy_spec = coerce_policy_spec(get_policy_spec())

    def reset(self) -> None:
        """Forward reset to the underlying remote runner when supported."""

        reset = getattr(self._runner, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        """Close the underlying remote runner when supported."""

        close = getattr(self._runner, "close", None)
        if callable(close):
            close()

    def get_server_metadata(self) -> dict[str, Any]:
        """Return metadata exposed by the remote server when available."""

        get_server_metadata = getattr(self._runner, "get_server_metadata", None)
        if not callable(get_server_metadata):
            return {}

        metadata = get_server_metadata()
        if not isinstance(metadata, Mapping):
            raise InterfaceValidationError(
                "RemotePolicy runner get_server_metadata() must return a "
                f"mapping, got {type(metadata).__name__}."
            )
        return dict(metadata)

    def get_spec(self) -> Any:
        """Return a cached policy spec or read it from remote metadata."""

        if self._policy_spec is not None:
            return self._policy_spec

        if self._openpi_adapter is not None:
            get_policy_spec = getattr(self._openpi_adapter, "get_policy_spec", None)
            if callable(get_policy_spec):
                try:
                    self._policy_spec = coerce_policy_spec(get_policy_spec())
                    return self._policy_spec
                except InterfaceValidationError:
                    pass

        metadata = self.get_server_metadata()
        embodia_metadata = metadata.get("embodia")
        if not isinstance(embodia_metadata, Mapping):
            raise InterfaceValidationError(
                "remote server metadata does not contain embodia.policy_spec."
            )

        raw_spec = embodia_metadata.get("policy_spec")
        if raw_spec is None:
            raise InterfaceValidationError(
                "remote server metadata does not contain embodia.policy_spec."
            )

        self._policy_spec = coerce_policy_spec(raw_spec)
        return self._policy_spec


class WebsocketPolicyServer:
    """Websocket server for lightweight remote inference.

    This version intentionally stays small and embeddable:

    - sync server API instead of forcing asyncio in user code
    - optional API-key check
    - same metadata handshake and msgpack payload protocol
    - works with any object exposing ``infer(obs) -> mapping``
    """

    def __init__(
        self,
        policy: object,
        *,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        include_server_timing: bool = True,
        codec: RemoteMsgpackCodec | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = None if metadata is None else dict(metadata)
        self._api_key = api_key
        self._include_server_timing = include_server_timing
        self._codec = codec or RemoteMsgpackCodec()

    def _resolve_metadata(self) -> dict[str, Any]:
        """Return the metadata sent immediately after connect."""

        if self._metadata is not None:
            return dict(self._metadata)

        get_server_metadata = getattr(self._policy, "get_server_metadata", None)
        if callable(get_server_metadata):
            metadata = get_server_metadata()
            if not isinstance(metadata, Mapping):
                raise InterfaceValidationError(
                    "policy get_server_metadata() must return a mapping, got "
                    f"{type(metadata).__name__}."
                )
            return dict(metadata)

        return {}

    def _process_request(self, connection: object, request: object) -> Any | None:
        """Handle health checks and optional API-key validation."""

        if getattr(request, "path", None) == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")

        if self._api_key is None:
            return None

        expected = f"Api-Key {self._api_key}"
        received = _request_header(request, "Authorization")
        if received == expected:
            return None
        return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")

    def _handle_connection(self, websocket: object) -> None:
        """Serve one websocket connection until it closes."""

        on_connect = getattr(self._policy, "on_connect", None)
        if callable(on_connect):
            on_connect()

        websocket.send(self._codec.packb(self._resolve_metadata()))
        prev_total_time_s: float | None = None

        while True:
            try:
                loop_start = time.monotonic()
                message = websocket.recv()
                if isinstance(message, str):
                    raise RuntimeError(
                        "Expected a binary msgpack frame from the client, "
                        "but received text."
                    )

                obs = self._codec.unpackb(message)
                if not isinstance(obs, Mapping):
                    raise InterfaceValidationError(
                        "websocket policy server expects observation mappings, "
                        f"got {type(obs).__name__}."
                    )

                infer_start = time.monotonic()
                response = getattr(self._policy, "infer")(obs)
                infer_time_s = time.monotonic() - infer_start

                if not isinstance(response, Mapping):
                    raise InterfaceValidationError(
                        "policy infer() must return a mapping, got "
                        f"{type(response).__name__}."
                    )

                packed_response = dict(response)
                if self._include_server_timing:
                    server_timing = {}
                    existing_timing = packed_response.get("server_timing")
                    if isinstance(existing_timing, Mapping):
                        server_timing.update(existing_timing)
                    server_timing["infer_ms"] = infer_time_s * 1000.0
                    if prev_total_time_s is not None:
                        server_timing["prev_total_ms"] = prev_total_time_s * 1000.0
                    packed_response["server_timing"] = server_timing

                websocket.send(self._codec.packb(packed_response))
                prev_total_time_s = time.monotonic() - loop_start
            except Exception as exc:
                if _is_connection_closed_error(exc):
                    break

                send = getattr(websocket, "send", None)
                if callable(send):
                    try:
                        send(traceback.format_exc())
                    except Exception:
                        pass
                _close_with_internal_error(
                    websocket,
                    "Internal server error. Traceback included in previous frame.",
                )
                raise

    def serve(self) -> Any:
        """Return the underlying websocket server context manager."""

        websocket_server = _import_websocket_server()
        return websocket_server.serve(
            self._handle_connection,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        )

    def serve_forever(self) -> None:
        """Block and serve websocket requests forever."""

        with self.serve() as server:
            server.serve_forever()


class WebsocketClientPolicy:
    """Thin websocket client for remote policy inference."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        *,
        retry_interval_s: float = 5.0,
        connect_timeout_s: float | None = None,
        additional_headers: Mapping[str, str] | None = None,
        connect_immediately: bool = True,
        wait_for_server: bool = True,
        codec: RemoteMsgpackCodec | None = None,
    ) -> None:
        if retry_interval_s <= 0.0:
            raise InterfaceValidationError(
                "retry_interval_s must be > 0."
            )
        if connect_timeout_s is not None and connect_timeout_s <= 0.0:
            raise InterfaceValidationError(
                "connect_timeout_s must be > 0 when provided."
            )

        self._uri = _build_websocket_uri(host, port)
        self._api_key = api_key
        self._retry_interval_s = retry_interval_s
        self._connect_timeout_s = connect_timeout_s
        self._additional_headers = (
            dict(additional_headers) if additional_headers is not None else {}
        )
        self._codec = codec or RemoteMsgpackCodec()
        self._ws: Any | None = None
        self._server_metadata: dict[str, Any] | None = None

        if connect_immediately:
            self.connect(wait_for_server=wait_for_server)

    @property
    def uri(self) -> str:
        """Return the websocket URI used by this client."""

        return self._uri

    @property
    def is_connected(self) -> bool:
        """Return whether the websocket connection has been established."""

        return self._ws is not None

    def _build_headers(self) -> Mapping[str, str] | None:
        """Build request headers for the websocket connect call."""

        headers = dict(self._additional_headers)
        if self._api_key is not None:
            headers.setdefault("Authorization", f"Api-Key {self._api_key}")
        return headers or None

    def connect(self, *, wait_for_server: bool = True) -> dict[str, Any]:
        """Connect to the remote policy server and read metadata."""

        deadline = (
            None
            if self._connect_timeout_s is None
            else time.monotonic() + self._connect_timeout_s
        )

        while True:
            try:
                connection = _connect_websocket(
                    self._uri,
                    headers=self._build_headers(),
                )
                metadata_message = connection.recv()
                if isinstance(metadata_message, str):
                    raise RuntimeError(
                        "remote server sent a text frame during metadata "
                        f"handshake:\n{metadata_message}"
                    )

                metadata = self._codec.unpackb(metadata_message)
                if not isinstance(metadata, dict):
                    raise InterfaceValidationError(
                        "remote server metadata must be a dict, got "
                        f"{type(metadata).__name__}."
                    )

                self._ws = connection
                self._server_metadata = metadata
                return dict(metadata)
            except (ConnectionRefusedError, OSError) as exc:
                if not wait_for_server:
                    raise
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for remote policy server at {self._uri}."
                    ) from exc
                time.sleep(self._retry_interval_s)

    def get_server_metadata(self) -> dict[str, Any]:
        """Return server metadata received during the websocket handshake."""

        if self._server_metadata is None:
            raise InterfaceValidationError(
                "remote server metadata is not available before connect(). "
                "Call connect() or infer() first."
            )
        return dict(self._server_metadata)

    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Send one observation dict and return the server response dict."""

        if not isinstance(obs, Mapping):
            raise InterfaceValidationError(
                f"obs must be a mapping, got {type(obs).__name__}."
            )

        if self._ws is None:
            self.connect(wait_for_server=True)

        assert self._ws is not None
        self._ws.send(self._codec.packb(dict(obs)))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")

        unpacked = self._codec.unpackb(response)
        if not isinstance(unpacked, dict):
            raise InterfaceValidationError(
                "remote inference response must be a dict, got "
                f"{type(unpacked).__name__}."
            )
        return unpacked

    def reset(self) -> None:
        """Keep API symmetry with local sources.

        The current remote websocket protocol exposes inference only, so reset
        is intentionally a no-op on the client side.
        """

    def close(self) -> None:
        """Close the websocket connection if it exists."""

        if self._ws is None:
            return

        close = getattr(self._ws, "close", None)
        if callable(close):
            close()
        self._ws = None

    def __enter__(self) -> "WebsocketClientPolicy":
        """Return ``self`` for context-manager use."""

        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the websocket on context-manager exit."""

        del exc_info
        self.close()


class RemotePolicyRunner:
    """Small lazy wrapper that hides websocket-client lifecycle details.

    The goal is to keep policy-source code from directly managing a websocket
    client. Higher-level code can hold one local robot plus one remote policy
    source without leaking transport details into the robot class.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        retry_interval_s: float = 5.0,
        connect_timeout_s: float | None = None,
        additional_headers: Mapping[str, str] | None = None,
        connect_immediately: bool = False,
        wait_for_server: bool = True,
        client_factory: Callable[..., WebsocketClientPolicy] | None = None,
    ) -> None:
        self.enabled = enabled
        self.host = host
        self.port = port
        self.api_key = api_key
        self.retry_interval_s = retry_interval_s
        self.connect_timeout_s = connect_timeout_s
        self.additional_headers = (
            None if additional_headers is None else dict(additional_headers)
        )
        self.connect_immediately = connect_immediately
        self.wait_for_server = wait_for_server
        self._client_factory = client_factory or WebsocketClientPolicy
        self._client: WebsocketClientPolicy | None = None

        if self.enabled and self.connect_immediately:
            self._ensure_client()

    @property
    def is_connected(self) -> bool:
        """Return whether the underlying websocket client is connected."""

        return self._client is not None and self._client.is_connected

    def _ensure_client(self) -> WebsocketClientPolicy:
        """Create the websocket client on first use."""

        if not self.enabled:
            raise InterfaceValidationError(
                "Remote policy access is disabled for this policy instance. "
                "Enable it with enabled=True."
            )

        if self._client is None:
            self._client = self._client_factory(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                retry_interval_s=self.retry_interval_s,
                connect_timeout_s=self.connect_timeout_s,
                additional_headers=self.additional_headers,
                connect_immediately=self.connect_immediately,
                wait_for_server=self.wait_for_server,
            )
        return self._client

    def get_server_metadata(self) -> dict[str, Any]:
        """Return metadata from the remote server."""

        return self._ensure_client().get_server_metadata()

    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Run one remote policy inference step."""

        return self._ensure_client().infer(obs)

    def reset(self) -> None:
        """Forward reset when a client has already been created."""

        if self._client is not None:
            self._client.reset()

    def close(self) -> None:
        """Close the underlying websocket client when present."""

        if self._client is not None:
            self._client.close()
            self._client = None

RemoteWebsocketClientPolicy = WebsocketClientPolicy
RemoteWebsocketPolicyServer = WebsocketPolicyServer


__all__ = [
    "EmbodiaPolicyAdapter",
    "RemoteTransform",
    "RemotePolicy",
    "RemoteMsgpackCodec",
    "RemotePolicyRunner",
    "build_policy_adapter",
    "build_policy_server",
    "RemoteWebsocketClientPolicy",
    "RemoteWebsocketPolicyServer",
    "serve_policy",
    "WebsocketClientPolicy",
    "WebsocketPolicyServer",
    "is_remote_available",
    "actions_to_action_plan",
    "first_action_from_response",
    "response_from_action_plan",
    "require_remote",
]
