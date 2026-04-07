"""Optional OpenPI remote-policy helpers.

This module keeps embodia's core package free of websocket and msgpack
dependencies, while still making it easy to talk to an OpenPI policy server
from robot-side Python code.

OpenPI payload conversion lives in ``embodia.contrib.openpi_transform`` so this
module can stay focused on transport, client/server lifecycle, and serving.

The websocket client intentionally follows the same wire protocol and close-to-
the-same API as OpenPI's official ``openpi_client.websocket_client_policy``:

- connect over websocket
- receive one metadata message immediately after connect
- send observation dictionaries encoded with msgpack
- receive inference results as msgpack dictionaries
- expose ``infer()``, ``reset()``, and ``get_server_metadata()``

OpenPI's server sends NumPy arrays in a custom msgpack encoding. When the user
environment already has ``numpy`` installed, this module decodes that payload
back into ``numpy.ndarray`` objects. Without ``numpy``, pure Python payloads
still work, but official ndarray payloads require installing numpy in the robot
environment.
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
from ..core.transform import coerce_frame, policy_spec_to_dict
from ..runtime.dispatch import (
    POLICY_GET_SPEC_METHODS,
    POLICY_INFER_METHODS,
    POLICY_RESET_METHODS,
    ROBOT_GET_SPEC_METHODS,
    format_method_options,
    resolve_callable_method,
)
from ..runtime.checks import validate_frame
from .openpi_transform import (
    OpenPITransform,
    _coerce_embodia_action_plan,
    openpi_actions_to_action_plan,
    openpi_first_action,
    openpi_response_from_action_plan,
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


def is_openpi_remote_available() -> bool:
    """Return whether the minimal websocket/msgpack client deps are available."""

    return _has_top_level_module("msgpack") and _has_nested_module(
        "websockets.sync.client"
    )


def is_official_openpi_client_available() -> bool:
    """Return whether the official ``openpi-client`` package is importable."""

    return _has_top_level_module("openpi_client")


def require_openpi_remote() -> None:
    """Raise a clear error if the minimal remote-policy deps are missing."""

    if not is_openpi_remote_available():
        raise InterfaceValidationError(
            "OpenPI remote policy support requires the optional 'msgpack' and "
            "'websockets' packages. Install them with "
            "`pip install 'embodia[openpi-remote]'`."
        )


def _import_msgpack() -> Any:
    """Import ``msgpack`` lazily."""

    try:
        return importlib.import_module("msgpack")
    except ImportError as exc:
        raise InterfaceValidationError(
            "OpenPI remote policy support requires the optional 'msgpack' "
            "package. Install it with `pip install 'embodia[openpi-remote]'`."
        ) from exc


def _import_websocket_client() -> Any:
    """Import ``websockets.sync.client`` lazily."""

    try:
        return importlib.import_module("websockets.sync.client")
    except ImportError as exc:
        raise InterfaceValidationError(
            "OpenPI remote policy support requires the optional 'websockets' "
            "package with sync client support. Install it with "
            "`pip install 'embodia[openpi-remote]'`."
        ) from exc


def _import_websocket_server() -> Any:
    """Import ``websockets.sync.server`` lazily."""

    try:
        return importlib.import_module("websockets.sync.server")
    except ImportError as exc:
        raise InterfaceValidationError(
            "OpenPI remote policy serving requires the optional 'websockets' "
            "package with sync server support. Install it with "
            "`pip install 'embodia[openpi-remote]'`."
        ) from exc


def _import_numpy() -> Any:
    """Import ``numpy`` lazily for OpenPI ndarray payloads."""

    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise InterfaceValidationError(
            "This OpenPI message contains NumPy array payloads encoded with the "
            "official msgpack format, but numpy is not installed in the current "
            "environment. Install numpy or the official 'openpi-client' package."
        ) from exc


class OpenPIMsgpackCodec:
    """Msgpack codec compatible with OpenPI's ndarray wire format."""

    def __init__(self) -> None:
        self._packer: Any | None = None

    def _pack_array(self, obj: object) -> object:
        """Encode optional ndarray/tensor values in OpenPI's msgpack format."""

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
        """Decode OpenPI's ndarray markers from one msgpack object hook."""

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
        """Serialize one OpenPI request payload."""

        return self._ensure_packer().pack(obj)

    def unpackb(self, data: bytes) -> Any:
        """Deserialize one OpenPI response payload."""

        msgpack = _import_msgpack()
        return msgpack.unpackb(data, object_hook=self._unpack_array)


def _build_websocket_uri(host: str, port: int | None) -> str:
    """Build a websocket URI compatible with OpenPI's official client."""

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
    """Expose an embodia-style policy through the OpenPI websocket protocol.

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
        transform: OpenPITransform | None = None,
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
            response_from_action_plan = getattr(
                transform,
                "response_from_action_plan",
                None,
            )
            if not callable(build_frame) or not callable(response_from_action_plan):
                raise InterfaceValidationError(
                    "transform must expose build_frame(obs) and "
                    "response_from_action_plan(plan, frame=...)."
                )
            self._obs_to_frame = build_frame
            self._response_builder = (
                lambda actions, frame: response_from_action_plan(
                    actions,
                    frame=frame,
                )
            )
        else:
            self._obs_to_frame = obs_to_frame or (lambda obs: coerce_frame(obs))
            self._response_builder = (
                response_builder
                or (lambda actions, frame: openpi_response_from_action_plan(actions))
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
    transform: OpenPITransform | None = None,
) -> EmbodiaPolicyAdapter:
    """Build one OpenPI-compatible adapter around a policy-like object."""

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
    transform: OpenPITransform | None = None,
) -> "WebsocketPolicyServer":
    """Build one OpenPI-compatible websocket server around a policy-like object."""

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
    transform: OpenPITransform | None = None,
) -> None:
    """Serve one policy-like object through the OpenPI-compatible websocket API."""

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


def configure_robot_remote_policy(
    robot: object,
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
    action_target: str | None = None,
    command_kind: str | None = None,
    dt: float | None = None,
    ref_frame: str | None = None,
    transform: OpenPITransform | None = None,
    enabled: bool = True,
) -> None:
    """Configure one RobotMixin-style object for OpenPI remote inference."""

    configure = getattr(robot, "configure_remote_policy", None)
    if not callable(configure):
        raise InterfaceValidationError(
            f"{type(robot).__name__} must expose configure_remote_policy(...) "
            "to use configure_robot_remote_policy(...)."
        )

    if transform is not None and any(
        value is not None
        for value in (obs_builder, action_target, command_kind, dt, ref_frame)
    ):
        raise InterfaceValidationError(
            "configure_robot_remote_policy() accepts either transform=... or "
            "obs_builder=/action_target=/command_kind=/dt=/ref_frame, not both."
        )

    response_to_action: Callable[[object], Action]
    request_builder: Callable[[Frame], Mapping[str, Any]] | None
    if transform is not None:
        request_builder = transform.build_obs
        response_to_action = transform.first_action_from_response
    else:
        resolved_action_target = action_target
        resolved_command_kind = command_kind
        if resolved_action_target is None or resolved_command_kind is None:
            get_spec, _ = resolve_callable_method(robot, ROBOT_GET_SPEC_METHODS)
            if callable(get_spec):
                spec = get_spec()
                components = getattr(spec, "components", None)
                if isinstance(components, list) and len(components) == 1:
                    component = components[0]
                    if resolved_action_target is None:
                        resolved_action_target = getattr(component, "name", None)
                    supported_command_kinds = getattr(
                        component,
                        "supported_command_kinds",
                        None,
                    )
                    if (
                        resolved_command_kind is None
                        and isinstance(supported_command_kinds, list)
                        and len(supported_command_kinds) == 1
                    ):
                        resolved_command_kind = supported_command_kinds[0]
        if not isinstance(resolved_action_target, str) or not resolved_action_target.strip():
            raise InterfaceValidationError(
                "configure_robot_remote_policy() requires action_target=... when "
                "transform is not provided and the robot spec does not expose "
                "exactly one component."
            )
        if not isinstance(resolved_command_kind, str) or not resolved_command_kind.strip():
            raise InterfaceValidationError(
                "configure_robot_remote_policy() requires command_kind=... when "
                "transform is not provided and the robot spec does not expose "
                "exactly one supported command kind for its only component."
            )
        response_to_action = lambda response: openpi_first_action(
            response,
            target=resolved_action_target,
            kind=resolved_command_kind,
            dt=0.1 if dt is None else dt,
            ref_frame=ref_frame,
        )
        request_builder = obs_builder

    runner = RemotePolicyRunner(
        enabled=enabled,
        host="localhost" if host is None else host,
        port=port,
        api_key=api_key,
        retry_interval_s=5.0 if retry_interval_s is None else retry_interval_s,
        connect_timeout_s=connect_timeout_s,
        additional_headers=additional_headers,
        connect_immediately=False if connect_immediately is None else connect_immediately,
        wait_for_server=True if wait_for_server is None else wait_for_server,
    )

    configure(
        runner=runner,
        request_builder=request_builder,
        response_to_action=response_to_action,
        enabled=enabled,
    )


class WebsocketPolicyServer:
    """OpenPI-compatible websocket server for lightweight remote inference.

    Relative to OpenPI's official server, this version intentionally stays
    smaller and more embeddable:

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
        codec: OpenPIMsgpackCodec | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = None if metadata is None else dict(metadata)
        self._api_key = api_key
        self._include_server_timing = include_server_timing
        self._codec = codec or OpenPIMsgpackCodec()

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
    """Thin OpenPI-compatible websocket client for remote policy inference."""

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
        codec: OpenPIMsgpackCodec | None = None,
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
        self._codec = codec or OpenPIMsgpackCodec()
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
        """Connect to the remote OpenPI policy server and read metadata."""

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
                        "OpenPI server sent a text frame during metadata "
                        f"handshake:\n{metadata_message}"
                    )

                metadata = self._codec.unpackb(metadata_message)
                if not isinstance(metadata, dict):
                    raise InterfaceValidationError(
                        "OpenPI server metadata must be a dict, got "
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
                        f"Timed out waiting for OpenPI server at {self._uri}."
                    ) from exc
                time.sleep(self._retry_interval_s)

    def get_server_metadata(self) -> dict[str, Any]:
        """Return server metadata received during the websocket handshake."""

        if self._server_metadata is None:
            raise InterfaceValidationError(
                "OpenPI server metadata is not available before connect(). "
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
                "OpenPI inference response must be a dict, got "
                f"{type(unpacked).__name__}."
            )
        return unpacked

    def reset(self) -> None:
        """Match the official client API.

        The current OpenPI websocket protocol exposes inference only, so reset is
        intentionally a no-op on the client side.
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

    The goal is to keep robot-side code from directly managing a websocket
    client. A robot class can expose a single boolean like
    ``use_remote_policy`` and delegate the rest to this helper.
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
                "Enable it with use_remote_policy=True."
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


OpenPIWebsocketClientPolicy = WebsocketClientPolicy
OpenPIWebsocketPolicyServer = WebsocketPolicyServer


__all__ = [
    "EmbodiaPolicyAdapter",
    "OpenPITransform",
    "OpenPIMsgpackCodec",
    "RemotePolicyRunner",
    "build_policy_adapter",
    "build_policy_server",
    "configure_robot_remote_policy",
    "OpenPIWebsocketClientPolicy",
    "OpenPIWebsocketPolicyServer",
    "serve_policy",
    "WebsocketClientPolicy",
    "WebsocketPolicyServer",
    "is_official_openpi_client_available",
    "is_openpi_remote_available",
    "openpi_actions_to_action_plan",
    "openpi_first_action",
    "openpi_response_from_action_plan",
    "require_openpi_remote",
]
