"""Optional OpenPI remote-policy helpers.

This module keeps embodia's core package free of websocket and msgpack
dependencies, while still making it easy to talk to an OpenPI policy server
from robot-side Python code.

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

from collections.abc import Callable, Mapping, Sequence
import http
import importlib
import importlib.util
from numbers import Real
import time
import traceback
from typing import Any

from ..core.arraylike import (
    numpy_ndarray_type,
    optional_array_to_list,
    torch_tensor_type,
)
from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from ..core.transform import coerce_action, coerce_frame, model_spec_to_dict
from ..runtime.checks import validate_action, validate_frame


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


def _coerce_optional_python_list(value: object, *, field_name: str) -> object:
    """Convert optional ndarray/tensor inputs into plain Python lists."""

    converted = optional_array_to_list(value, field_name=field_name)
    if converted is not None:
        return converted

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()

    return value


def _ensure_float_list(value: object, *, field_name: str) -> list[float]:
    """Validate and normalize one numeric action vector."""

    value = _coerce_optional_python_list(value, field_name=field_name)
    if isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list):
        raise InterfaceValidationError(
            f"{field_name} must be a list-like numeric vector, got "
            f"{type(value).__name__}."
        )

    numbers: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise InterfaceValidationError(
                f"{field_name}[{index}] must be a real number, got "
                f"{type(item).__name__}."
            )
        numbers.append(float(item))
    return numbers


def _extract_actions_value(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
) -> object:
    """Return the raw OpenPI ``actions`` payload."""

    if isinstance(response_or_actions, Mapping):
        if "actions" not in response_or_actions:
            raise InterfaceValidationError(
                "OpenPI response must contain an 'actions' field."
            )
        return response_or_actions["actions"]
    return response_or_actions


def _coerce_action_rows(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
) -> list[list[float]]:
    """Normalize OpenPI action payloads into ``list[list[float]]``."""

    actions = _coerce_optional_python_list(
        _extract_actions_value(response_or_actions),
        field_name="actions",
    )
    if isinstance(actions, tuple):
        actions = list(actions)

    if not isinstance(actions, list):
        raise InterfaceValidationError(
            "OpenPI actions must be a 1D or 2D list-like numeric payload, got "
            f"{type(actions).__name__}."
        )
    if not actions:
        raise InterfaceValidationError("OpenPI returned an empty action chunk.")

    first = actions[0]
    if isinstance(first, bool):
        raise InterfaceValidationError(
            "OpenPI actions must contain numeric values, not booleans."
        )
    if isinstance(first, Real):
        return [_ensure_float_list(actions, field_name="actions")]

    rows: list[list[float]] = []
    for row_index, row in enumerate(actions):
        rows.append(
            _ensure_float_list(row, field_name=f"actions[{row_index}]")
        )
    return rows


def openpi_actions_to_action_plan(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    *,
    mode: str,
    dt: float = 0.1,
    ref_frame: str | None = None,
) -> list[Action]:
    """Convert an OpenPI action chunk into embodia-standard actions."""

    plan = [
        Action(
            mode=mode,
            value=row,
            ref_frame=ref_frame,
            dt=dt,
        )
        for row in _coerce_action_rows(response_or_actions)
    ]
    for action in plan:
        validate_action(action)
    return plan


def openpi_first_action(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    *,
    mode: str,
    dt: float = 0.1,
    ref_frame: str | None = None,
) -> Action:
    """Convert an OpenPI action chunk and return its first action."""

    return openpi_actions_to_action_plan(
        response_or_actions,
        mode=mode,
        dt=dt,
        ref_frame=ref_frame,
    )[0]


def _coerce_embodia_action_plan(
    plan: Action | Mapping[str, Any] | Sequence[Action | Mapping[str, Any]],
) -> list[Action]:
    """Normalize an embodia action plan into a validated list of actions."""

    if isinstance(plan, Action) or isinstance(plan, Mapping):
        action = coerce_action(plan)
        validate_action(action)
        return [action]

    if isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
        raise InterfaceValidationError(
            "action plan must be one action-like object or a sequence of "
            f"action-like objects, got {type(plan).__name__}."
        )

    actions: list[Action] = []
    for index, item in enumerate(plan):
        action = coerce_action(item)
        try:
            validate_action(action)
        except InterfaceValidationError as exc:
            raise InterfaceValidationError(
                f"invalid action at plan index {index}: {exc}"
            ) from exc
        actions.append(action)

    if not actions:
        raise InterfaceValidationError("action plan must not be empty.")
    return actions


def openpi_response_from_action_plan(
    plan: Action | Mapping[str, Any] | Sequence[Action | Mapping[str, Any]],
    *,
    include_embodia_metadata: bool = True,
) -> dict[str, Any]:
    """Convert embodia actions into an OpenPI-compatible response dict.

    The core OpenPI field is ``actions``. embodia-specific metadata is attached
    under ``embodia`` so official OpenPI clients can ignore it safely.
    """

    actions = _coerce_embodia_action_plan(plan)
    response: dict[str, Any] = {
        "actions": [list(action.value) for action in actions],
    }

    if include_embodia_metadata:
        response["embodia"] = {
            "action_mode": actions[0].mode,
            "action_dt": actions[0].dt,
            "action_ref_frame": actions[0].ref_frame,
            "chunk_size": len(actions),
            "gripper": [action.gripper for action in actions],
        }

    return response


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


class EmbodiaModelPolicyAdapter:
    """Expose an embodia-style model through the OpenPI websocket protocol.

    This adapter keeps the model itself simple: the model still only receives a
    normalized frame and returns one action. Chunking, alternate response
    shapes, or native observation conversion stay outside the model and can be
    swapped in here.
    """

    def __init__(
        self,
        model: object,
        *,
        obs_to_frame: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]] | None = None,
        action_plan_provider: Callable[[object, Frame], Any] | None = None,
        response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None,
        server_metadata: Mapping[str, Any] | None = None,
        reset_model_on_connect: bool = False,
    ) -> None:
        self.model = model
        self._obs_to_frame = obs_to_frame or (lambda obs: coerce_frame(obs))
        self._action_plan_provider = action_plan_provider
        self._response_builder = (
            response_builder
            or (lambda actions, frame: openpi_response_from_action_plan(actions))
        )
        self._server_metadata = (
            dict(server_metadata) if server_metadata is not None else {}
        )
        self._reset_model_on_connect = reset_model_on_connect

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

        get_spec = getattr(self.model, "get_spec", None)
        if callable(get_spec):
            embodia_meta.setdefault("model_spec", model_spec_to_dict(get_spec()))

        return metadata

    def on_connect(self) -> None:
        """Optional lifecycle hook used by :class:`WebsocketPolicyServer`."""

        if self._reset_model_on_connect:
            self.reset()

    def reset(self) -> None:
        """Reset the wrapped model when it exposes ``reset()``."""

        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Convert remote observations, run the model, and build a response."""

        frame = self._coerce_frame(obs)
        if self._action_plan_provider is not None:
            raw_plan = self._action_plan_provider(self.model, frame)
        else:
            step = getattr(self.model, "step", None)
            if not callable(step):
                raise InterfaceValidationError(
                    f"{type(self.model).__name__} must expose step(frame) to be "
                    "served through EmbodiaModelPolicyAdapter."
                )
            raw_plan = step(frame)

        actions = _coerce_embodia_action_plan(raw_plan)
        response = self._response_builder(actions, frame)
        if not isinstance(response, Mapping):
            raise InterfaceValidationError(
                "response_builder must return a mapping, got "
                f"{type(response).__name__}."
            )
        return dict(response)


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
                "Remote policy access is disabled for this model instance. "
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
    "EmbodiaModelPolicyAdapter",
    "OpenPIMsgpackCodec",
    "RemotePolicyRunner",
    "OpenPIWebsocketClientPolicy",
    "OpenPIWebsocketPolicyServer",
    "WebsocketClientPolicy",
    "WebsocketPolicyServer",
    "is_official_openpi_client_available",
    "is_openpi_remote_available",
    "openpi_actions_to_action_plan",
    "openpi_first_action",
    "openpi_response_from_action_plan",
    "require_openpi_remote",
]
