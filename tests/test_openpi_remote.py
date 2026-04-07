"""Tests for the optional OpenPI remote-policy helpers."""

from __future__ import annotations

import http
import unittest
from unittest import mock

from embodia import (
    Action,
    Frame,
    InferenceMode,
    InferenceRuntime,
    InterfaceValidationError,
    ModelMixin,
    RobotMixin,
    run_step,
)
from embodia.contrib import openpi_remote as em_openpi_remote
from tests.helpers import DummyModel


class FakeCodec:
    """Small fake codec used to exercise websocket client control flow."""

    def __init__(self) -> None:
        self.packed_payloads: list[object] = []

    def packb(self, obj: object) -> bytes:
        self.packed_payloads.append(obj)
        return b"packed"

    def unpackb(self, data: bytes) -> object:
        if data == b"metadata":
            return {"name": "demo_server"}
        if data == b"response":
            return {"actions": [[1.0, 2.0, 3.0]]}
        if data == b"request":
            return {"state": [0.0]}
        raise AssertionError(f"unexpected payload {data!r}")


class FakeConnection:
    """Simple fake websocket connection."""

    def __init__(self, responses: list[bytes | str]) -> None:
        self._responses = list(responses)
        self.sent: list[bytes] = []
        self.closed = False

    def recv(self) -> bytes | str:
        if not self._responses:
            raise AssertionError("no more fake websocket responses available")
        return self._responses.pop(0)

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


class FakeServerConnection:
    """Small fake HTTP connection for ``process_request`` tests."""

    def __init__(self) -> None:
        self.responses: list[tuple[http.HTTPStatus, str]] = []

    def respond(self, status: http.HTTPStatus, body: str) -> tuple[http.HTTPStatus, str]:
        response = (status, body)
        self.responses.append(response)
        return response


class FakeRequest:
    """Small request object with path and headers."""

    def __init__(self, *, path: str, headers: dict[str, str] | None = None) -> None:
        self.path = path
        self.headers = headers or {}


class ConnectionClosed(Exception):
    """Fake websocket close signal used by the server handler tests."""


class FakeServerWebsocket:
    """Fake websocket object for server handler tests."""

    def __init__(self, messages: list[bytes | str]) -> None:
        self._messages = list(messages)
        self.sent: list[bytes | str] = []
        self.closed_with: tuple[tuple[object, ...], dict[str, object]] | None = None

    def recv(self) -> bytes | str:
        if self._messages:
            return self._messages.pop(0)
        raise ConnectionClosed()

    def send(self, data: bytes | str) -> None:
        self.sent.append(data)

    def close(self, *args: object, **kwargs: object) -> None:
        self.closed_with = (args, kwargs)


class OpenPIRemoteTests(unittest.TestCase):
    """Focused coverage for the optional remote-policy integration."""

    def test_openpi_first_action_converts_chunk_response(self) -> None:
        action = em_openpi_remote.openpi_first_action(
            {"actions": [[1, 2, 3], [4, 5, 6]]},
            mode="joint_position",
            dt=0.05,
        )

        self.assertIsInstance(action, Action)
        self.assertEqual(action.mode, "joint_position")
        self.assertEqual(action.value, [1.0, 2.0, 3.0])
        self.assertEqual(action.dt, 0.05)

    def test_openpi_actions_to_action_plan_accepts_one_dimensional_payload(self) -> None:
        plan = em_openpi_remote.openpi_actions_to_action_plan(
            [0.1, 0.2, 0.3],
            mode="ee_delta",
            dt=0.1,
            ref_frame="tool",
        )

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].value, [0.1, 0.2, 0.3])
        self.assertEqual(plan[0].ref_frame, "tool")

    def test_openpi_response_from_action_plan_preserves_embodia_metadata(self) -> None:
        response = em_openpi_remote.openpi_response_from_action_plan(
            [
                {"mode": "ee_delta", "value": [1.0, 2.0], "gripper": 0.5, "dt": 0.05},
                {"mode": "ee_delta", "value": [3.0, 4.0], "gripper": 0.2, "dt": 0.05},
            ]
        )

        self.assertEqual(response["actions"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(response["embodia"]["action_mode"], "ee_delta")
        self.assertEqual(response["embodia"]["chunk_size"], 2)
        self.assertEqual(response["embodia"]["gripper"], [0.5, 0.2])

    def test_embodia_model_policy_adapter_serves_single_step_model(self) -> None:
        model = DummyModel()
        adapter = em_openpi_remote.EmbodiaModelPolicyAdapter(model)

        response = adapter.infer(
            {
                "timestamp_ns": 1,
                "images": {"front_rgb": None},
                "state": {"joint_positions": [0.0] * 6},
            }
        )
        metadata = adapter.get_server_metadata()

        self.assertEqual(response["actions"], [[0.0] * 6])
        self.assertEqual(
            metadata["embodia"]["model_spec"]["name"],
            "dummy_model",
        )

    def test_embodia_model_policy_adapter_can_use_chunk_provider(self) -> None:
        model = DummyModel()

        def provider(model: object, frame: object) -> list[dict[str, object]]:
            del model, frame
            return [
                {"mode": "ee_delta", "value": [1.0] * 6, "dt": 0.1},
                {"mode": "ee_delta", "value": [2.0] * 6, "dt": 0.1},
            ]

        adapter = em_openpi_remote.EmbodiaModelPolicyAdapter(
            model,
            action_plan_provider=provider,
        )
        response = adapter.infer(
            {
                "timestamp_ns": 1,
                "images": {"front_rgb": None},
                "state": {"joint_positions": [0.0] * 6},
            }
        )

        self.assertEqual(response["actions"], [[1.0] * 6, [2.0] * 6])

    def test_require_openpi_remote_reports_missing_optional_deps(self) -> None:
        with mock.patch.object(
            em_openpi_remote,
            "is_openpi_remote_available",
            return_value=False,
        ):
            with self.assertRaises(InterfaceValidationError) as ctx:
                em_openpi_remote.require_openpi_remote()

        self.assertIn("openpi-remote", str(ctx.exception))

    def test_websocket_client_reads_metadata_and_infers(self) -> None:
        connection = FakeConnection([b"metadata", b"response"])
        codec = FakeCodec()

        with mock.patch.object(
            em_openpi_remote,
            "_connect_websocket",
            return_value=connection,
        ) as connect:
            client = em_openpi_remote.WebsocketClientPolicy(
                host="localhost",
                port=8000,
                codec=codec,
                retry_interval_s=0.01,
            )
            response = client.infer({"state": [0.0]})
            metadata = client.get_server_metadata()
            client.close()

        self.assertEqual(metadata["name"], "demo_server")
        self.assertEqual(response["actions"], [[1.0, 2.0, 3.0]])
        self.assertEqual(codec.packed_payloads, [{"state": [0.0]}])
        self.assertEqual(connection.sent, [b"packed"])
        self.assertTrue(connection.closed)
        connect.assert_called_once_with(
            "ws://localhost:8000",
            headers=None,
        )

    def test_websocket_client_can_connect_lazily(self) -> None:
        connection = FakeConnection([b"metadata", b"response"])
        codec = FakeCodec()

        with mock.patch.object(
            em_openpi_remote,
            "_connect_websocket",
            return_value=connection,
        ) as connect:
            client = em_openpi_remote.WebsocketClientPolicy(
                host="localhost",
                port=9000,
                codec=codec,
                retry_interval_s=0.01,
                connect_immediately=False,
            )
            self.assertFalse(client.is_connected)
            client.infer({"obs": 1})

        self.assertTrue(client.is_connected)
        connect.assert_called_once_with(
            "ws://localhost:9000",
            headers=None,
        )

    def test_websocket_client_raises_on_text_error_frame(self) -> None:
        connection = FakeConnection([b"metadata", "remote traceback"])
        codec = FakeCodec()

        with mock.patch.object(
            em_openpi_remote,
            "_connect_websocket",
            return_value=connection,
        ):
            client = em_openpi_remote.WebsocketClientPolicy(
                codec=codec,
                retry_interval_s=0.01,
            )
            with self.assertRaises(RuntimeError) as ctx:
                client.infer({"state": [0.0]})

        self.assertIn("remote traceback", str(ctx.exception))

    def test_remote_policy_runner_creates_client_lazily(self) -> None:
        created: list[dict[str, object]] = []

        class FakeRemoteClient:
            def __init__(self, **kwargs: object) -> None:
                created.append(dict(kwargs))

            @property
            def is_connected(self) -> bool:
                return True

            def get_server_metadata(self) -> dict[str, object]:
                return {"name": "server"}

            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                return {"obs": obs}

            def reset(self) -> None:
                return None

            def close(self) -> None:
                return None

        runner = em_openpi_remote.RemotePolicyRunner(
            enabled=True,
            host="localhost",
            port=8000,
            client_factory=FakeRemoteClient,
        )

        self.assertFalse(runner.is_connected)
        response = runner.infer({"x": 1})
        metadata = runner.get_server_metadata()

        self.assertEqual(response, {"obs": {"x": 1}})
        self.assertEqual(metadata, {"name": "server"})
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["host"], "localhost")
        self.assertEqual(created[0]["port"], 8000)

    def test_remote_policy_runner_rejects_when_disabled(self) -> None:
        runner = em_openpi_remote.RemotePolicyRunner(enabled=False)

        with self.assertRaises(InterfaceValidationError) as ctx:
            runner.infer({"x": 1})

        self.assertIn("use_remote_policy=True", str(ctx.exception))

    def test_robot_mixin_can_hide_remote_policy_backend(self) -> None:
        runner_events: list[tuple[str, object]] = []

        class FakeRunner:
            def __init__(self, **kwargs: object) -> None:
                runner_events.append(("init", dict(kwargs)))
                self.enabled = bool(kwargs["enabled"])

            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                runner_events.append(("infer", dict(obs)))
                return {"actions": [[1.0] * 6]}

            def reset(self) -> None:
                runner_events.append(("reset", None))

            def close(self) -> None:
                runner_events.append(("close", None))

        class RemoteRobot(RobotMixin):
            ROBOT_SPEC = {
                "name": "remote_robot",
                "action_modes": ["ee_delta"],
                "image_keys": [],
                "state_keys": ["joint_positions"],
            }
            METHOD_ALIASES = {
                "observe": "capture",
                "act": "send_command",
                "reset": "home",
            }

            def __init__(self, *, dt: float = 0.05) -> None:
                self.dt = dt
                self.last_policy_output = None
                self.last_action = None

            def capture(self) -> dict[str, object]:
                return {
                    "timestamp_ns": 1,
                    "images": {},
                    "state": {"joint_positions": [0.0] * 6},
                }

            def send_command(self, action: Action) -> None:
                self.last_action = action

            def home(self) -> dict[str, object]:
                return self.capture()

        with mock.patch.object(
            em_openpi_remote,
            "RemotePolicyRunner",
            FakeRunner,
        ):
            robot = RemoteRobot(dt=0.05)
            robot.configure_remote_policy(
                host="localhost",
                port=8000,
                obs_builder=lambda frame: {
                    "state": list(frame.state["joint_positions"]),
                },
                action_mode="ee_delta",
                dt=0.05,
            )
            action = robot.request_remote_policy_action(
                Frame(
                    timestamp_ns=1,
                    images={},
                    state={"joint_positions": [0.0] * 6},
                )
            )
            policy_output = robot.last_policy_output
            robot.act(action)
            robot.close_remote_policy()

        self.assertFalse(hasattr(robot, "_remote_policy"))
        self.assertEqual(action.value, [1.0] * 6)
        self.assertEqual(policy_output, {"actions": [[1.0] * 6]})
        self.assertEqual(runner_events[0][0], "init")
        self.assertEqual(runner_events[0][1]["host"], "localhost")
        self.assertEqual(runner_events[0][1]["port"], 8000)
        self.assertEqual(runner_events[1], ("infer", {"state": [0.0] * 6}))
        self.assertEqual(runner_events[2], ("close", None))
        self.assertEqual(robot.last_action.value, [1.0] * 6)

    def test_robot_mixin_from_config_can_enable_remote_policy(self) -> None:
        runner_events: list[tuple[str, object]] = []

        class FakeRunner:
            def __init__(self, **kwargs: object) -> None:
                runner_events.append(("init", dict(kwargs)))
                self.enabled = bool(kwargs["enabled"])

            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                runner_events.append(("infer", dict(obs)))
                return {"actions": [[1.0] * 6]}

            def reset(self) -> None:
                runner_events.append(("reset", None))

            def close(self) -> None:
                runner_events.append(("close", None))

        class RemoteRobot(RobotMixin):
            ROBOT_SPEC = {
                "name": "remote_robot",
                "action_modes": ["ee_delta"],
                "image_keys": [],
                "state_keys": ["joint_positions"],
            }
            METHOD_ALIASES = {
                "observe": "capture",
                "act": "send_command",
                "reset": "home",
            }

            def capture(self) -> dict[str, object]:
                return {
                    "timestamp_ns": 1,
                    "images": {},
                    "state": {"joint_positions": [0.0] * 6},
                }

            def send_command(self, action: Action) -> None:
                del action

            def home(self) -> dict[str, object]:
                return self.capture()

        with mock.patch.object(
            em_openpi_remote,
            "RemotePolicyRunner",
            FakeRunner,
        ):
            robot = RemoteRobot.from_config(
                remote_policy={
                    "host": "localhost",
                    "port": 8000,
                    "obs_builder": lambda frame: {
                        "state": list(frame.state["joint_positions"]),
                    },
                    "action_mode": "ee_delta",
                    "dt": 0.05,
                }
            )
            action = robot.request_remote_policy_action(
                Frame(
                    timestamp_ns=1,
                    images={},
                    state={"joint_positions": [0.0] * 6},
                )
            )
            robot.close_remote_policy()

        self.assertEqual(action.value, [1.0] * 6)
        self.assertEqual(runner_events[0][0], "init")
        self.assertEqual(runner_events[0][1]["host"], "localhost")
        self.assertEqual(runner_events[0][1]["port"], 8000)
        self.assertEqual(runner_events[1], ("infer", {"state": [0.0] * 6}))

    def test_run_step_can_auto_use_robot_remote_policy(self) -> None:
        runner_events: list[tuple[str, object]] = []

        class FakeRunner:
            def __init__(self, **kwargs: object) -> None:
                runner_events.append(("init", dict(kwargs)))
                self.enabled = bool(kwargs["enabled"])

            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                runner_events.append(("infer", dict(obs)))
                return {"actions": [[2.0] * 6]}

            def close(self) -> None:
                runner_events.append(("close", None))

        class RemoteRobot(RobotMixin):
            ROBOT_SPEC = {
                "name": "remote_robot",
                "action_modes": ["ee_delta"],
                "image_keys": [],
                "state_keys": ["joint_positions"],
            }
            METHOD_ALIASES = {
                "observe": "capture",
                "act": "send_command",
                "reset": "home",
            }

            def __init__(self) -> None:
                self.last_action = None

            def capture(self) -> dict[str, object]:
                return {
                    "timestamp_ns": 1,
                    "images": {},
                    "state": {"joint_positions": [0.0] * 6},
                }

            def send_command(self, action: Action) -> None:
                self.last_action = action

            def home(self) -> dict[str, object]:
                return self.capture()

        with mock.patch.object(
            em_openpi_remote,
            "RemotePolicyRunner",
            FakeRunner,
        ):
            robot = RemoteRobot.from_config(
                remote_policy={
                    "host": "localhost",
                    "port": 8000,
                    "obs_builder": lambda frame: {
                        "state": list(frame.state["joint_positions"]),
                    },
                    "action_mode": "ee_delta",
                    "dt": 0.05,
                }
            )
            result = run_step(robot)
            robot.close_remote_policy()

        self.assertEqual(result.action.value, [2.0] * 6)
        self.assertEqual(robot.last_action.value, [2.0] * 6)
        self.assertEqual(runner_events[1], ("infer", {"state": [0.0] * 6}))

    def test_inference_runtime_can_auto_use_robot_remote_policy(self) -> None:
        runner_events: list[tuple[str, object]] = []

        class FakeRunner:
            def __init__(self, **kwargs: object) -> None:
                runner_events.append(("init", dict(kwargs)))
                self.enabled = bool(kwargs["enabled"])

            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                runner_events.append(("infer", dict(obs)))
                return {"actions": [[3.0] * 6]}

            def close(self) -> None:
                runner_events.append(("close", None))

        class RemoteRobot(RobotMixin):
            ROBOT_SPEC = {
                "name": "remote_robot",
                "action_modes": ["ee_delta"],
                "image_keys": [],
                "state_keys": ["joint_positions"],
            }
            METHOD_ALIASES = {
                "observe": "capture",
                "act": "send_command",
                "reset": "home",
            }

            def __init__(self) -> None:
                self.last_action = None

            def capture(self) -> dict[str, object]:
                return {
                    "timestamp_ns": 1,
                    "images": {},
                    "state": {"joint_positions": [0.0] * 6},
                }

            def send_command(self, action: Action) -> None:
                self.last_action = action

            def home(self) -> dict[str, object]:
                return self.capture()

        with mock.patch.object(
            em_openpi_remote,
            "RemotePolicyRunner",
            FakeRunner,
        ):
            robot = RemoteRobot.from_config(
                remote_policy={
                    "host": "localhost",
                    "port": 8000,
                    "obs_builder": lambda frame: {
                        "state": list(frame.state["joint_positions"]),
                    },
                    "action_mode": "ee_delta",
                    "dt": 0.05,
                }
            )
            runtime = InferenceRuntime(mode=InferenceMode.SYNC)
            result = run_step(robot, runtime=runtime)
            robot.close_remote_policy()

        self.assertEqual(result.action.value, [3.0] * 6)
        self.assertEqual(robot.last_action.value, [3.0] * 6)
        self.assertEqual(runner_events[1], ("infer", {"state": [0.0] * 6}))

    def test_model_mixin_can_build_openpi_policy_adapter(self) -> None:
        model = DummyModel()

        adapter = model.build_openpi_policy_adapter(
            obs_to_frame=lambda obs: {
                "timestamp_ns": 1,
                "images": {"front_rgb": obs.get("front_rgb")},
                "state": {"joint_positions": obs.get("joint_positions", [0.0] * 6)},
            }
        )

        response = adapter.infer(
            {
                "front_rgb": None,
                "joint_positions": [0.0] * 6,
            }
        )

        self.assertEqual(response["actions"], [[0.0] * 6])

    def test_websocket_policy_server_handles_healthz_and_api_key(self) -> None:
        policy = mock.Mock()
        server = em_openpi_remote.WebsocketPolicyServer(
            policy,
            api_key="secret",
        )
        connection = FakeServerConnection()

        healthz = server._process_request(
            connection,
            FakeRequest(path="/healthz"),
        )
        unauthorized = server._process_request(
            connection,
            FakeRequest(path="/infer", headers={}),
        )
        authorized = server._process_request(
            connection,
            FakeRequest(
                path="/infer",
                headers={"Authorization": "Api-Key secret"},
            ),
        )

        self.assertEqual(healthz, (http.HTTPStatus.OK, "OK\n"))
        self.assertEqual(unauthorized, (http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n"))
        self.assertIsNone(authorized)

    def test_websocket_policy_server_sends_metadata_and_response(self) -> None:
        codec = FakeCodec()
        websocket = FakeServerWebsocket([b"request"])
        policy = mock.Mock()
        policy.infer.return_value = {"actions": [[4.0, 5.0]]}
        policy.get_server_metadata.return_value = {"name": "served_policy"}

        server = em_openpi_remote.WebsocketPolicyServer(
            policy,
            codec=codec,
        )

        with mock.patch.object(
            em_openpi_remote,
            "_is_connection_closed_error",
            side_effect=lambda exc: isinstance(exc, ConnectionClosed),
        ):
            server._handle_connection(websocket)

        self.assertEqual(
            codec.packed_payloads,
            [
                {"name": "served_policy"},
                {
                    "actions": [[4.0, 5.0]],
                    "server_timing": mock.ANY,
                },
            ],
        )
        self.assertEqual(websocket.sent, [b"packed", b"packed"])
        policy.infer.assert_called_once_with({"state": [0.0]})


if __name__ == "__main__":
    unittest.main()
