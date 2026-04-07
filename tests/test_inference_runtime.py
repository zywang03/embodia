"""Tests for embodia's low-coupling inference runtime helpers."""

from __future__ import annotations

from concurrent.futures import Future
import unittest

from embodia import (
    Action,
    ActionEnsembler,
    AsyncInference,
    ChunkRequest,
    Frame,
    InferenceMode,
    InferenceRuntime,
    InferenceStepResult,
    InterfaceValidationError,
    RealtimeController,
    run_step,
)
from tests.helpers import DummyModel, DummyRobot


class SequenceModel(DummyModel):
    """Dummy model that emits a predictable action sequence."""

    def __init__(self) -> None:
        self.step_index = 0

    def _reset_impl(self) -> None:
        self.step_index = 0

    def _step_impl(self, frame: Frame) -> dict[str, object]:
        del frame

        base = float(1 + self.step_index * 2)
        self.step_index += 1
        return {"mode": "ee_delta", "value": [base] * 6, "dt": 0.1}


class FakeClock:
    """Small mutable clock used to test pacing without real sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ManualAsyncInference(AsyncInference):
    """AsyncInference subclass with manual completion control for tests."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pending_requests: list[
            tuple[Future, object, Frame, ChunkRequest, object | None]
        ] = []

    def _submit_request(
        self,
        source: object,
        frame: Frame,
        request: ChunkRequest,
        fallback_action_source,
    ) -> Future:
        future: Future = Future()
        self.pending_requests.append(
            (future, source, frame, request, fallback_action_source)
        )
        return future

    def finish_next_request(self) -> ChunkRequest:
        future, source, frame, request, fallback_action_source = self.pending_requests.pop(0)
        future.set_result(
            self._execute_request(
                source,
                frame,
                request,
                fallback_action_source,
            )
        )
        return request


class InferenceRuntimeTests(unittest.TestCase):
    """Focused coverage for inference-time optimization helpers."""

    def test_action_ensembler_averages_recent_actions(self) -> None:
        frame = Frame(timestamp_ns=1, images={}, state={})
        ensembler = ActionEnsembler(window_size=2)

        first = ensembler(
            Action(mode="ee_delta", value=[0.0, 2.0], dt=0.1),
            frame,
        )
        second = ensembler(
            Action(mode="ee_delta", value=[2.0, 4.0], dt=0.1),
            frame,
        )

        self.assertEqual(first.value, [0.0, 2.0])
        self.assertEqual(second.value, [1.0, 3.0])

    def test_action_ensembler_restarts_on_incompatible_actions(self) -> None:
        frame = Frame(timestamp_ns=1, images={}, state={})
        ensembler = ActionEnsembler(window_size=3)

        ensembler(Action(mode="ee_delta", value=[1.0] * 6, dt=0.1), frame)
        restarted = ensembler(
            Action(mode="joint_position", value=[3.0] * 6, dt=0.1),
            frame,
        )

        self.assertEqual(restarted.mode, "joint_position")
        self.assertEqual(restarted.value, [3.0] * 6)

    def test_realtime_controller_waits_for_target_hz(self) -> None:
        clock = FakeClock()
        sleep_calls: list[float] = []

        def sleeper(duration: float) -> None:
            sleep_calls.append(duration)
            clock.now += duration

        controller = RealtimeController(
            hz=10.0,
            clock=clock,
            sleeper=sleeper,
            warning_emitter=lambda message: None,
        )

        first_wait = controller.wait()
        clock.now += 0.04
        second_wait = controller.wait()
        clock.now += 0.12
        third_wait = controller.wait()

        self.assertAlmostEqual(first_wait, 0.1)
        self.assertAlmostEqual(second_wait, 0.06)
        self.assertEqual(third_wait, 0.0)
        self.assertEqual(len(sleep_calls), 2)

    def test_realtime_controller_throttles_missed_hz_warnings(self) -> None:
        clock = FakeClock()
        warnings_seen: list[str] = []

        controller = RealtimeController(
            hz=10.0,
            clock=clock,
            sleeper=lambda duration: None,
            warning_interval_s=5.0,
            warning_emitter=warnings_seen.append,
        )

        first_wait = controller.wait()
        clock.now += 0.25
        second_wait = controller.wait()
        clock.now += 1.0
        third_wait = controller.wait()
        clock.now += 5.1
        fourth_wait = controller.wait()

        self.assertAlmostEqual(first_wait, 0.1)
        self.assertEqual(second_wait, 0.0)
        self.assertEqual(third_wait, 0.0)
        self.assertEqual(fourth_wait, 0.0)
        self.assertEqual(len(warnings_seen), 2)
        self.assertIn("10.000 Hz", warnings_seen[0])

    def test_async_inference_computes_minimum_steps_from_latency_budget(self) -> None:
        async_inference = AsyncInference(
            condition_steps=2,
            prefetch_steps=1,
            control_hz=50.0,
            latency_budget_s=0.035,
        )

        self.assertEqual(async_inference.minimum_prefetch_steps(), 3)
        self.assertEqual(async_inference.minimum_condition_steps(), 3)
        self.assertEqual(async_inference.effective_prefetch_steps(), 3)

    def test_async_inference_rejects_prefetch_longer_than_condition(self) -> None:
        with self.assertRaises(InterfaceValidationError) as ctx:
            AsyncInference(condition_steps=1, prefetch_steps=2)

        self.assertIn("condition_steps must be >= prefetch_steps", str(ctx.exception))

    def test_async_inference_prefetches_async_chunk_and_trims_consumed_prefix(
        self,
    ) -> None:
        robot = DummyRobot()
        model = SequenceModel()
        clock = FakeClock()
        seen_requests: list[ChunkRequest] = []

        def chunk_provider(
            source: object,
            frame: Frame,
            request: ChunkRequest,
        ) -> list[dict[str, object]]:
            del source, frame
            seen_requests.append(request)
            if not request.history_actions:
                return [
                    {"mode": "ee_delta", "value": [1.0] * 6, "dt": 0.1},
                    {"mode": "ee_delta", "value": [2.0] * 6, "dt": 0.1},
                    {"mode": "ee_delta", "value": [3.0] * 6, "dt": 0.1},
                    {"mode": "ee_delta", "value": [4.0] * 6, "dt": 0.1},
                ]

            overlap = [
                {"mode": action.mode, "value": list(action.value), "dt": action.dt}
                for action in request.history_actions
            ]
            return overlap + [
                {"mode": "ee_delta", "value": [5.0] * 6, "dt": 0.1},
                {"mode": "ee_delta", "value": [6.0] * 6, "dt": 0.1},
            ]

        runtime = InferenceRuntime(
            mode=InferenceMode.ASYNC,
            async_inference=ManualAsyncInference(
                chunk_provider=chunk_provider,
                condition_steps=2,
                prefetch_steps=2,
                control_hz=10.0,
                clock=clock,
                warning_emitter=lambda message: None,
            ),
        )

        first = runtime.step(robot, model)
        second = runtime.step(robot, model)
        async_inference = runtime.async_inference
        assert isinstance(async_inference, ManualAsyncInference)

        self.assertEqual(len(async_inference.pending_requests), 1)
        pending_request = async_inference.pending_requests[0][3]
        self.assertEqual(pending_request.request_step, 2)
        self.assertEqual(pending_request.history_start, 2)
        self.assertEqual(pending_request.history_end, 4)
        self.assertEqual(
            [action.value[0] for action in pending_request.history_actions],
            [3.0, 4.0],
        )

        clock.now = 0.15
        async_inference.finish_next_request()

        third = runtime.step(robot, model)
        fourth = runtime.step(robot, model)
        fifth = runtime.step(robot, model)

        self.assertTrue(first.plan_refreshed)
        self.assertEqual(first.action.value, [1.0] * 6)
        self.assertFalse(second.plan_refreshed)
        self.assertEqual(second.action.value, [2.0] * 6)
        self.assertFalse(third.plan_refreshed)
        self.assertEqual(third.action.value, [3.0] * 6)
        self.assertFalse(fourth.plan_refreshed)
        self.assertEqual(fourth.action.value, [4.0] * 6)
        self.assertTrue(fifth.plan_refreshed)
        self.assertEqual(fifth.action.value, [5.0] * 6)
        self.assertEqual(len(seen_requests), 2)
        self.assertAlmostEqual(async_inference.estimated_p99_latency_s or 0.0, 0.15)
        self.assertEqual(async_inference.effective_prefetch_steps(), 3)

    def test_async_inference_warns_when_overlap_window_is_smaller_than_latency_budget(
        self,
    ) -> None:
        robot = DummyRobot()
        model = DummyModel()
        clock = FakeClock()
        warnings_seen: list[str] = []

        def chunk_provider(
            source: object,
            frame: Frame,
            request: ChunkRequest,
        ) -> list[dict[str, object]]:
            del source, frame
            if not request.history_actions:
                return [
                    {"mode": "ee_delta", "value": [1.0] * 6, "dt": 0.1},
                    {"mode": "ee_delta", "value": [2.0] * 6, "dt": 0.1},
                    {"mode": "ee_delta", "value": [3.0] * 6, "dt": 0.1},
                ]

            overlap = [
                {"mode": action.mode, "value": list(action.value), "dt": action.dt}
                for action in request.history_actions
            ]
            return overlap + [
                {"mode": "ee_delta", "value": [9.0] * 6, "dt": 0.1},
            ]

        runtime = InferenceRuntime(
            mode=InferenceMode.ASYNC,
            async_inference=ManualAsyncInference(
                chunk_provider=chunk_provider,
                condition_steps=1,
                prefetch_steps=1,
                control_hz=10.0,
                clock=clock,
                warning_interval_s=0.001,
                warning_emitter=warnings_seen.append,
            ),
        )

        runtime.step(robot, model)
        runtime.step(robot, model)
        async_inference = runtime.async_inference
        assert isinstance(async_inference, ManualAsyncInference)
        clock.now = 0.15
        async_inference.finish_next_request()
        runtime.step(robot, model)

        self.assertTrue(any("prefetch_steps" in message for message in warnings_seen))
        self.assertTrue(any("condition_steps" in message for message in warnings_seen))

    def test_inference_runtime_applies_action_optimizers_before_execution(self) -> None:
        robot = DummyRobot()
        model = SequenceModel()
        runtime = InferenceRuntime(
            mode=InferenceMode.SYNC,
            action_optimizers=[ActionEnsembler(window_size=2)],
        )

        first = run_step(robot, model, runtime=runtime)
        second = run_step(robot, model, runtime=runtime)

        self.assertIsInstance(first, InferenceStepResult)
        self.assertEqual(first.raw_action.value, [1.0] * 6)
        self.assertEqual(first.action.value, [1.0] * 6)
        self.assertEqual(second.raw_action.value, [3.0] * 6)
        self.assertEqual(second.action.value, [2.0] * 6)
        self.assertEqual(robot.last_action.value, [2.0] * 6)

    def test_inference_runtime_accepts_string_mode_for_compatibility(self) -> None:
        runtime = InferenceRuntime(mode="sync")

        self.assertIs(runtime.mode, InferenceMode.SYNC)

    def test_inference_runtime_reset_clears_optimizer_and_model_state(self) -> None:
        robot = DummyRobot()
        model = SequenceModel()
        runtime = InferenceRuntime(
            mode=InferenceMode.SYNC,
            action_optimizers=[ActionEnsembler(window_size=2)],
        )

        run_step(robot, model, runtime=runtime)
        run_step(robot, model, runtime=runtime)
        restarted = run_step(robot, model, runtime=runtime, reset_model=True)

        self.assertEqual(restarted.raw_action.value, [1.0] * 6)
        self.assertEqual(restarted.action.value, [1.0] * 6)

    def test_run_step_can_use_inference_runtime_as_single_entrypoint(self) -> None:
        robot = DummyRobot()
        model = SequenceModel()
        clock = FakeClock()

        def chunk_provider(
            source: object,
            frame: Frame,
            request: ChunkRequest,
        ) -> list[dict[str, object]]:
            del source, frame, request
            return [
                {"mode": "ee_delta", "value": [1.0] * 6, "dt": 0.1},
                {"mode": "ee_delta", "value": [2.0] * 6, "dt": 0.1},
            ]

        runtime = InferenceRuntime(
            mode=InferenceMode.ASYNC,
            action_optimizers=[ActionEnsembler(window_size=2)],
            async_inference=AsyncInference(
                chunk_provider=chunk_provider,
                condition_steps=1,
                prefetch_steps=1,
                control_hz=10.0,
                clock=clock,
            ),
        )

        first = run_step(robot, model, runtime=runtime)
        second = run_step(robot, model, runtime=runtime)

        self.assertIsInstance(first, InferenceStepResult)
        self.assertTrue(first.plan_refreshed)
        self.assertFalse(second.plan_refreshed)

    def test_inference_runtime_can_run_with_external_action_fn(self) -> None:
        robot = DummyRobot()
        runtime = InferenceRuntime(
            mode=InferenceMode.SYNC,
            action_optimizers=[ActionEnsembler(window_size=2)],
        )

        def remote_policy(frame: Frame) -> dict[str, object]:
            base = float(frame.timestamp_ns >= 0)
            return {"mode": "ee_delta", "value": [base] * 6, "dt": 0.1}

        first = runtime.step(robot, remote_policy)
        second = runtime.step(robot, remote_policy)

        self.assertEqual(first.raw_action.value, [1.0] * 6)
        self.assertEqual(second.raw_action.value, [1.0] * 6)
        self.assertEqual(second.action.value, [1.0] * 6)
        self.assertEqual(robot.last_action.value, [1.0] * 6)

    def test_inference_runtime_rejects_missing_source_without_remote_policy(self) -> None:
        robot = DummyRobot()
        runtime = InferenceRuntime(mode=InferenceMode.SYNC)

        with self.assertRaises(Exception) as ctx:
            runtime.step(robot)

        self.assertIn("configured remote policy", str(ctx.exception))

    def test_inference_runtime_rejects_sync_mode_with_async_inference(self) -> None:
        with self.assertRaises(InterfaceValidationError) as ctx:
            InferenceRuntime(
                mode=InferenceMode.SYNC,
                async_inference=AsyncInference(plan_provider=lambda source, frame: []),
            )

        self.assertIn("mode='sync'", str(ctx.exception))

    def test_inference_runtime_rejects_async_mode_without_async_inference(
        self,
    ) -> None:
        with self.assertRaises(InterfaceValidationError) as ctx:
            InferenceRuntime(mode=InferenceMode.ASYNC)

        self.assertIn("mode='async'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
