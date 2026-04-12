"""Tests for inferaxis's inference runtime utilities."""

from __future__ import annotations

from collections import deque
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import inferaxis as infra
import numpy as np

from inferaxis.runtime.inference.chunk_scheduler import ChunkScheduler, _CompletedChunk
from inferaxis.runtime.inference.profile import (
    _ProfiledRequestSample,
    _build_async_buffer_trace,
)

from helpers import assert_array_equal, demo_image


def arm_value(action: infra.Action) -> float:
    """Return the first arm dimension from one action."""

    command = action.get_command("arm")
    assert command is not None
    return float(command.value[0])


def arm_action(value: float) -> infra.Action:
    """Build one single-arm action for compact scheduler tests."""

    return infra.Action.single(
        target="arm",
        command="cartesian_pose_delta",
        value=[value] * 6,
    )


class RuntimeRobot:
    """Tiny plain local executor used by inference-runtime tests."""

    def __init__(self) -> None:
        self.last_action: infra.Action | None = None

    def get_obs(self) -> infra.Frame:
        return infra.Frame(
            images={"front_rgb": demo_image()},
            state={"arm": np.zeros(6, dtype=np.float64)},
        )

    def send_action(self, action: infra.Action) -> None:
        self.last_action = action

    def reset(self) -> infra.Frame:
        return self.get_obs()


class RuntimePolicy:
    """Policy used by runtime tests through one infer(frame, request) entrypoint."""

    def __init__(self) -> None:
        self.step_index = 0
        self.chunk_seed = 1.0

    def reset(self) -> None:
        self.step_index = 0
        self.chunk_seed = 1.0

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        del obs
        history_actions = request.history_actions
        if not history_actions:
            start = self.chunk_seed
            self.chunk_seed += 2.0
            return [
                infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[start] * 6,
                ),
                infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[start + 1.0] * 6,
                ),
            ]

        last = arm_value(history_actions[-1])
        return [
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[last + 1.0] * 6,
            ),
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[last + 2.0] * 6,
            ),
        ]


class PlainRuntimeExecutor:
    """Plain local executor without inferaxis mixins."""

    def __init__(self) -> None:
        self.last_action: infra.Action | None = None

    def get_obs(self) -> infra.Frame:
        return infra.Frame(
            images={"front_rgb": demo_image()},
            state={"arm": np.zeros(6, dtype=np.float64)},
        )

    def send_action(self, action: infra.Action) -> None:
        self.last_action = action

    def reset(self) -> infra.Frame:
        return self.get_obs()


class SingleActionChunkPolicy:
    """Policy that returns one action directly from infer(frame, request)."""

    def __init__(self) -> None:
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> infra.Action:
        del obs, request
        value = float(1 + self.step_index)
        self.step_index += 1
        return infra.Action.single(
            target="arm",
            command="cartesian_pose_delta",
            value=[value] * 6,
        )


class PlanningSource:
    """Simple source exposing one future-action provider."""

    def __init__(self) -> None:
        self.plan_base = 10.0

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        del obs, request
        base = self.plan_base
        self.plan_base += 10.0
        return [
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[base] * 6,
            ),
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[base + 1.0] * 6,
            ),
        ]


class DeterministicClock:
    """Clock that returns one predefined timestamp per call."""

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        try:
            return next(self._values)
        except StopIteration as exc:
            raise AssertionError("DeterministicClock exhausted.") from exc


def make_profile_clock(
    *,
    step_durations: list[float],
    inference_durations: list[float],
    observe_duration: float = 0.001,
    inter_step_gap: float = 0.001,
) -> DeterministicClock:
    """Build one deterministic clock for profile_sync_inference tests."""

    timestamps: list[float] = []
    current = 0.0
    for step_duration, inference_duration in zip(
        step_durations,
        inference_durations,
        strict=True,
    ):
        inference_start = current + observe_duration
        timestamps.extend(
            [
                current,
                inference_start,
                inference_start + inference_duration,
                current + step_duration,
            ]
        )
        current += step_duration + inter_step_gap
    return DeterministicClock(timestamps)


class InferenceRuntimeTests(unittest.TestCase):
    """Coverage for action optimizers and sync/async runtime flow."""

    def test_action_ensembler_is_identity_when_called_directly(self) -> None:
        frame = infra.Frame(images={}, state={})
        ensembler = infra.ActionEnsembler(current_weight=0.5)

        first = ensembler(
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[0.0, 2.0],
            ),
            frame,
        )
        second = ensembler(
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[2.0, 4.0],
            ),
            frame,
        )

        assert_array_equal(self, first.get_command("arm").value, [0.0, 2.0])  # type: ignore[union-attr]
        assert_array_equal(self, second.get_command("arm").value, [2.0, 4.0])  # type: ignore[union-attr]

    def test_action_interpolator_blends_over_one_extra_step(self) -> None:
        frame = infra.Frame(images={}, state={})
        interpolator = infra.ActionInterpolator(steps=1)

        first = interpolator(
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[0.0, 0.0],
            ),
            frame,
        )
        second = interpolator(
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[2.0, 4.0],
            ),
            frame,
        )
        third = interpolator(
            infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[2.0, 4.0],
            ),
            frame,
        )

        assert_array_equal(self, first.get_command("arm").value, [0.0, 0.0])  # type: ignore[union-attr]
        assert_array_equal(self, second.get_command("arm").value, [1.0, 2.0])  # type: ignore[union-attr]
        assert_array_equal(self, third.get_command("arm").value, [2.0, 4.0])  # type: ignore[union-attr]

    def test_sync_runtime_skips_action_optimizers_for_single_action_source(self) -> None:
        robot = RuntimeRobot()
        policy = SingleActionChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            action_optimizers=[infra.ActionEnsembler(current_weight=0.5)],
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.raw_action), 1.0)
        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.raw_action), 2.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(robot.last_action), 2.0)  # type: ignore[arg-type]

    def test_async_runtime_rejects_single_action_source(self) -> None:
        robot = RuntimeRobot()
        policy = SingleActionChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            overlap_ratio=0.5,
            action_optimizers=[infra.ActionEnsembler(current_weight=0.5)],
        )

        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )

        self.assertIn("more than one action", str(ctx.exception))

    def test_sync_runtime_does_not_filter_multi_step_chunks_without_overlap(self) -> None:
        robot = RuntimeRobot()
        source = PlanningSource()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            action_optimizers=[infra.ActionEnsembler(current_weight=0.5)],
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )

        self.assertTrue(first.plan_refreshed)
        self.assertFalse(second.plan_refreshed)
        self.assertEqual(arm_value(first.raw_action), 10.0)
        self.assertEqual(arm_value(first.action), 10.0)
        self.assertEqual(arm_value(second.raw_action), 11.0)
        self.assertEqual(arm_value(second.action), 11.0)
        self.assertEqual(arm_value(robot.last_action), 11.0)  # type: ignore[arg-type]

    def test_async_runtime_does_not_filter_each_emitted_action(self) -> None:
        robot = RuntimeRobot()
        source = PlanningSource()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            overlap_ratio=0.5,
            action_optimizers=[infra.ActionEnsembler(current_weight=0.5)],
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.raw_action), 10.0)
        self.assertEqual(arm_value(first.action), 10.0)
        self.assertEqual(arm_value(second.raw_action), 11.0)
        self.assertEqual(arm_value(second.action), 11.0)
        self.assertEqual(arm_value(robot.last_action), 11.0)  # type: ignore[arg-type]

    def test_sync_runtime_memoizes_single_action_source_and_rejects_later_chunks(self) -> None:
        class InconsistentPolicy:
            def __init__(self) -> None:
                self.step_index = 0

            def reset(self) -> None:
                self.step_index = 0

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> infra.Action | list[infra.Action]:
                del obs, request
                if self.step_index == 0:
                    self.step_index += 1
                    return infra.Action.single(
                        target="arm",
                        command="cartesian_pose_delta",
                        value=[1.0] * 6,
                    )
                return [
                    infra.Action.single(
                        target="arm",
                        command="cartesian_pose_delta",
                        value=[2.0] * 6,
                    ),
                    infra.Action.single(
                        target="arm",
                        command="cartesian_pose_delta",
                        value=[3.0] * 6,
                    ),
                ]

        robot = RuntimeRobot()
        policy = InconsistentPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertIsNone(runtime._chunk_scheduler)

        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )

        self.assertIn("previously classified as single-step", str(ctx.exception))

    def test_sync_runtime_accepts_plain_local_executor(self) -> None:
        executor = PlainRuntimeExecutor()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(mode=infra.InferenceMode.SYNC)

        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.action), 1.0)
        self.assertEqual(arm_value(executor.last_action), 1.0)  # type: ignore[arg-type]

    def test_profile_and_recommend_accept_plain_local_executor(self) -> None:
        executor = PlainRuntimeExecutor()
        policy = SingleActionChunkPolicy()

        profile = infra.profile_sync_inference(
            observe_fn=executor.get_obs,
            target_hz=50.0,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            clock=make_profile_clock(
                step_durations=[0.02, 0.02, 0.02, 0.02],
                inference_durations=[0.01, 0.009, 0.008, 0.007],
            ),
        )
        recommendation = infra.recommend_inference_mode(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
            target_hz=50.0,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            clock=make_profile_clock(
                step_durations=[0.018, 0.018, 0.018, 0.018],
                inference_durations=[0.01, 0.01, 0.01, 0.01],
            ),
        )

        self.assertAlmostEqual(profile.estimated_inference_time_s, 0.0085)
        self.assertEqual(profile.target_hz, 50.0)
        self.assertEqual(recommendation.recommended_mode, infra.InferenceMode.SYNC)
        self.assertFalse(recommendation.async_supported)
        self.assertTrue(recommendation.sync_expected_to_meet_target)

    def test_sync_runtime_prefers_robot_returned_action(self) -> None:
        class ReturningRobot(RuntimeRobot):
            def send_action(self, action: object) -> infra.Action:
                del action
                self.last_action = infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[9.0] * 6,
                )
                return self.last_action

        robot = ReturningRobot()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(mode=infra.InferenceMode.SYNC)

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.raw_action), 1.0)
        self.assertEqual(arm_value(result.action), 9.0)
        self.assertEqual(arm_value(robot.last_action), 9.0)  # type: ignore[arg-type]

    def test_runtime_step_keeps_public_shape_small(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            action_optimizers=[infra.ActionEnsembler(current_weight=0.5)],
        )

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertFalse(hasattr(result, "timing"))
        self.assertGreaterEqual(result.control_wait_s, 0.0)

    def test_sync_runtime_can_use_overlap_with_overlap_aware_source(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            overlap_ratio=0.5,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        third = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 3.0)
        self.assertEqual(arm_value(third.action), 5.0)

    def test_async_runtime_can_use_internal_scheduler(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            overlap_ratio=0.5,
        )

        first = runtime.step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
        )
        second = runtime.step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
        )
        third = runtime.step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
        )

        self.assertTrue(first.plan_refreshed)
        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(third.action), 4.0)

    def test_chunk_scheduler_merges_ready_response_into_current_buffer(self) -> None:
        scheduler = ChunkScheduler(
            overlap_ratio=0.5,
            use_overlap_blend=True,
            overlap_current_weight=0.5,
        )
        scheduler._buffer = deque(
            [
                arm_action(12.0),
                arm_action(13.0),
                arm_action(14.0),
            ]
        )
        scheduler._global_step = 2
        scheduler._reference_chunk_size = 4
        scheduler._active_source_plan_length = 4
        refreshed = scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=infra.ChunkRequest(
                    request_step=1,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=4,
                    remaining_steps=3,
                    overlap_steps=2,
                    latency_steps=1,
                    request_trigger_steps=3,
                    plan_start_step=1,
                    history_actions=[],
                ),
                prepared_actions=[
                    arm_action(11.0),
                    arm_action(56.0),
                    arm_action(57.0),
                    arm_action(58.0),
                ],
                source_plan_length=3,
            )
        )

        self.assertTrue(refreshed)
        self.assertEqual(
            [arm_value(action) for action in scheduler._buffer],
            [56.0, 57.0, 58.0],
        )

    def test_chunk_scheduler_replaces_overlap_when_blending_is_disabled(self) -> None:
        scheduler = ChunkScheduler(
            overlap_ratio=0.5,
            use_overlap_blend=False,
        )
        scheduler._buffer = deque(
            [
                arm_action(12.0),
                arm_action(13.0),
                arm_action(14.0),
            ]
        )
        scheduler._global_step = 2
        scheduler._reference_chunk_size = 4
        scheduler._active_source_plan_length = 4
        refreshed = scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=infra.ChunkRequest(
                    request_step=1,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=4,
                    remaining_steps=3,
                    overlap_steps=2,
                    latency_steps=1,
                    request_trigger_steps=3,
                    plan_start_step=1,
                    history_actions=[],
                ),
                prepared_actions=[
                    arm_action(11.0),
                    arm_action(100.0),
                    arm_action(101.0),
                    arm_action(102.0),
                ],
                source_plan_length=3,
            )
        )

        self.assertTrue(refreshed)
        self.assertEqual(
            [arm_value(action) for action in scheduler._buffer],
            [100.0, 101.0, 102.0],
        )

    def test_chunk_scheduler_uses_step_latency_ema_for_triggering(self) -> None:
        scheduler = ChunkScheduler(
            overlap_ratio=0.25,
            latency_ema_beta=0.5,
            initial_latency_steps=2.0,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=infra.ChunkRequest(
                    request_step=0,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    overlap_steps=0,
                    latency_steps=2,
                    request_trigger_steps=2,
                    plan_start_step=0,
                    history_actions=[],
                ),
                prepared_actions=[
                    arm_action(1.0),
                    arm_action(2.0),
                    arm_action(3.0),
                    arm_action(4.0),
                ],
                source_plan_length=4,
            )
        )
        scheduler._global_step = 3

        self.assertEqual(scheduler.estimated_latency_steps(), 1)
        self.assertEqual(scheduler.request_trigger_steps(4, include_latency=True), 2)
        self.assertEqual(scheduler.request_trigger_steps(4, include_latency=False), 1)

    def test_async_scheduler_drops_only_executed_steps_after_wall_clock_wait(self) -> None:
        release_second_chunk = threading.Event()
        request_count = 0

        def action_source(
            frame: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del frame
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                return [arm_action(float(index)) for index in range(10)]
            self.assertEqual(request.request_step, 5)
            release_second_chunk.wait(timeout=1.0)
            self.assertTrue(release_second_chunk.is_set())
            return [arm_action(float(100 + index)) for index in range(10)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            overlap_ratio=0.5,
            latency_ema_beta=1.0,
            initial_latency_steps=0.0,
        )
        frame = infra.Frame(images={}, state={})

        emitted = [
            arm_value(scheduler.next_action(frame, prefetch_async=True)[0])
            for _ in range(10)
        ]
        self.assertEqual(emitted, [float(index) for index in range(10)])
        self.assertEqual(scheduler._global_step, 10)
        self.assertEqual(len(scheduler._buffer), 0)

        time.sleep(0.05)
        release_second_chunk.set()

        next_action, refreshed = scheduler.next_action(frame, prefetch_async=True)

        self.assertTrue(refreshed)
        self.assertEqual(arm_value(next_action), 105.0)
        self.assertEqual(
            [arm_value(action) for action in scheduler._buffer],
            [106.0, 107.0, 108.0, 109.0],
        )
        self.assertEqual(scheduler.estimated_latency_steps(), 5)

    def test_async_runtime_requires_action_source(self) -> None:
        robot = RuntimeRobot()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            overlap_ratio=0.5,
        )

        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                runtime=runtime,
            )

        self.assertIn("act_src_fn", str(ctx.exception))

    def test_sync_overlap_requires_action_source(self) -> None:
        robot = RuntimeRobot()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            overlap_ratio=0.5,
        )

        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                runtime=runtime,
            )

        self.assertIn("act_src_fn", str(ctx.exception))

    def test_sync_runtime_future_only_chunk_handoff_uses_request_step_origin(self) -> None:
        robot = RuntimeRobot()
        source = PlanningSource()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            overlap_ratio=0.5,
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        third = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )
        fourth = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 10.0)
        self.assertEqual(arm_value(second.action), 20.0)
        self.assertEqual(arm_value(third.action), 30.0)
        self.assertEqual(arm_value(fourth.action), 40.0)

    def test_runtime_closes_old_scheduler_when_source_changes(self) -> None:
        robot = RuntimeRobot()
        first_policy = RuntimePolicy()
        second_policy = RuntimePolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            overlap_ratio=0.5,
        )

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=first_policy.infer,
            runtime=runtime,
        )
        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=first_policy.infer,
            runtime=runtime,
        )
        first_scheduler = runtime._chunk_scheduler
        self.assertIsNotNone(first_scheduler)
        self.assertIsNotNone(first_scheduler._executor)  # type: ignore[union-attr]

        infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=second_policy.infer,
            runtime=runtime,
        )

        self.assertIsNot(runtime._chunk_scheduler, first_scheduler)
        self.assertIsNone(first_scheduler._executor)  # type: ignore[union-attr]

    def test_runtime_resets_action_optimizers_when_chunk_source_changes(self) -> None:
        robot = RuntimeRobot()
        first_policy = PlanningSource()
        second_policy = PlanningSource()
        second_policy.plan_base = 100.0
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            overlap_ratio=0.0,
            action_optimizers=[infra.ActionEnsembler(current_weight=0.5)],
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=first_policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=second_policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 10.0)
        self.assertEqual(arm_value(second.raw_action), 100.0)
        self.assertEqual(arm_value(second.action), 100.0)

    def test_async_runtime_blends_chunk_handoff_overlap_with_action_ensembler(self) -> None:
        class FourStepPolicy:
            def __init__(self) -> None:
                self.base = 1.0

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> list[infra.Action]:
                del obs, request
                base = self.base
                self.base += 4.0
                return [
                    infra.Action.single(
                        target="arm",
                        command="cartesian_pose_delta",
                        value=[base + offset] * 6,
                    )
                    for offset in range(4)
                ]

        robot = RuntimeRobot()
        policy = FourStepPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            overlap_ratio=0.5,
            action_optimizers=[infra.ActionEnsembler(current_weight=0.5)],
        )

        first = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        second = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        third = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )
        fourth = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(third.action), 3.0)
        self.assertEqual(arm_value(fourth.raw_action), 5.0)
        self.assertEqual(arm_value(fourth.action), 5.0)

    def test_profile_sync_inference_uses_requested_stable_sample_window(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        profile = infra.profile_sync_inference(
            observe_fn=robot.get_obs,
            target_hz=50.0,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            clock=make_profile_clock(
                step_durations=[0.02, 0.02, 0.02, 0.02, 0.02],
                inference_durations=[0.01, 0.009, 0.008, 0.006, 0.004],
            ),
        )

        self.assertEqual(profile.stable_inference_sample_count, 2)
        self.assertAlmostEqual(profile.target_period_s, 0.02)
        self.assertAlmostEqual(profile.estimated_inference_time_s, 0.0085)
        self.assertAlmostEqual(profile.estimated_step_time_s, 0.02)

    def test_profile_sync_inference_estimates_sustainable_hz_from_chunk_size(self) -> None:
        robot = RuntimeRobot()
        source = PlanningSource()

        profile = infra.profile_sync_inference(
            observe_fn=robot.get_obs,
            target_hz=50.0,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            overlap_ratio=0.5,
            clock=make_profile_clock(
                step_durations=[0.02, 0.02, 0.02, 0.02],
                inference_durations=[0.01, 0.009, 0.008, 0.007],
            ),
        )

        self.assertAlmostEqual(profile.estimated_chunk_steps, 2.0)
        self.assertAlmostEqual(profile.estimated_inference_time_s, 0.0085)
        self.assertAlmostEqual(profile.estimated_max_buffered_hz, 2.0 / 0.0085)
        self.assertAlmostEqual(profile.estimated_max_overlap_safe_hz, 1.0 / 0.0085)
        self.assertEqual(profile.estimated_latency_steps, 2)

    def test_profile_sync_inference_can_log_requests_and_ignore_flags(self) -> None:
        robot = RuntimeRobot()
        source = PlanningSource()
        lines: list[str] = []

        infra.profile_sync_inference(
            observe_fn=robot.get_obs,
            target_hz=50.0,
            act_fn=robot.send_action,
            act_src_fn=source.infer,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            request_log_fn=lines.append,
            clock=make_profile_clock(
                step_durations=[0.02, 0.02, 0.02],
                inference_durations=[0.01, 0.009, 0.008],
            ),
        )

        self.assertEqual(len(lines), 3)
        self.assertIn("request=0", lines[0])
        self.assertIn("inference_sample=ignored", lines[0])
        self.assertIn("front_rgb", lines[0])
        self.assertIn("shape", lines[0])
        self.assertIn("arm", lines[0])
        self.assertTrue(any("inference_sample=used" in line for line in lines[1:]))

    def test_profile_sync_inference_uses_target_hz_for_latency_steps(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        profile = infra.profile_sync_inference(
            observe_fn=robot.get_obs,
            target_hz=100.0,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            clock=make_profile_clock(
                step_durations=[0.03, 0.03, 0.03, 0.03],
                inference_durations=[0.024, 0.024, 0.024, 0.024],
            ),
        )

        self.assertAlmostEqual(profile.estimated_step_time_s, 0.03)
        self.assertAlmostEqual(profile.target_period_s, 0.01)
        self.assertEqual(profile.estimated_latency_steps, 4)

    def test_recommend_inference_mode_prefers_sync_when_budget_fits(self) -> None:
        robot = RuntimeRobot()
        policy = SingleActionChunkPolicy()

        recommendation = infra.recommend_inference_mode(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            target_hz=50.0,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            clock=make_profile_clock(
                step_durations=[0.018, 0.018, 0.018, 0.018],
                inference_durations=[0.01, 0.01, 0.01, 0.01],
            ),
        )

        self.assertEqual(recommendation.recommended_mode, infra.InferenceMode.SYNC)
        self.assertFalse(recommendation.async_supported)
        self.assertTrue(recommendation.sync_expected_to_meet_target)

    def test_recommend_inference_mode_prefers_async_when_sync_is_too_slow(self) -> None:
        robot = RuntimeRobot()
        policy = PlanningSource()

        recommendation = infra.recommend_inference_mode(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            target_hz=50.0,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            clock=make_profile_clock(
                step_durations=[0.03, 0.03, 0.03, 0.03],
                inference_durations=[0.024, 0.024, 0.024, 0.024],
            ),
        )

        self.assertEqual(recommendation.recommended_mode, infra.InferenceMode.ASYNC)
        self.assertTrue(recommendation.async_supported)
        self.assertFalse(recommendation.sync_expected_to_meet_target)
        self.assertIn("act_src_fn", recommendation.reason)

    def test_recommend_inference_mode_treats_single_action_source_as_sync_only(self) -> None:
        robot = RuntimeRobot()
        policy = SingleActionChunkPolicy()

        recommendation = infra.recommend_inference_mode(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            target_hz=50.0,
            startup_ignore_inference_samples=1,
            stable_inference_sample_count=2,
            timing_trim_ratio=0.0,
            clock=make_profile_clock(
                step_durations=[0.03, 0.03, 0.03, 0.03],
                inference_durations=[0.024, 0.024, 0.024, 0.024],
            ),
        )

        self.assertEqual(recommendation.recommended_mode, infra.InferenceMode.SYNC)
        self.assertFalse(recommendation.async_supported)
        self.assertFalse(recommendation.sync_expected_to_meet_target)
        self.assertIn("does not return", recommendation.reason)

    def test_profile_sync_inference_writes_json_report(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profile.json"
            profile = infra.profile_sync_inference(
                observe_fn=robot.get_obs,
                target_hz=50.0,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                execute_action=False,
                stable_inference_sample_count=3,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["steps"], 4)
            self.assertEqual(payload["stable_inference_sample_count"], 3)
            self.assertEqual(payload["target_hz"], 50.0)
            self.assertIn("estimated_max_buffered_hz", payload)
            self.assertEqual(profile.to_dict()["steps"], 4)

    def test_profile_sync_inference_builds_async_buffer_trace_and_svg(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_json_path = Path(tmpdir) / "buffer_trace.json"
            trace_svg_path = Path(tmpdir) / "buffer_trace.svg"
            profile = infra.profile_sync_inference(
                observe_fn=robot.get_obs,
                target_hz=100.0,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                overlap_ratio=0.5,
                startup_ignore_inference_samples=1,
                stable_inference_sample_count=2,
                timing_trim_ratio=0.0,
                clock=make_profile_clock(
                    step_durations=[0.03, 0.03, 0.03],
                    inference_durations=[0.024, 0.024, 0.024],
                ),
            )

            trace = profile.async_buffer_trace
            self.assertIsNotNone(trace)
            assert trace is not None
            self.assertGreater(len(trace.steps), 0)
            self.assertGreaterEqual(len(trace.requests), 1)
            self.assertEqual(trace.requests[0].request_index, 0)
            self.assertTrue(any(step.request_started for step in trace.steps))
            self.assertTrue(any(step.request_completed for step in trace.steps))
            self.assertTrue(any(step.underrun for step in trace.steps))

            payload = profile.to_dict(include_trace=True)
            self.assertIn("async_buffer_trace", payload)

            profile.write_async_buffer_trace_json(trace_json_path)
            profile.write_async_buffer_trace_svg(trace_svg_path)

            self.assertTrue(trace_json_path.exists())
            self.assertTrue(trace_svg_path.exists())
            trace_payload = json.loads(trace_json_path.read_text(encoding="utf-8"))
            self.assertIn("steps", trace_payload)
            self.assertIn("requests", trace_payload)
            self.assertIn("<svg", trace_svg_path.read_text(encoding="utf-8"))

    def test_async_buffer_trace_drops_only_executed_steps_during_underrun(self) -> None:
        trace = _build_async_buffer_trace(
            request_samples=[
                _ProfiledRequestSample(
                    request_index=0,
                    chunk_steps=10,
                    inference_time_s=0.0,
                    ignored_inference_sample=True,
                    observed_latency_steps=1,
                ),
                _ProfiledRequestSample(
                    request_index=1,
                    chunk_steps=10,
                    inference_time_s=0.0,
                    ignored_inference_sample=False,
                    observed_latency_steps=8,
                ),
            ],
            target_hz=50.0,
            overlap_ratio=0.5,
        )

        self.assertGreaterEqual(len(trace.requests), 2)
        second_request = trace.requests[1]
        self.assertEqual(second_request.observed_latency_steps, 8)
        self.assertEqual(second_request.executed_wait_steps, 5)
        self.assertEqual(second_request.aligned_chunk_steps, 5)
        self.assertTrue(any(step.underrun for step in trace.steps))


if __name__ == "__main__":
    unittest.main()
