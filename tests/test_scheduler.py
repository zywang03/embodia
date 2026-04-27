"""Scheduler-domain tests for inference runtime chunk scheduling."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
import math
import threading
import time
import unittest
from unittest import mock
import warnings

import inferaxis as infra
import numpy as np

from inferaxis.runtime.inference.scheduler import ChunkScheduler, _CompletedChunk

from helpers import (
    DeterministicClock,
    RtcLoggingChunkPolicy,
    RuntimeRobot,
    arm_action,
    arm_and_gripper_action,
    arm_value,
    assert_array_equal,
    gripper_value,
    make_chunk_request,
)


class SchedulerTests(unittest.TestCase):
    """Coverage for chunk scheduler blending, latency, RTC, and async flow."""

    def test_chunk_scheduler_no_longer_accepts_removed_transition_bridge_configuration(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            ChunkScheduler(
                enable_mismatch_bridge=False,  # type: ignore[call-arg]
            )
        with self.assertRaises(TypeError):
            ChunkScheduler(
                transition_bridge_steps=2,  # type: ignore[call-arg]
            )
        with self.assertRaises(TypeError):
            ChunkScheduler(
                transition_bridge_mismatch_threshold=1.5,  # type: ignore[call-arg]
            )

    def test_chunk_scheduler_blends_arm_but_not_gripper_overlap(self) -> None:
        scheduler = ChunkScheduler(
            use_overlap_blend=True,
            overlap_current_weight=0.25,
        )

        blended = scheduler._blend_overlap_action(
            arm_and_gripper_action(arm=0.0, gripper=0.0),
            arm_and_gripper_action(arm=4.0, gripper=1.0),
        )

        assert_array_equal(self, blended.get_command("arm").value, [1.0] * 6)  # type: ignore[union-attr]
        assert_array_equal(self, blended.get_command("gripper").value, [1.0])  # type: ignore[union-attr]

    def test_chunk_scheduler_weight_schedule_preserves_old_early_and_new_late(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            use_overlap_blend=True,
            overlap_current_weight=(0.2, 0.8),
        )

        first = scheduler._blend_overlap_action(
            arm_action(0.0),
            arm_action(10.0),
            overlap_index=0,
            overlap_count=3,
        )
        middle = scheduler._blend_overlap_action(
            arm_action(0.0),
            arm_action(10.0),
            overlap_index=1,
            overlap_count=3,
        )
        last = scheduler._blend_overlap_action(
            arm_action(0.0),
            arm_action(10.0),
            overlap_index=2,
            overlap_count=3,
        )

        self.assertEqual(arm_value(first), 2.0)
        self.assertEqual(arm_value(middle), 5.0)
        self.assertEqual(arm_value(last), 8.0)

    def test_chunk_scheduler_weight_schedule_uses_low_when_overlap_has_one_step(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            use_overlap_blend=True,
            overlap_current_weight=(0.2, 0.8),
        )

        blended = scheduler._blend_overlap_action(
            arm_action(0.0),
            arm_action(10.0),
            overlap_index=0,
            overlap_count=1,
        )

        self.assertEqual(arm_value(blended), 2.0)

    def test_chunk_scheduler_keeps_new_gripper_open_close_value_during_overlap(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            use_overlap_blend=True,
            overlap_current_weight=0.5,
        )

        blended = scheduler._blend_overlap_action(
            arm_and_gripper_action(
                arm=1.0,
                gripper=0.0,
                gripper_command=infra.BuiltinCommandKind.GRIPPER_OPEN_CLOSE,
            ),
            arm_and_gripper_action(
                arm=3.0,
                gripper=1.0,
                gripper_command=infra.BuiltinCommandKind.GRIPPER_OPEN_CLOSE,
            ),
        )

        assert_array_equal(self, blended.get_command("arm").value, [2.0] * 6)  # type: ignore[union-attr]
        assert_array_equal(self, blended.get_command("gripper").value, [1.0])  # type: ignore[union-attr]

    def test_chunk_scheduler_merges_ready_response_into_current_buffer(self) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
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
        scheduler._active_source_plan_length = 4
        refreshed = scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
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
            steps_before_request=0,
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
        scheduler._active_source_plan_length = 4
        refreshed = scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
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

    def test_chunk_scheduler_execute_request_reuses_source_plan_actions_without_blending(
        self,
    ) -> None:
        plan = [arm_action(1.0), arm_action(2.0), arm_action(3.0)]

        scheduler = ChunkScheduler(
            action_source=lambda obs, request: plan,
            use_overlap_blend=False,
        )

        completed = scheduler._execute_request(
            infra.Frame(images={}, state={}),
            scheduler._build_request_job(include_latency=False),
        )

        self.assertIs(completed.prepared_actions[0], plan[0])
        self.assertIs(completed.prepared_actions[1], plan[1])
        self.assertIs(completed.prepared_actions[2], plan[2])

    def test_chunk_scheduler_execute_request_blends_prefix_and_reuses_suffix(
        self,
    ) -> None:
        plan = [
            arm_action(50.0),
            arm_action(60.0),
            arm_action(70.0),
            arm_action(80.0),
        ]
        scheduler = ChunkScheduler(
            action_source=lambda obs, request: plan,
            use_overlap_blend=True,
            overlap_current_weight=0.5,
        )
        scheduler._buffer = deque(
            [
                arm_action(10.0),
                arm_action(20.0),
                arm_action(30.0),
            ]
        )

        completed = scheduler._execute_request(
            infra.Frame(images={}, state={}),
            scheduler._build_request_job(include_latency=False),
        )

        self.assertIsNot(completed.prepared_actions[0], plan[0])
        self.assertIsNot(completed.prepared_actions[1], plan[1])
        self.assertIsNot(completed.prepared_actions[2], plan[2])
        self.assertIs(completed.prepared_actions[3], plan[3])

    def test_chunk_scheduler_prepared_overlap_keeps_new_gripper_values(self) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
            use_overlap_blend=True,
            overlap_current_weight=0.5,
        )
        scheduler._buffer = deque(
            [
                arm_and_gripper_action(arm=12.0, gripper=0.2),
                arm_and_gripper_action(arm=13.0, gripper=0.3),
                arm_and_gripper_action(arm=14.0, gripper=0.4),
            ]
        )
        scheduler._global_step = 2
        scheduler._active_source_plan_length = 4
        refreshed = scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
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
                    arm_and_gripper_action(arm=11.0, gripper=0.1),
                    arm_and_gripper_action(arm=56.0, gripper=0.8),
                    arm_and_gripper_action(arm=57.0, gripper=0.9),
                    arm_and_gripper_action(arm=58.0, gripper=1.0),
                ],
                source_plan_length=3,
            )
        )

        self.assertTrue(refreshed)
        self.assertEqual(
            [arm_value(action) for action in scheduler._buffer],
            [56.0, 57.0, 58.0],
        )
        self.assertEqual(
            [gripper_value(action) for action in scheduler._buffer],
            [0.8, 0.9, 1.0],
        )

    def test_chunk_scheduler_uses_step_latency_ema_for_triggering_after_three_request_warmup(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=2,
            latency_ema_beta=0.5,
            initial_latency_steps=2.0,
            warmup_requests=3,
        )

        scheduler._update_latency_estimate(3)
        scheduler._update_latency_estimate(4)
        scheduler._update_latency_estimate(5)
        self.assertEqual(scheduler.estimated_latency_steps(), 2)

        scheduler._update_latency_estimate(0)

        self.assertEqual(scheduler.estimated_latency_steps(), 1)
        scheduler._active_chunk_waited_raw_steps = 1
        self.assertFalse(scheduler._steps_before_request_satisfied())
        scheduler._active_chunk_waited_raw_steps = 2
        self.assertTrue(scheduler._steps_before_request_satisfied())

    def test_chunk_scheduler_fixed_latency_steps_override_blocks_ema_updates(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=3,
            initial_latency_steps=1.0,
            fixed_latency_steps=4.0,
            warmup_requests=3,
            profile_delay_requests=3,
        )

        scheduler._update_latency_estimate(100)

        self.assertTrue(scheduler.latency_estimate_ready())
        self.assertEqual(scheduler.estimated_latency_steps(), 4)
        scheduler._active_chunk_waited_raw_steps = 2
        self.assertFalse(scheduler._steps_before_request_satisfied())
        scheduler._active_chunk_waited_raw_steps = 3
        self.assertTrue(scheduler._steps_before_request_satisfied())

    def test_chunk_scheduler_interpolation_expands_execution_sequence(self) -> None:
        scheduler = ChunkScheduler(
            interpolation_steps=2,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    overlap_steps=0,
                    latency_steps=0,
                    request_trigger_steps=0,
                    plan_start_step=0,
                    history_actions=[],
                ),
                prepared_actions=[
                    arm_action(0.0),
                    arm_action(3.0),
                    arm_action(6.0),
                ],
                source_plan_length=3,
            )
        )

        emitted: list[float] = []
        while scheduler._buffer or scheduler._execution_buffer:
            emitted.append(arm_value(scheduler._pop_next_action()))

        self.assertEqual(emitted, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_chunk_scheduler_pop_next_action_reuses_buffer_action_without_interpolation(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            interpolation_steps=0,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    latency_steps=0,
                ),
                prepared_actions=[arm_action(1.0), arm_action(2.0)],
                source_plan_length=2,
            )
        )

        first_buffer_action = scheduler._buffer[0]
        emitted = scheduler._pop_next_action()

        self.assertIs(emitted, first_buffer_action)

    def test_chunk_scheduler_next_action_tracks_live_state_in_raw_buffer_and_cursor(
        self,
    ) -> None:
        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            return [arm_action(1.0), arm_action(3.0)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            interpolation_steps=1,
        )

        action, refreshed = scheduler.next_action(
            infra.Frame(images={}, state={}),
            prefetch_async=False,
        )

        self.assertEqual(arm_value(action), 1.0)
        self.assertTrue(refreshed)
        self.assertTrue(scheduler._raw_buffer.has_actions)
        self.assertEqual(
            [
                arm_value(buffered)
                for buffered in scheduler._raw_buffer.remaining_actions()
            ],
            [1.0, 3.0],
        )
        self.assertFalse(scheduler._execution_cursor.at_raw_boundary)
        self.assertEqual(scheduler._execution_cursor.remaining_segment_steps, 1)

    def test_chunk_scheduler_execution_buffer_compatibility_view_stays_nonempty_at_raw_boundary(
        self,
    ) -> None:
        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            return [arm_action(1.0), arm_action(2.0)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            interpolation_steps=0,
        )

        action, refreshed = scheduler.next_action(
            infra.Frame(images={}, state={}),
            prefetch_async=False,
        )

        self.assertEqual(arm_value(action), 1.0)
        self.assertTrue(refreshed)
        self.assertTrue(scheduler._execution_cursor.at_raw_boundary)
        self.assertTrue(scheduler._raw_buffer.has_actions)
        self.assertEqual(scheduler._execution_cursor.remaining_segment_steps, 1)
        self.assertEqual(len(scheduler._execution_buffer), 1)
        self.assertEqual(
            [arm_value(buffered) for buffered in scheduler._execution_buffer],
            [2.0],
        )

    def test_chunk_scheduler_interpolation_count_matches_formula(self) -> None:
        scheduler = ChunkScheduler(
            interpolation_steps=2,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    overlap_steps=0,
                    latency_steps=0,
                    request_trigger_steps=0,
                    plan_start_step=0,
                    history_actions=[],
                ),
                prepared_actions=[arm_action(float(index)) for index in range(50)],
                source_plan_length=50,
            )
        )

        emitted_count = 0
        while scheduler._buffer or scheduler._execution_buffer:
            scheduler._pop_next_action()
            emitted_count += 1

        self.assertEqual(emitted_count, 50 + 49 * 2)

    def test_chunk_scheduler_interpolation_keeps_gripper_stepwise(self) -> None:
        scheduler = ChunkScheduler(
            interpolation_steps=2,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    overlap_steps=0,
                    latency_steps=0,
                    request_trigger_steps=0,
                    plan_start_step=0,
                    history_actions=[],
                ),
                prepared_actions=[
                    arm_and_gripper_action(arm=0.0, gripper=0.0),
                    arm_and_gripper_action(arm=3.0, gripper=1.0),
                ],
                source_plan_length=2,
            )
        )

        emitted: list[infra.Action] = []
        while scheduler._buffer or scheduler._execution_buffer:
            emitted.append(scheduler._pop_next_action())

        self.assertEqual(
            [arm_value(action) for action in emitted], [0.0, 1.0, 2.0, 3.0]
        )
        self.assertEqual(
            [gripper_value(action) for action in emitted],
            [0.0, 0.0, 0.0, 1.0],
        )

    def test_chunk_scheduler_interpolation_keeps_rtc_in_raw_steps_mid_segment(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
            execution_steps=3,
            interpolation_steps=2,
            enable_rtc=True,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    overlap_steps=0,
                    latency_steps=0,
                    request_trigger_steps=0,
                    plan_start_step=0,
                    history_actions=[],
                ),
                prepared_actions=[
                    arm_action(1.0),
                    arm_action(4.0),
                    arm_action(7.0),
                    arm_action(10.0),
                ],
                source_plan_length=4,
            )
        )

        self.assertEqual(arm_value(scheduler._pop_next_action()), 1.0)
        self.assertEqual(arm_value(scheduler._pop_next_action()), 2.0)
        self.assertEqual(scheduler._global_step, 0)
        scheduler._latency_steps_estimate = 1.0

        mid_segment_job = scheduler._build_request_job(include_latency=True)
        assert mid_segment_job.request.prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in mid_segment_job.request.prev_action_chunk],
            [1.0, 4.0, 7.0, 7.0],
        )
        self.assertEqual(mid_segment_job.request.execute_horizon, 3)
        self.assertEqual(mid_segment_job.request.inference_delay, 1)

        self.assertEqual(arm_value(scheduler._pop_next_action()), 3.0)
        self.assertEqual(scheduler._global_step, 1)
        boundary_job = scheduler._build_request_job(include_latency=True)
        self.assertEqual(boundary_job.request.execute_horizon, 3)
        assert boundary_job.request.prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in boundary_job.request.prev_action_chunk],
            [4.0, 7.0, 10.0, 10.0],
        )

    def test_chunk_scheduler_interpolated_execution_only_refreshes_on_raw_boundary(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            interpolation_steps=2,
            steps_before_request=0,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    overlap_steps=0,
                    latency_steps=0,
                    request_trigger_steps=0,
                    plan_start_step=0,
                    history_actions=[],
                ),
                prepared_actions=[
                    arm_action(0.0),
                    arm_action(3.0),
                ],
                source_plan_length=2,
            )
        )

        frame = infra.Frame(images={}, state={})
        first_action, first_refreshed = scheduler.next_action(
            frame, prefetch_async=True
        )
        self.assertEqual(arm_value(first_action), 0.0)
        self.assertFalse(first_refreshed)

        future: Future[_CompletedChunk] = Future()
        future.set_result(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    overlap_steps=0,
                    latency_steps=0,
                    request_trigger_steps=0,
                    plan_start_step=0,
                    history_actions=[],
                ),
                prepared_actions=[
                    arm_action(50.0),
                    arm_action(100.0),
                ],
                source_plan_length=2,
            )
        )
        scheduler._pending_future = future

        second_action, second_refreshed = scheduler.next_action(
            frame, prefetch_async=True
        )
        third_action, third_refreshed = scheduler.next_action(
            frame, prefetch_async=True
        )
        fourth_action, fourth_refreshed = scheduler.next_action(
            frame, prefetch_async=True
        )

        self.assertEqual(arm_value(second_action), 1.0)
        self.assertFalse(second_refreshed)
        self.assertEqual(arm_value(third_action), 2.0)
        self.assertFalse(third_refreshed)
        self.assertEqual(arm_value(fourth_action), 100.0)
        self.assertTrue(fourth_refreshed)

    def test_chunk_scheduler_enable_rtc_builds_fixed_prev_chunk_window_from_buffer_head(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    history_start=0,
                    history_end=0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    overlap_steps=0,
                    latency_steps=0,
                    request_trigger_steps=0,
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
        scheduler._pop_next_action()
        scheduler._pop_next_action()
        scheduler._latency_steps_estimate = 1.0

        job = scheduler._build_request_job(include_latency=True)

        rtc_args = job.request.rtc_args
        self.assertIsNotNone(rtc_args)
        assert rtc_args is not None
        self.assertEqual(rtc_args.inference_delay, 1)
        self.assertEqual(rtc_args.execute_horizon, 3)
        self.assertEqual(
            [arm_value(action) for action in rtc_args.prev_action_chunk],
            [3.0, 4.0, 4.0, 4.0],
        )
        self.assertEqual(job.request.inference_delay, 1)
        self.assertEqual(job.request.execute_horizon, 3)
        assert job.request.prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in job.request.prev_action_chunk],
            [3.0, 4.0, 4.0, 4.0],
        )

    def test_chunk_scheduler_enable_rtc_prev_chunk_reuses_live_buffer_actions(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    latency_steps=0,
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
        scheduler._latency_steps_estimate = 1.0

        job = scheduler._build_request_job(include_latency=True)

        assert job.request.prev_action_chunk is not None
        self.assertIs(job.request.prev_action_chunk[0], scheduler._buffer[0])
        self.assertIs(job.request.prev_action_chunk[1], scheduler._buffer[1])
        self.assertIs(job.request.prev_action_chunk[2], scheduler._buffer[2])
        self.assertIs(job.request.prev_action_chunk[3], scheduler._buffer[2])

    def test_chunk_scheduler_accepts_signed_latency_steps_offset(self) -> None:
        scheduler = ChunkScheduler(
            enable_rtc=True,
            execution_steps=1,
            latency_steps_offset=-3,
        )

        self.assertEqual(scheduler.latency_steps_offset, -3)

    def test_chunk_scheduler_accepts_startup_validation_only(self) -> None:
        scheduler = ChunkScheduler(
            startup_validation_only=True,
        )

        self.assertTrue(scheduler.startup_validation_only)

    def test_chunk_scheduler_rejects_invalid_latency_steps_offset(self) -> None:
        for invalid in (1.5, True, "2"):
            with self.assertRaises(infra.InterfaceValidationError):
                ChunkScheduler(
                    enable_rtc=True,
                    execution_steps=1,
                    latency_steps_offset=invalid,  # type: ignore[arg-type]
                )

    def test_chunk_scheduler_rejects_invalid_startup_validation_only(self) -> None:
        for invalid in (1, "yes", None):
            with self.assertRaises(infra.InterfaceValidationError):
                ChunkScheduler(
                    startup_validation_only=invalid,  # type: ignore[arg-type]
                )

    def test_chunk_scheduler_no_longer_accepts_legacy_rtc_delay_offset_keyword(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            ChunkScheduler(
                enable_rtc=True,
                rtc_inference_delay_offset_steps=1,  # type: ignore[call-arg]
            )

    def test_chunk_scheduler_enable_rtc_projects_shifted_latency_hint(self) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
            latency_steps_offset=1,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    latency_steps=0,
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
        scheduler._pop_next_action()
        scheduler._latency_steps_estimate = 1.0

        job = scheduler._build_request_job(include_latency=True)

        self.assertEqual(job.request.latency_steps, 2)
        self.assertEqual(job.request.inference_delay, 2)
        assert job.request.rtc_args is not None
        self.assertEqual(job.request.rtc_args.inference_delay, 2)
        self.assertEqual(job.request.execute_horizon, 3)

    def test_chunk_scheduler_enable_rtc_clamps_negative_latency_steps_offset(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
            latency_steps_offset=-10,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    latency_steps=0,
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
        scheduler._pop_next_action()
        scheduler._latency_steps_estimate = 2.0

        job = scheduler._build_request_job(include_latency=True)

        self.assertEqual(job.request.latency_steps, 0)
        self.assertEqual(job.request.inference_delay, 1)
        assert job.request.rtc_args is not None
        self.assertEqual(job.request.rtc_args.inference_delay, 1)
        self.assertEqual(job.request.execute_horizon, 3)

    def test_chunk_request_syncs_top_level_rtc_fields_into_rtc_args(self) -> None:
        request = make_chunk_request(
            request_step=0,
            request_time_s=0.0,
            history_start=0,
            history_end=0,
            active_chunk_length=0,
            remaining_steps=0,
            overlap_steps=0,
            latency_steps=0,
            request_trigger_steps=0,
            plan_start_step=0,
            history_actions=[],
            prev_action_chunk=[arm_action(7.0), arm_action(8.0)],
            inference_delay=3,
            execute_horizon=2,
        )

        self.assertIsNotNone(request.rtc_args)
        assert request.rtc_args is not None
        self.assertEqual(request.rtc_args.inference_delay, 3)
        self.assertEqual(request.rtc_args.execute_horizon, 2)
        self.assertEqual(
            [arm_value(action) for action in request.rtc_args.prev_action_chunk],
            [7.0, 8.0],
        )

    def test_chunk_scheduler_enable_rtc_does_not_change_latency_or_trigger_logic(
        self,
    ) -> None:
        def make_scheduler(enable_rtc: bool) -> ChunkScheduler:
            scheduler = ChunkScheduler(
                steps_before_request=0,
                execution_steps=3,
                enable_rtc=enable_rtc,
                clock=lambda: 123.0,
            )
            scheduler._buffer = deque(
                [
                    arm_action(3.0),
                    arm_action(4.0),
                    arm_action(5.0),
                ]
            )
            scheduler._active_chunk_consumed_steps = 1
            scheduler._latency_steps_estimate = 2.0
            if enable_rtc:
                scheduler._rtc_chunk_total_length = 4
            return scheduler

        without_rtc = make_scheduler(False)._build_request_job(include_latency=True)
        with_rtc = make_scheduler(True)._build_request_job(include_latency=True)

        self.assertEqual(
            without_rtc.request.request_step, with_rtc.request.request_step
        )
        self.assertEqual(
            without_rtc.request.request_time_s, with_rtc.request.request_time_s
        )
        self.assertEqual(
            without_rtc.request.active_chunk_length,
            with_rtc.request.active_chunk_length,
        )
        self.assertEqual(
            without_rtc.request.remaining_steps, with_rtc.request.remaining_steps
        )
        self.assertEqual(
            without_rtc.request.latency_steps, with_rtc.request.latency_steps
        )
        self.assertIsNone(without_rtc.request.prev_action_chunk)
        self.assertIsNone(without_rtc.request.inference_delay)
        self.assertIsNone(without_rtc.request.execute_horizon)
        self.assertIsNone(without_rtc.request.rtc_args)
        self.assertEqual(with_rtc.request.inference_delay, 2)
        self.assertEqual(with_rtc.request.execute_horizon, 3)
        assert with_rtc.request.prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in with_rtc.request.prev_action_chunk],
            [3.0, 4.0, 5.0, 5.0],
        )
        self.assertIsNotNone(with_rtc.request.rtc_args)

    def test_chunk_scheduler_latency_steps_offset_changes_request_hints_without_affecting_trigger_logic(
        self,
    ) -> None:
        def make_scheduler(offset: int) -> ChunkScheduler:
            scheduler = ChunkScheduler(
                steps_before_request=2,
                execution_steps=3,
                enable_rtc=True,
                latency_steps_offset=offset,
                clock=lambda: 123.0,
            )
            scheduler._buffer = deque(
                [
                    arm_action(3.0),
                    arm_action(4.0),
                    arm_action(5.0),
                ]
            )
            scheduler._active_chunk_consumed_steps = 1
            scheduler._active_chunk_waited_raw_steps = 1
            scheduler._latency_steps_estimate = 2.0
            scheduler._rtc_chunk_total_length = 4
            return scheduler

        base = make_scheduler(0)
        shifted = make_scheduler(2)

        self.assertEqual(base._latency_steps_estimate, shifted._latency_steps_estimate)
        self.assertEqual(
            base._remaining_control_steps(), shifted._remaining_control_steps()
        )
        self.assertEqual(
            base._steps_before_request_satisfied(),
            shifted._steps_before_request_satisfied(),
        )

        base_job = base._build_request_job(include_latency=True)
        shifted_job = shifted._build_request_job(include_latency=True)

        self.assertEqual(base_job.request.latency_steps, 2)
        self.assertEqual(shifted_job.request.latency_steps, 4)
        self.assertEqual(
            base_job.request.execute_horizon, shifted_job.request.execute_horizon
        )
        assert base_job.request.prev_action_chunk is not None
        assert shifted_job.request.prev_action_chunk is not None
        self.assertEqual(
            [arm_value(action) for action in base_job.request.prev_action_chunk],
            [arm_value(action) for action in shifted_job.request.prev_action_chunk],
        )
        self.assertEqual(base_job.request.inference_delay, 2)
        self.assertEqual(shifted_job.request.inference_delay, 3)

    def test_chunk_scheduler_latency_steps_offset_applies_even_when_rtc_is_disabled(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
            enable_rtc=False,
            latency_steps_offset=5,
            clock=lambda: 123.0,
        )
        scheduler._buffer = deque([arm_action(3.0), arm_action(4.0)])
        scheduler._active_chunk_consumed_steps = 1
        scheduler._latency_steps_estimate = 2.0

        job = scheduler._build_request_job(include_latency=True)

        self.assertEqual(job.request.latency_steps, 7)
        self.assertIsNone(job.request.inference_delay)
        self.assertIsNone(job.request.execute_horizon)
        self.assertIsNone(job.request.prev_action_chunk)
        self.assertIsNone(job.request.rtc_args)

    def test_chunk_scheduler_bootstrap_warns_before_continuing_after_slow_rtc_request(
        self,
    ) -> None:
        requests: list[infra.ChunkRequest] = []

        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs
            requests.append(request)
            return [arm_action(float(index + 1)) for index in range(18)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            steps_before_request=0,
            execution_steps=17,
            control_period_s=0.02,
            warmup_requests=1,
            profile_delay_requests=2,
            enable_rtc=True,
            slow_rtc_bootstrap="confirm",
            clock=DeterministicClock(
                [
                    0.00,
                    0.01,
                    0.11,
                    0.20,
                    0.21,
                    0.31,
                    0.40,
                    0.41,
                    0.96,
                ]
            ),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch("builtins.input", return_value="y") as prompt:
                bootstrapped = scheduler.bootstrap(RuntimeRobot().get_obs())

        self.assertTrue(bootstrapped)
        self.assertEqual(prompt.call_count, 1)
        self.assertEqual(len(requests), 3)
        self.assertIsNone(requests[0].rtc_args)
        self.assertIsNotNone(requests[1].rtc_args)
        self.assertIsNotNone(requests[2].rtc_args)
        self.assertTrue(
            any(
                "last RTC warmup request carrying prev_action_chunk took"
                in str(warning.message)
                for warning in caught
            )
        )

    def test_chunk_scheduler_bootstrap_warns_without_prompt_by_default_after_slow_rtc_request(
        self,
    ) -> None:
        requests: list[infra.ChunkRequest] = []

        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs
            requests.append(request)
            return [arm_action(float(index + 1)) for index in range(18)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            steps_before_request=0,
            execution_steps=17,
            control_period_s=0.02,
            warmup_requests=1,
            profile_delay_requests=2,
            enable_rtc=True,
            clock=DeterministicClock(
                [
                    0.00,
                    0.01,
                    0.11,
                    0.20,
                    0.21,
                    0.31,
                    0.40,
                    0.41,
                    0.96,
                ]
            ),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch("builtins.input") as prompt:
                bootstrapped = scheduler.bootstrap(RuntimeRobot().get_obs())

        self.assertTrue(bootstrapped)
        self.assertEqual(prompt.call_count, 0)
        self.assertEqual(len(requests), 3)
        self.assertTrue(scheduler.latency_estimate_ready())
        self.assertGreater(len(scheduler._buffer), 0)
        self.assertTrue(
            any(
                "last RTC warmup request carrying prev_action_chunk took"
                in str(warning.message)
                for warning in caught
            )
        )

    def test_chunk_scheduler_startup_validation_only_still_validates_startup_frame(
        self,
    ) -> None:
        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            return [arm_action(1.0), arm_action(2.0)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            startup_validation_only=True,
        )
        bad_frame = infra.Frame(images={}, state={})
        bad_frame.timestamp_ns = -1

        with self.assertRaises(infra.InterfaceValidationError):
            scheduler.next_action(bad_frame, prefetch_async=False)

    def test_chunk_scheduler_validation_off_skips_startup_frame_validation(
        self,
    ) -> None:
        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            return [arm_action(1.0), arm_action(2.0)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            validation="off",
        )
        bad_frame = infra.Frame(images={}, state={})
        bad_frame.timestamp_ns = -1

        action, refreshed = scheduler.next_action(bad_frame, prefetch_async=False)

        self.assertTrue(refreshed)
        self.assertEqual(arm_value(action), 1.0)

    def test_chunk_scheduler_startup_validation_only_skips_steady_state_plan_validation(
        self,
    ) -> None:
        request_count = 0

        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                return [arm_action(1.0), arm_action(2.0)]
            return [
                infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[np.nan] * 6,
                ),
                arm_action(4.0),
            ]

        scheduler = ChunkScheduler(
            action_source=action_source,
            steps_before_request=0,
            startup_validation_only=True,
            clock=lambda: 123.0,
        )

        action, refreshed = scheduler.next_action(
            infra.Frame(images={}, state={}),
            prefetch_async=False,
        )

        self.assertTrue(refreshed)
        self.assertEqual(request_count, 2)
        self.assertTrue(scheduler._startup_validation_complete)
        self.assertTrue(math.isnan(arm_value(action)))

    def test_chunk_scheduler_bootstrap_aborts_when_slow_rtc_request_is_rejected(
        self,
    ) -> None:
        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            return [arm_action(float(index + 1)) for index in range(18)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            steps_before_request=0,
            execution_steps=17,
            control_period_s=0.02,
            warmup_requests=1,
            profile_delay_requests=2,
            enable_rtc=True,
            slow_rtc_bootstrap="confirm",
            clock=DeterministicClock(
                [
                    0.00,
                    0.01,
                    0.11,
                    0.20,
                    0.21,
                    0.31,
                    0.40,
                    0.41,
                    0.96,
                ]
            ),
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with mock.patch("builtins.input", return_value="n") as prompt:
                with self.assertRaises(infra.InterfaceValidationError) as ctx:
                    scheduler.bootstrap(RuntimeRobot().get_obs())

        self.assertEqual(prompt.call_count, 1)
        self.assertIn("RTC startup warmup aborted", str(ctx.exception))
        self.assertEqual(len(scheduler._buffer), 0)
        self.assertFalse(scheduler.latency_estimate_ready())

    def test_chunk_scheduler_startup_rejects_invalid_rtc_execution_window_structure(
        self,
    ) -> None:
        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            return [arm_action(1.0), arm_action(2.0), arm_action(3.0)]

        scheduler = ChunkScheduler(
            action_source=action_source,
            steps_before_request=1,
            execution_steps=3,
            control_period_s=0.02,
            warmup_requests=1,
            profile_delay_requests=1,
            enable_rtc=True,
        )

        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            scheduler.bootstrap(RuntimeRobot().get_obs())

        self.assertIn(
            "execution_steps < chunk_total_length - steps_before_request",
            str(ctx.exception),
        )

    def test_chunk_scheduler_startup_warns_when_delay_exceeds_execution_steps(
        self,
    ) -> None:
        frame = infra.Frame(images={}, state={})

        def action_source(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            return [
                arm_action(1.0),
                arm_action(2.0),
                arm_action(3.0),
                arm_action(4.0),
            ]

        scheduler = ChunkScheduler(
            action_source=action_source,
            steps_before_request=0,
            execution_steps=3,
            fixed_latency_steps=4.0,
            enable_rtc=True,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            action, refreshed = scheduler.next_action(frame, prefetch_async=True)

        self.assertEqual(arm_value(action), 1.0)
        self.assertTrue(refreshed)
        self.assertTrue(
            any("exceeds execution_steps" in str(warning.message) for warning in caught)
        )

    def test_chunk_scheduler_steady_state_warns_when_delay_exceeds_execution_steps(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            steps_before_request=0,
            execution_steps=2,
            enable_rtc=True,
            fixed_latency_steps=3.0,
            clock=lambda: 123.0,
        )
        scheduler._buffer = deque(
            [
                arm_action(3.0),
                arm_action(4.0),
                arm_action(5.0),
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            job = scheduler._build_request_job(include_latency=True)

        self.assertEqual(job.request.execute_horizon, 2)
        self.assertEqual(job.request.inference_delay, 2)
        self.assertTrue(
            any("exceeds execution_steps" in str(warning.message) for warning in caught)
        )

    def test_sync_overlap_runtime_enable_rtc_discards_warmup_chunk(self) -> None:
        robot = RuntimeRobot()
        policy = RtcLoggingChunkPolicy()
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.SYNC,
            steps_before_request=0,
            execution_steps=3,
            enable_rtc=True,
        )

        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
        )

        self.assertEqual(arm_value(result.action), 1.0)
        self.assertEqual(len(policy.requests), 3)
        self.assertIsNone(policy.requests[0].rtc_args)
        self.assertIsNone(policy.requests[0].prev_action_chunk)
        rtc_args = policy.requests[-1].rtc_args
        self.assertIsNotNone(rtc_args)
        assert rtc_args is not None
        self.assertEqual(rtc_args.inference_delay, 1)
        self.assertEqual(rtc_args.execute_horizon, 3)
        self.assertEqual(
            [arm_value(action) for action in rtc_args.prev_action_chunk],
            [1.0, 2.0, 3.0, 3.0],
        )

    def test_async_scheduler_drops_only_executed_steps_after_wall_clock_wait(
        self,
    ) -> None:
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
            steps_before_request=5,
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
        self.assertEqual(scheduler.estimated_latency_steps(), 0)

    def test_chunk_scheduler_latency_steps_offset_flows_through_interpolation_projection(
        self,
    ) -> None:
        scheduler = ChunkScheduler(
            interpolation_steps=2,
            enable_rtc=True,
            execution_steps=1,
            latency_steps_offset=4,
        )
        scheduler._integrate_completed_chunk(
            _CompletedChunk(
                request=make_chunk_request(
                    request_step=0,
                    request_time_s=0.0,
                    active_chunk_length=0,
                    remaining_steps=0,
                    latency_steps=0,
                ),
                prepared_actions=[
                    arm_action(10.0),
                    arm_action(12.0),
                ],
                source_plan_length=2,
            )
        )
        scheduler._latency_steps_estimate = 2.0

        job = scheduler._build_request_job(include_latency=True)

        self.assertEqual(job.request.latency_steps, 5)
        self.assertEqual(job.request.inference_delay, 1)
        self.assertEqual(job.request.execute_horizon, 1)


if __name__ == "__main__":
    unittest.main()
