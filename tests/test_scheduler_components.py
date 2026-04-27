"""Focused tests for scheduler v2 internal components."""

from __future__ import annotations

from concurrent.futures import Future
import unittest

import inferaxis as infra

from inferaxis.runtime.inference.scheduler.buffers import (
    ExecutionCursor,
    RawChunkBuffer,
)
from inferaxis.runtime.inference.scheduler.latency import LatencyTracker
from inferaxis.runtime.inference.scheduler.pipeline import RequestPipeline
from inferaxis.runtime.inference.scheduler.rtc import RtcWindowBuilder

from helpers import arm_action, arm_value


class RawChunkBufferTests(unittest.TestCase):
    def test_buffer_current_action_reports_empty_buffer(self) -> None:
        buffer = RawChunkBuffer()

        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            buffer.current_action()

        self.assertIn("RawChunkBuffer", str(ctx.exception))

    def test_buffer_accepts_chunk_and_drops_stale_prefix_by_index(self) -> None:
        buffer = RawChunkBuffer()

        buffer.accept_chunk(
            actions=[arm_action(1.0), arm_action(2.0), arm_action(3.0)],
            request_step=3,
            current_raw_step=5,
            source_plan_length=3,
        )

        self.assertEqual(buffer.remaining_raw_count, 1)
        self.assertEqual(arm_value(buffer.current_action()), 3.0)
        self.assertEqual(buffer.active_source_plan_length, 3)

    def test_buffer_advance_tracks_waited_and_global_steps(self) -> None:
        buffer = RawChunkBuffer()
        buffer.accept_chunk(
            actions=[arm_action(1.0), arm_action(2.0)],
            request_step=0,
            current_raw_step=0,
            source_plan_length=2,
        )

        buffer.advance_raw_step()

        self.assertEqual(buffer.global_step, 1)
        self.assertEqual(buffer.active_chunk_waited_raw_steps, 1)
        self.assertEqual(arm_value(buffer.current_action()), 2.0)


class ExecutionCursorTests(unittest.TestCase):
    def test_cursor_tracks_remaining_interpolation_segment_steps(self) -> None:
        buffer = RawChunkBuffer()
        buffer.accept_chunk(
            actions=[arm_action(0.0), arm_action(9.0)],
            request_step=0,
            current_raw_step=0,
            source_plan_length=2,
        )
        cursor = ExecutionCursor(buffer=buffer, interpolation_steps=2)

        self.assertEqual(cursor.remaining_segment_steps, 3)
        self.assertEqual(arm_value(cursor.next_action()), 0.0)
        self.assertEqual(cursor.remaining_segment_steps, 2)
        self.assertEqual(arm_value(cursor.next_action()), 3.0)
        self.assertEqual(cursor.remaining_segment_steps, 1)
        self.assertEqual(arm_value(cursor.next_action()), 6.0)
        self.assertEqual(cursor.remaining_segment_steps, 1)
        self.assertEqual(arm_value(cursor.next_action()), 9.0)
        self.assertFalse(buffer.has_actions)

    def test_cursor_last_raw_action_has_one_remaining_step(self) -> None:
        buffer = RawChunkBuffer()
        buffer.accept_chunk(
            actions=[arm_action(7.0)],
            request_step=0,
            current_raw_step=0,
            source_plan_length=1,
        )
        cursor = ExecutionCursor(buffer=buffer, interpolation_steps=3)

        self.assertEqual(cursor.remaining_segment_steps, 1)
        self.assertEqual(arm_value(cursor.next_action()), 7.0)
        self.assertEqual(cursor.remaining_segment_steps, 0)
        self.assertFalse(buffer.has_actions)

    def test_cursor_no_interpolation_emits_without_extra_segment(self) -> None:
        buffer = RawChunkBuffer()
        buffer.accept_chunk(
            actions=[arm_action(1.0), arm_action(2.0)],
            request_step=0,
            current_raw_step=0,
            source_plan_length=2,
        )
        cursor = ExecutionCursor(buffer=buffer, interpolation_steps=0)

        first = cursor.next_action()
        second = cursor.next_action()

        self.assertEqual(arm_value(first), 1.0)
        self.assertEqual(arm_value(second), 2.0)
        self.assertFalse(buffer.has_actions)

    def test_cursor_interpolates_between_raw_actions_on_demand(self) -> None:
        buffer = RawChunkBuffer()
        buffer.accept_chunk(
            actions=[arm_action(0.0), arm_action(10.0)],
            request_step=0,
            current_raw_step=0,
            source_plan_length=2,
        )
        cursor = ExecutionCursor(buffer=buffer, interpolation_steps=1)

        first = cursor.next_action()
        interpolated = cursor.next_action()
        final = cursor.next_action()

        self.assertEqual(arm_value(first), 0.0)
        self.assertEqual(arm_value(interpolated), 5.0)
        self.assertEqual(arm_value(final), 10.0)
        self.assertFalse(buffer.has_actions)


class LatencyTrackerTests(unittest.TestCase):
    def test_tracker_control_steps_for_raw_count_rejects_bool(self) -> None:
        tracker = LatencyTracker(interpolation_steps=1)

        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            tracker.control_steps_for_raw_count(True)

        self.assertEqual(
            str(ctx.exception),
            "raw_steps must be an int, got bool.",
        )

    def test_tracker_control_steps_for_raw_count_rejects_negative(self) -> None:
        tracker = LatencyTracker(interpolation_steps=1)

        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            tracker.control_steps_for_raw_count(-1)

        self.assertEqual(
            str(ctx.exception),
            "raw_steps must be >= 0, got -1.",
        )

    def test_tracker_control_steps_for_raw_count_accepts_zero(self) -> None:
        tracker = LatencyTracker(interpolation_steps=1)

        self.assertEqual(tracker.control_steps_for_raw_count(0), 0)

    def test_tracker_projects_control_latency_to_raw_steps(self) -> None:
        tracker = LatencyTracker(interpolation_steps=1)

        raw_delay = tracker.project_control_latency_to_raw_steps(
            control_latency_steps=3,
            raw_count=4,
            execution_buffer_steps=0,
        )

        self.assertEqual(raw_delay, 2)

    def test_tracker_clamps_negative_requested_latency_offset(self) -> None:
        tracker = LatencyTracker(latency_steps_offset=-10)

        self.assertEqual(
            tracker.estimated_request_latency_steps(
                control_latency_steps=2,
                raw_count=1,
                execution_buffer_steps=0,
            ),
            0,
        )

    def test_tracker_updates_after_warmup_requests(self) -> None:
        tracker = LatencyTracker(
            latency_ema_beta=0.5,
            initial_latency_steps=0.0,
            warmup_requests=1,
        )

        tracker.update(waited_steps=4)
        tracker.update(waited_steps=6)

        self.assertEqual(tracker.estimated_latency_steps(), 6)
        self.assertTrue(tracker.latency_estimate_ready())

    def test_tracker_honors_fixed_latency_steps(self) -> None:
        tracker = LatencyTracker(fixed_latency_steps=3.0)

        tracker.update(waited_steps=10)

        self.assertEqual(tracker.estimated_latency_steps(), 3)
        self.assertTrue(tracker.latency_estimate_ready())


class RtcWindowBuilderTests(unittest.TestCase):
    def test_builder_pads_prev_action_chunk_to_locked_length(self) -> None:
        builder = RtcWindowBuilder(
            enabled=True,
            execution_steps=2,
            steps_before_request=0,
        )
        source = [arm_action(1.0), arm_action(2.0), arm_action(3.0)]
        builder.lock_chunk_total_length(len(source))

        rtc_args = builder.build_args(
            remaining_chunk=source[:2],
            inference_delay=3,
        )

        assert rtc_args is not None
        self.assertEqual(len(rtc_args.prev_action_chunk), 3)
        self.assertEqual(rtc_args.inference_delay, 2)
        self.assertEqual(rtc_args.execute_horizon, 2)
        self.assertEqual(arm_value(rtc_args.prev_action_chunk[-1]), 2.0)

    def test_builder_rejects_changed_chunk_length(self) -> None:
        builder = RtcWindowBuilder(
            enabled=True,
            execution_steps=2,
            steps_before_request=0,
        )

        builder.lock_chunk_total_length(4)

        with self.assertRaises(infra.InterfaceValidationError):
            builder.lock_chunk_total_length(3)


class RequestPipelineTests(unittest.TestCase):
    def test_pipeline_tracks_pending_future(self) -> None:
        pipeline = RequestPipeline()

        self.assertFalse(pipeline.has_pending)
        pipeline.pending = Future()
        self.assertTrue(pipeline.has_pending)
        pipeline.clear_pending()
        self.assertFalse(pipeline.has_pending)

    def test_pipeline_manages_pending_executor_lifecycle(self) -> None:
        pipeline = RequestPipeline()

        self.assertFalse(pipeline.has_ready_pending)

        executor = pipeline.ensure_executor()
        self.assertIs(executor, pipeline.executor)

        future: Future[object] = Future()
        future.set_result(object())
        pipeline.pending = future

        self.assertTrue(pipeline.has_pending)
        self.assertTrue(pipeline.has_ready_pending)

        pipeline.close()

        self.assertIsNone(pipeline.pending)
        self.assertIsNone(pipeline.executor)
