"""Focused tests for scheduler v2 internal components."""

from __future__ import annotations

import unittest

import inferaxis as infra

from inferaxis.runtime.inference.scheduler.buffers import (
    ExecutionCursor,
    RawChunkBuffer,
)

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
