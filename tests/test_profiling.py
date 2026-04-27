"""Profiling-domain tests for inference runtime utilities."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import inferaxis as infra

from inferaxis.core.errors import InterfaceValidationError
from inferaxis.runtime.inference.scheduler.buffers import (
    ExecutionCursor,
    RawChunkBuffer,
)
from inferaxis.runtime.inference.profiling.models import (
    RuntimeInferenceProfile,
    RuntimeProfileActionCommand,
    RuntimeProfileActionStep,
    RuntimeProfileChunkAction,
    RuntimeProfileRequest,
)
from inferaxis.runtime.inference.profiling.render_runtime_html import (
    _runtime_profile_html,
)

from helpers import (
    RuntimePolicy,
    RuntimeRobot,
)


class ProfilingTests(unittest.TestCase):
    """Coverage for live runtime profiling helpers."""

    def test_async_runtime_profile_resolves_default_output_dir(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "pathlib.Path.cwd",
                return_value=Path(tmpdir),
            ),
        ):
            runtime = infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                profile=True,
            )

            assert runtime.profile_output_dir is not None
            self.assertEqual(runtime.profile_output_dir.parent.name, "profiles")
            self.assertTrue(
                runtime.profile_output_dir.name.startswith("inference-runtime-")
            )
            runtime.close()

    def test_profiling_package_exposes_live_runtime_profile_models(self) -> None:
        import inferaxis.runtime.inference.profiling as profiling_module

        self.assertIs(
            profiling_module.RuntimeInferenceProfile,
            RuntimeInferenceProfile,
        )
        self.assertIs(
            profiling_module.RuntimeProfileRequest,
            RuntimeProfileRequest,
        )
        self.assertFalse(hasattr(profiling_module, "profile_sync_inference"))
        self.assertFalse(hasattr(profiling_module, "recommend_inference_mode"))

    def test_removed_sync_profile_helpers_are_not_public_api(self) -> None:
        import inferaxis.runtime.inference as inference_module

        self.assertFalse(hasattr(infra, "profile_sync_inference"))
        self.assertFalse(hasattr(infra, "recommend_inference_mode"))
        self.assertFalse(hasattr(inference_module, "profile_sync_inference"))
        self.assertFalse(hasattr(inference_module, "recommend_inference_mode"))

    def test_async_realtime_profile_accepts_plain_local_executor(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = infra.InferenceRuntime.async_realtime(
                profile=True,
                profile_output_dir=tmpdir,
                control_hz=50.0,
                warmup_requests=1,
                profile_delay_requests=2,
                steps_before_request=0,
            )

            result = infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
                pace_control=False,
            )
            runtime.close()

            payload = json.loads(
                (Path(tmpdir) / "runtime_profile.json").read_text(encoding="utf-8")
            )

        self.assertTrue(result.plan_refreshed)
        self.assertEqual(payload["summary"]["total_requests"], 4)
        self.assertEqual(payload["summary"]["accepted_requests"], 1)
        self.assertEqual(payload["summary"]["average_returned_chunk_length"], 2.0)
        self.assertEqual(payload["config"]["control_hz"], 50.0)
        self.assertEqual(payload["config"]["warmup_requests"], 1)
        self.assertEqual(payload["config"]["profile_delay_requests"], 2)

    def test_async_runtime_profile_writes_live_request_report(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                profile=True,
                profile_output_dir=tmpdir,
                steps_before_request=2,
            )

            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            runtime.close()

            json_path = Path(tmpdir) / "runtime_profile.json"
            html_path = Path(tmpdir) / "runtime_profile.html"
            self.assertTrue(json_path.exists())
            self.assertTrue(html_path.exists())
            self.assertFalse((Path(tmpdir) / "runtime_profile.svg").exists())

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_requests"], 1)
            self.assertEqual(payload["summary"]["accepted_requests"], 1)
            self.assertEqual(payload["summary"]["failed_requests"], 0)
            self.assertEqual(payload["config"]["profile_output_dir"], tmpdir)

            request = payload["requests"][0]
            self.assertEqual(request["request_index"], 0)
            self.assertEqual(request["request_step"], 0)
            self.assertEqual(request["returned_chunk_length"], 2)
            self.assertEqual(request["accepted_chunk_length"], 2)
            self.assertTrue(request["accepted"])
            self.assertFalse(request["dropped_as_stale"])
            self.assertIsNone(request["error"])

            self.assertEqual(len(payload["action_steps"]), 2)
            first_action_step = payload["action_steps"][0]
            self.assertEqual(first_action_step["step_index"], 0)
            self.assertTrue(first_action_step["plan_refreshed"])
            self.assertEqual(first_action_step["buffer_size"], 1)
            self.assertEqual(first_action_step["execution_buffer_size"], 1)
            self.assertEqual(
                first_action_step["action_commands"][0]["target"],
                "arm",
            )
            self.assertEqual(
                first_action_step["action_commands"][0]["value"],
                [1.0] * 6,
            )
            self.assertEqual(len(payload["chunk_actions"]), 2)
            self.assertEqual(payload["chunk_actions"][0]["request_index"], 0)
            self.assertEqual(payload["chunk_actions"][0]["status"], "accepted")
            self.assertEqual(
                payload["chunk_actions"][0]["commands"][0]["value"], [1.0] * 6
            )
            self.assertEqual(payload["chunk_actions"][1]["status"], "accepted")

            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("Plotly.newPlot", html_text)
            self.assertIn("InferenceRuntime Live Profile", html_text)
            self.assertIn("Step Trace: buffer size + chunk actions", html_text)
            self.assertIn("buffer_size", html_text)
            self.assertIn("All channels", html_text)
            self.assertIn("arm[0]", html_text)
            self.assertIn("arm[5]", html_text)
            self.assertIn("chunk 0 accepted", html_text)

    def test_async_runtime_profile_reads_buffer_counts_without_snapshots(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                profile=True,
                profile_output_dir=tmpdir,
                steps_before_request=99,
            )

            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
                pace_control=False,
            )
            assert runtime._chunk_scheduler is not None

            original_remaining_actions = RawChunkBuffer.remaining_actions
            original_remaining_segment_actions = (
                ExecutionCursor.remaining_segment_actions
            )

            def fail_remaining_actions(self: RawChunkBuffer) -> list[infra.Action]:
                del self
                raise AssertionError("profile hot path built raw buffer snapshot")

            def fail_remaining_segment_actions(
                self: ExecutionCursor,
            ) -> list[infra.Action]:
                del self
                raise AssertionError("profile hot path built execution snapshot")

            RawChunkBuffer.remaining_actions = fail_remaining_actions
            ExecutionCursor.remaining_segment_actions = fail_remaining_segment_actions
            try:
                infra.run_step(
                    observe_fn=robot.get_obs,
                    act_fn=robot.send_action,
                    act_src_fn=policy.infer,
                    runtime=runtime,
                    pace_control=False,
                )
            finally:
                RawChunkBuffer.remaining_actions = original_remaining_actions
                ExecutionCursor.remaining_segment_actions = (
                    original_remaining_segment_actions
                )
                runtime.close()

    def test_async_runtime_profile_closes_completed_pending_request_cleanly(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                profile=True,
                profile_output_dir=tmpdir,
                control_hz=50.0,
                steps_before_request=0,
                warmup_requests=0,
                profile_delay_requests=0,
            )

            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            scheduler = runtime._chunk_scheduler
            self.assertIsNotNone(scheduler)
            pending = scheduler._pending_future  # type: ignore[union-attr]
            self.assertIsNotNone(pending)
            deadline_s = time.monotonic() + 1.0
            while not pending.done():  # type: ignore[union-attr]
                if time.monotonic() >= deadline_s:
                    self.fail("pending profile request did not finish in time")
                time.sleep(0.001)

            runtime.close()

            payload = json.loads(
                (Path(tmpdir) / "runtime_profile.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["summary"]["total_requests"], 2)
            self.assertEqual(payload["summary"]["accepted_requests"], 1)
            self.assertEqual(payload["summary"]["failed_requests"], 0)

            pending_request = payload["requests"][1]
            self.assertFalse(pending_request["accepted"])
            self.assertIsNone(pending_request["error"])
            self.assertEqual(pending_request["returned_chunk_length"], 2)
            self.assertEqual(pending_request["accepted_chunk_length"], 0)
            request_one_actions = [
                action
                for action in payload["chunk_actions"]
                if action["request_index"] == 1
            ]
            self.assertEqual(
                [action["status"] for action in request_one_actions],
                ["unused", "unused"],
            )

            html_text = (Path(tmpdir) / "runtime_profile.html").read_text(
                encoding="utf-8",
            )
            self.assertNotIn("chunk 1 unused", html_text)

    def test_async_runtime_profile_marks_dropped_chunk_prefix_actions(self) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                profile=True,
                profile_output_dir=tmpdir,
                control_hz=50.0,
                steps_before_request=0,
                warmup_requests=0,
                profile_delay_requests=0,
            )

            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            scheduler = runtime._chunk_scheduler
            self.assertIsNotNone(scheduler)
            pending = scheduler._pending_future  # type: ignore[union-attr]
            self.assertIsNotNone(pending)
            pending.result(timeout=1.0)  # type: ignore[union-attr]
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            runtime.close()

            payload = json.loads(
                (Path(tmpdir) / "runtime_profile.json").read_text(encoding="utf-8")
            )
            request_one_actions = [
                action
                for action in payload["chunk_actions"]
                if action["request_index"] == 1
            ]
            self.assertEqual(
                [action["status"] for action in request_one_actions],
                ["dropped", "accepted"],
            )
            self.assertEqual(
                [action["step_index"] for action in request_one_actions],
                [0, 1],
            )

            html_text = (Path(tmpdir) / "runtime_profile.html").read_text(
                encoding="utf-8",
            )
            self.assertIn("chunk 1 dropped", html_text)
            self.assertIn("chunk 1 accepted", html_text)
            self.assertIn('"dash":"dash"', html_text)

    def test_runtime_profile_html_hides_future_unexecuted_chunk_tail(self) -> None:
        executed_command = RuntimeProfileActionCommand(
            target="arm",
            command="cartesian_pose_delta",
            value=[11.125] * 6,
            ref_frame=None,
        )
        later_executed_command = RuntimeProfileActionCommand(
            target="arm",
            command="cartesian_pose_delta",
            value=[22.25] * 6,
            ref_frame=None,
        )
        future_tail_command = RuntimeProfileActionCommand(
            target="arm",
            command="cartesian_pose_delta",
            value=[98765.4321] * 6,
            ref_frame=None,
        )
        profile = RuntimeInferenceProfile(
            mode="async",
            config={},
            requests=[
                RuntimeProfileRequest(
                    request_index=0,
                    request_step=0,
                    launch_control_step=0,
                    launch_time_s=0.0,
                    reply_time_s=0.01,
                    accepted_time_s=0.02,
                    request_duration_s=0.01,
                    prepare_duration_s=0.0,
                    accept_delay_s=0.01,
                    usable_latency_s=0.02,
                    latency_hint_raw_steps=0,
                    waited_control_steps=0,
                    stale_raw_steps=0,
                    returned_chunk_length=4,
                    accepted_chunk_length=4,
                    accepted=True,
                    dropped_as_stale=False,
                    error=None,
                )
            ],
            action_steps=[
                RuntimeProfileActionStep(
                    step_index=0,
                    action_time_s=0.0,
                    plan_refreshed=True,
                    control_wait_s=0.0,
                    buffer_size=4,
                    execution_buffer_size=4,
                    raw_commands=[executed_command],
                    action_commands=[executed_command],
                ),
                RuntimeProfileActionStep(
                    step_index=1,
                    action_time_s=0.02,
                    plan_refreshed=False,
                    control_wait_s=0.0,
                    buffer_size=3,
                    execution_buffer_size=3,
                    raw_commands=[later_executed_command],
                    action_commands=[later_executed_command],
                ),
            ],
            chunk_actions=[
                RuntimeProfileChunkAction(
                    request_index=0,
                    request_step=0,
                    action_index=0,
                    step_index=0,
                    status="accepted",
                    commands=[executed_command],
                ),
                RuntimeProfileChunkAction(
                    request_index=0,
                    request_step=0,
                    action_index=1,
                    step_index=1,
                    status="accepted",
                    commands=[later_executed_command],
                ),
                RuntimeProfileChunkAction(
                    request_index=0,
                    request_step=0,
                    action_index=2,
                    step_index=2,
                    status="accepted",
                    commands=[future_tail_command],
                ),
                RuntimeProfileChunkAction(
                    request_index=0,
                    request_step=0,
                    action_index=3,
                    step_index=3,
                    status="accepted",
                    commands=[future_tail_command],
                ),
            ],
        )

        html_text = _runtime_profile_html(profile)

        self.assertIn("11.125", html_text)
        self.assertIn("22.25", html_text)
        self.assertNotIn("98765.4321", html_text)

    def test_async_runtime_profile_tracks_startup_probe_requests_without_errors(
        self,
    ) -> None:
        robot = RuntimeRobot()
        policy = RuntimePolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                profile=True,
                profile_output_dir=tmpdir,
                control_hz=50.0,
                warmup_requests=1,
                profile_delay_requests=1,
                steps_before_request=2,
            )

            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            runtime.close()

            payload = json.loads(
                (Path(tmpdir) / "runtime_profile.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["summary"]["total_requests"], 2)
            self.assertEqual(payload["summary"]["accepted_requests"], 1)
            self.assertEqual(payload["summary"]["failed_requests"], 0)
            self.assertFalse(payload["requests"][0]["accepted"])
            self.assertIsNone(payload["requests"][0]["error"])
            self.assertTrue(payload["requests"][1]["accepted"])

    def test_async_runtime_profile_records_request_errors(self) -> None:
        robot = RuntimeRobot()

        def failing_policy(
            obs: infra.Frame,
            request: infra.ChunkRequest,
        ) -> list[infra.Action]:
            del obs, request
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                profile=True,
                profile_output_dir=tmpdir,
                steps_before_request=0,
            )

            with self.assertRaises(InterfaceValidationError) as ctx:
                infra.run_step(
                    observe_fn=robot.get_obs,
                    act_fn=robot.send_action,
                    act_src_fn=failing_policy,
                    runtime=runtime,
                )

            self.assertIn("RuntimeError: boom", str(ctx.exception))
            runtime.close()

            payload = json.loads(
                (Path(tmpdir) / "runtime_profile.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["summary"]["failed_requests"], 1)
            self.assertEqual(payload["summary"]["accepted_requests"], 0)
            self.assertIn("RuntimeError: boom", payload["requests"][0]["error"])


if __name__ == "__main__":
    unittest.main()
