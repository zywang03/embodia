"""Profiling-domain tests for inference runtime utilities."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import inferaxis as infra

from inferaxis.runtime.inference.profiling import (
    _ProfiledRequestSample,
    _build_async_buffer_trace,
)

from helpers import (
    DeterministicClock,
    PlainRuntimeExecutor,
    PlanningSource,
    RuntimePolicy,
    RuntimeRobot,
    SingleActionChunkPolicy,
    arm_action,
    arm_value,
    assert_array_equal,
    make_profile_clock,
)


class ProfilingTests(unittest.TestCase):
    """Coverage for runtime profiling and recommendation helpers."""

    def test_async_runtime_profile_resolves_default_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "pathlib.Path.cwd",
            return_value=Path(tmpdir),
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

    def test_profiling_package_reexports_existing_trace_helpers(self) -> None:
        import inferaxis.runtime.inference.profiling as profiling_module

        self.assertIs(profiling_module._ProfiledRequestSample, _ProfiledRequestSample)
        self.assertIs(profiling_module._build_async_buffer_trace, _build_async_buffer_trace)

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
            self.assertEqual(first_action_step["execution_buffer_size"], 0)
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
            self.assertEqual(payload["chunk_actions"][0]["commands"][0]["value"], [1.0] * 6)
            self.assertEqual(payload["chunk_actions"][1]["status"], "accepted")

            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("Plotly.newPlot", html_text)
            self.assertIn("InferenceRuntime Live Profile", html_text)
            self.assertIn("Step Trace: buffer size + chunk actions", html_text)
            self.assertIn("buffer_size", html_text)
            self.assertIn("arm[0]", html_text)
            self.assertIn("chunk 0 accepted", html_text)

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
            self.assertIn(
                "unused",
                (Path(tmpdir) / "runtime_profile.html").read_text(encoding="utf-8"),
            )

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

            with self.assertRaises(infra.InterfaceValidationError) as ctx:
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
            steps_before_request=0,
            clock=make_profile_clock(
                step_durations=[0.02, 0.02, 0.02, 0.02],
                inference_durations=[0.01, 0.009, 0.008, 0.007],
            ),
        )

        self.assertAlmostEqual(profile.estimated_chunk_steps, 2.0)
        self.assertAlmostEqual(profile.estimated_inference_time_s, 0.0085)
        self.assertAlmostEqual(profile.estimated_max_buffered_hz, 2.0 / 0.0085)
        self.assertEqual(profile.steps_before_request, 0)
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
                steps_before_request=0,
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
            steps_before_request=5,
        )

        self.assertGreaterEqual(len(trace.requests), 2)
        second_request = trace.requests[1]
        self.assertEqual(second_request.observed_latency_steps, 8)
        self.assertEqual(second_request.executed_wait_raw_steps, 5)
        self.assertEqual(second_request.aligned_chunk_steps, 5)
        self.assertTrue(any(step.underrun for step in trace.steps))


if __name__ == "__main__":
    unittest.main()
