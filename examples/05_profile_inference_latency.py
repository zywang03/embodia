"""Example 5: capture an async realtime runtime profile.

Run with:

    PYTHONPATH=src python examples/05_profile_inference_latency.py --target-hz 50
    PYTHONPATH=src python examples/05_profile_inference_latency.py --target-hz 60 --policy-latency-ms 25
    PYTHONPATH=src python examples/05_profile_inference_latency.py --target-hz 50 --print-requests
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import inferaxis as infra
import numpy as np


class YourRobot:
    """Plain local executor used by the profiling helpers."""

    def __init__(self) -> None:
        self.last_action: infra.Action | None = None

    def get_obs(self) -> infra.Frame:
        return infra.Frame(
            images={"YOUR_OWN_front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={
                "YOUR_OWN_arm": np.zeros(6, dtype=np.float64),
                "YOUR_OWN_gripper": np.array([0.5], dtype=np.float64),
            },
        )

    def send_action(self, action: infra.Action) -> infra.Action:
        self.last_action = action
        return action


class YourPolicy:
    """Plain policy returning future actions through the same infer entrypoint."""

    def __init__(self, *, latency_s: float, chunk_steps: int) -> None:
        self.latency_s = latency_s
        self.chunk_steps = chunk_steps
        self.step_index = 0

    def infer(
        self,
        frame: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        del frame, request
        time.sleep(self.latency_s)
        base = float(self.step_index)
        self.step_index += self.chunk_steps

        return [
            infra.Action(
                commands={
                    "YOUR_OWN_arm": infra.Command(
                        command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                        value=np.full(6, base + offset, dtype=np.float64),
                    ),
                    "YOUR_OWN_gripper": infra.Command(
                        command=infra.BuiltinCommandKind.GRIPPER_POSITION,
                        value=np.array([0.5], dtype=np.float64),
                    ),
                }
            )
            for offset in range(self.chunk_steps)
        ]


def print_profile_requests(profile_payload: dict[str, object]) -> None:
    """Print compact request rows from one runtime profile JSON payload."""

    for request in profile_payload.get("requests", []):
        if not isinstance(request, dict):
            continue
        status = "accepted" if request.get("accepted") else "profiled"
        if request.get("dropped_as_stale"):
            status = "dropped"
        if request.get("error") is not None:
            status = "error"

        duration_s = request.get("request_duration_s")
        duration_text = (
            "n/a" if duration_s is None else f"{float(duration_s) * 1000.0:.3f}"
        )
        print(
            "[runtime_profile] "
            f"request={request.get('request_index')} "
            f"status={status} "
            f"step={request.get('request_step')} "
            f"duration_ms={duration_text} "
            f"returned={request.get('returned_chunk_length')} "
            f"accepted={request.get('accepted_chunk_length')} "
            f"latency_hint_raw_steps={request.get('latency_hint_raw_steps')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the live profile emitted by InferenceRuntime.async_realtime. "
            "In this example, --policy-latency-ms and --chunk-steps only control "
            "the fake policy used for the demo."
        ),
    )
    # fmt: off
    parser.add_argument("--target-hz", type=float, required=True, help="Target control hz for the async realtime runtime.")
    parser.add_argument("--startup-ignore-samples", type=int, default=1, help="Number of startup warmup requests before latency profiling.")
    parser.add_argument("--stable-sample-count", type=int, default=4, help="Number of profile-delay requests used to seed latency.")
    parser.add_argument("--policy-latency-ms", type=float, default=15.0, help="Fake policy sleep used only by this example to simulate inference latency.")
    parser.add_argument(
        "--chunk-steps", type=int, default=50,
        help="Fake policy chunk length used only by this example. Default 50 matches the most common pi06star deployment/default setting.",
    )
    parser.add_argument(
        "--steps-before-request", type=int, default=0,
        help="Number of raw chunk steps to execute after a chunk is accepted before starting the next request.",
    )
    parser.add_argument("--execute-action", action="store_true", help="Also call act_fn(...) so step timing includes local action execution.")
    parser.add_argument("--print-requests", action="store_true", help="Print one compact summary for each runtime profile request.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for runtime_profile.json/html.")
    # fmt: on
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = YourRobot()
    policy = YourPolicy(
        latency_s=args.policy_latency_ms / 1000.0,
        chunk_steps=args.chunk_steps,
    )

    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    runtime = infra.InferenceRuntime.async_realtime(
        profile=True,
        profile_output_dir=output_dir,
        control_hz=args.target_hz,
        warmup_requests=args.startup_ignore_samples,
        profile_delay_requests=args.stable_sample_count,
        steps_before_request=args.steps_before_request,
    )

    try:
        result = infra.run_step(
            observe_fn=robot.get_obs,
            act_fn=robot.send_action,
            act_src_fn=policy.infer,
            runtime=runtime,
            execute_action=args.execute_action,
        )
    finally:
        runtime.close()

    assert runtime.profile_output_dir is not None
    profile_output_dir = Path(runtime.profile_output_dir)
    profile_json_path = profile_output_dir / "runtime_profile.json"
    profile_html_path = profile_output_dir / "runtime_profile.html"
    profile_payload = json.loads(profile_json_path.read_text(encoding="utf-8"))

    if args.print_requests:
        print_profile_requests(profile_payload)

    print("runtime_profile_summary:")
    print(json.dumps(profile_payload["summary"], indent=2, sort_keys=True))
    print()
    print("summary:")
    print(
        "  runtime_profile_request_count:", profile_payload["summary"]["total_requests"]
    )
    print(
        "  startup_profile_request_count:",
        args.startup_ignore_samples + args.stable_sample_count,
    )
    print("  fake_policy_chunk_steps:", args.chunk_steps)
    print("  fake_policy_latency_ms:", args.policy_latency_ms)
    print("  steps_before_request:", args.steps_before_request)
    print("  target_hz:", args.target_hz)
    print("  plan_refreshed:", result.plan_refreshed)
    print("  last_action:", robot.last_action)
    print("  runtime_profile_dir:", profile_output_dir)
    print("  wrote_runtime_profile_json:", profile_json_path)
    print("  wrote_runtime_profile_html:", profile_html_path)


if __name__ == "__main__":
    main()
