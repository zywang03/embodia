"""Example 5: profile inference latency and estimate sustainable control hz.

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one callable policy against one required target control hz. "
            "In this example, --policy-latency-ms and --chunk-steps only control "
            "the fake policy used for the demo. The profiler itself now uses "
            "startup_ignore_samples + stable_sample_count total requests."
        ),
    )
    # fmt: off
    parser.add_argument("--target-hz", type=float, required=True, help="Required target control hz for the closed-loop system.")
    parser.add_argument("--startup-ignore-samples", type=int, default=1, help="Number of early inference samples to ignore.")
    parser.add_argument("--stable-sample-count", type=int, default=4, help="Number of stable inference samples kept for the final estimate.")
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
    parser.add_argument("--print-requests", action="store_true", help="Print one compact observation summary for each profiled request.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for writing profile/recommendation JSON.")
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

    profile = infra.profile_sync_inference(
        observe_fn=robot.get_obs,
        target_hz=args.target_hz,
        act_fn=robot.send_action,
        act_src_fn=policy.infer,
        execute_action=args.execute_action,
        startup_ignore_inference_samples=args.startup_ignore_samples,
        stable_inference_sample_count=args.stable_sample_count,
        steps_before_request=args.steps_before_request,
        output_path=(output_dir / "profile.json") if output_dir is not None else None,
        request_log_fn=print if args.print_requests else None,
    )
    if output_dir is not None:
        profile.write_async_buffer_trace_json(output_dir / "buffer_trace.json")

    recommendation = infra.recommend_inference_mode(
        observe_fn=robot.get_obs,
        act_fn=robot.send_action,
        act_src_fn=policy.infer,
        target_hz=args.target_hz,
        execute_action=args.execute_action,
        startup_ignore_inference_samples=args.startup_ignore_samples,
        stable_inference_sample_count=args.stable_sample_count,
        steps_before_request=args.steps_before_request,
        output_path=(output_dir / "recommendation.json")
        if output_dir is not None
        else None,
        request_log_fn=print if args.print_requests else None,
    )

    print("profile:")
    print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
    print()
    print("recommendation:")
    print(json.dumps(recommendation.to_dict(), indent=2, sort_keys=True))
    print()
    print("summary:")
    print(
        "  profiled_request_count:",
        args.startup_ignore_samples + args.stable_sample_count,
    )
    print("  fake_policy_chunk_steps:", args.chunk_steps)
    print("  fake_policy_latency_ms:", args.policy_latency_ms)
    print("  steps_before_request:", args.steps_before_request)
    print("  estimated_chunk_steps:", profile.estimated_chunk_steps)
    print("  estimated_max_buffered_hz:", profile.estimated_max_buffered_hz)
    print("  target_hz:", args.target_hz)
    print("  recommended_mode:", recommendation.recommended_mode)
    print("  reason:", recommendation.reason)
    print("  last_action:", robot.last_action)
    if output_dir is not None:
        print("  wrote_profile_json:", output_dir / "profile.json")
        print("  wrote_recommendation_json:", output_dir / "recommendation.json")
        print("  wrote_buffer_trace_json:", output_dir / "buffer_trace.json")


if __name__ == "__main__":
    main()
