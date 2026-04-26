"""Example 7: lightweight runtime overhead benchmark.

Run with:

    UV_CACHE_DIR=/tmp/uv-cache uv run python examples/07_benchmark_runtime.py
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time

import inferaxis as infra
import numpy as np


WARMUP_STEPS = 20
MEASURED_STEPS = 200


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    samples_ms: list[float]

    def summary(self) -> str:
        ordered = sorted(self.samples_ms)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        return (
            f"{self.name}:\n"
            f"  mean: {statistics.fmean(ordered):.4f} ms\n"
            f"  p50:  {p50:.4f} ms\n"
            f"  p95:  {p95:.4f} ms\n"
            f"  max:  {max(ordered):.4f} ms"
        )


class BenchmarkRobot:
    def __init__(self) -> None:
        self.last_action: infra.Action | None = None
        self._frame = infra.Frame(
            images={"front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={"arm": np.zeros(6, dtype=np.float64)},
        )

    def get_obs(self) -> infra.Frame:
        return self._frame

    def send_action(self, action: infra.Action) -> infra.Action:
        self.last_action = action
        return action


class BenchmarkPolicy:
    def __init__(self, *, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.request_count = 0

    def infer(
        self,
        obs: infra.Frame,
        request: infra.ChunkRequest,
    ) -> list[infra.Action]:
        del obs
        self.request_count += 1
        base = float(request.request_step)
        return [
            infra.Action.single(
                target="arm",
                command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                value=np.full(6, base + offset, dtype=np.float64),
            )
            for offset in range(self.chunk_size)
        ]


def measure(
    name: str,
    *,
    runtime: infra.InferenceRuntime,
    policy: BenchmarkPolicy,
    robot: BenchmarkRobot,
) -> BenchmarkResult:
    samples_ms: list[float] = []
    total_steps = WARMUP_STEPS + MEASURED_STEPS
    try:
        for step_index in range(total_steps):
            start = time.perf_counter()
            infra.run_step(
                observe_fn=robot.get_obs,
                act_fn=robot.send_action,
                act_src_fn=policy.infer,
                runtime=runtime,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if step_index >= WARMUP_STEPS:
                samples_ms.append(elapsed_ms)
    finally:
        runtime.close()
    return BenchmarkResult(name=name, samples_ms=samples_ms)


def main() -> None:
    scenarios = [
        (
            "sync run_step",
            infra.InferenceRuntime(
                mode=infra.InferenceMode.SYNC,
                startup_validation_only=True,
            ),
            BenchmarkPolicy(chunk_size=1),
        ),
        (
            "async scheduler",
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                warmup_requests=0,
                profile_delay_requests=0,
                startup_validation_only=True,
            ),
            BenchmarkPolicy(chunk_size=4),
        ),
        (
            "async scheduler rtc",
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                control_hz=50.0,
                execution_steps=3,
                warmup_requests=0,
                profile_delay_requests=0,
                enable_rtc=True,
                startup_validation_only=True,
            ),
            BenchmarkPolicy(chunk_size=5),
        ),
    ]
    for name, runtime, policy in scenarios:
        print(
            measure(
                name, runtime=runtime, policy=policy, robot=BenchmarkRobot()
            ).summary()
        )


if __name__ == "__main__":
    main()
