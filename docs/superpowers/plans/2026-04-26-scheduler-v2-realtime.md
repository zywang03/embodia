# Scheduler v2 Realtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the async scheduler internals around explicit realtime components, add small compatibility-preserving realtime APIs, and add a benchmark entrypoint for measuring runtime overhead.

**Architecture:** Keep `ChunkScheduler`, `InferenceRuntime`, `run_step(...)`, and `policy.infer(frame, request)` as the user-facing surface. Internally introduce `ValidationMode`, `RawChunkBuffer`, `ExecutionCursor`, `LatencyTracker`, `RtcWindowBuilder`, and `RequestPipeline`, then wire them into `ChunkScheduler` behind existing behavior tests. Add benchmark and API coverage before changing scheduler behavior.

**Tech Stack:** Python 3.11+, dataclasses, unittest, pytest, uv, ruff, numpy.

---

## Current State and Safety Notes

- Design spec: `docs/superpowers/specs/2026-04-26-scheduler-v2-realtime-design.md`
- Current branch may already be ahead of `origin/main`; do not push unless asked.
- `.codex` is an unrelated untracked local file. Do not add or modify it.
- Public usage must remain mostly compatible:

```python
runtime = infra.InferenceRuntime(
    mode=infra.InferenceMode.ASYNC,
    startup_validation_only=True,
)
```

- Existing verification baseline:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall -q src tests examples
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests examples
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests examples
```

Expected before starting: all commands exit `0`; unittest/pytest run `146`
tests. Existing RTC delay warnings in scheduler tests are acceptable.

## Target File Structure

- Create: `examples/07_benchmark_runtime.py`
  - Deterministic local benchmark for sync, async, and async RTC runtime paths.

- Create: `src/inferaxis/runtime/inference/validation.py`
  - Owns `ValidationMode` and `resolve_validation_mode(...)`.

- Modify: `src/inferaxis/runtime/inference/engine.py`
  - Adds `validation`, `slow_rtc_bootstrap`, and `async_realtime(...)`.
  - Keeps `startup_validation_only` compatible.

- Modify: `src/inferaxis/runtime/inference/engine_config.py`
  - Resolves validation mode and validates `slow_rtc_bootstrap`.
  - Passes new settings to `ChunkScheduler`.

- Modify: `src/inferaxis/runtime/inference/scheduler/core.py`
  - Becomes the scheduler orchestrator around v2 components.

- Create: `src/inferaxis/runtime/inference/scheduler/buffers.py`
  - Owns `RawChunkBuffer` and `ExecutionCursor`.

- Create: `src/inferaxis/runtime/inference/scheduler/pipeline.py`
  - Owns `RequestPipeline`.

- Modify: `src/inferaxis/runtime/inference/scheduler/latency.py`
  - Introduces `LatencyTracker` while preserving wrapper methods on
    `ChunkScheduler`.

- Modify: `src/inferaxis/runtime/inference/scheduler/rtc.py`
  - Introduces `RtcWindowBuilder` while preserving wrapper methods on
    `ChunkScheduler`.

- Modify: `src/inferaxis/runtime/inference/scheduler/actions.py`
  - Keep normalization/blending/interpolation helpers. Remove execution-buffer
    responsibilities after `ExecutionCursor` owns them.

- Modify: `src/inferaxis/runtime/inference/scheduler/bootstrap.py`
  - Uses `LatencyTracker` and `RtcWindowBuilder`.
  - Replaces unconditional interactive slow RTC prompt with policy handling.

- Modify: `src/inferaxis/runtime/inference/scheduler/execution.py`
  - Emits actions through `ExecutionCursor` and delegates pending request
    storage to `RequestPipeline`.

- Create: `tests/test_scheduler_components.py`
  - Focused unit tests for buffer, cursor, latency, RTC, and pipeline units.

- Modify: `tests/test_runtime_engine.py`
  - API tests for validation strategy, realtime preset, and slow RTC bootstrap.

- Modify: `tests/test_scheduler.py`
  - Compatibility behavior tests updated only when internals move from
    `_buffer`/`_execution_buffer` to explicit components.

## Task 1: Add Realtime Benchmark Example

**Files:**
- Create: `examples/07_benchmark_runtime.py`

- [ ] **Step 1: Create the benchmark script**

Create `examples/07_benchmark_runtime.py` with deterministic robot/policy
objects and compact statistics helpers:

```python
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
        print(measure(name, runtime=runtime, policy=policy, robot=BenchmarkRobot()).summary())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark once**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/07_benchmark_runtime.py
```

Expected: command exits `0` and prints sections for `sync run_step`,
`async scheduler`, and `async scheduler rtc`.

- [ ] **Step 3: Run existing tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: `Ran 146 tests` and `OK`.

- [ ] **Step 4: Commit**

Run:

```bash
git add examples/07_benchmark_runtime.py
git commit -m "example: add runtime benchmark"
```

Expected: commit succeeds with only the benchmark file.

## Task 2: Add Validation API Tests

**Files:**
- Modify: `tests/test_runtime_engine.py`

- [ ] **Step 1: Add failing runtime validation strategy tests**

Append these methods to `RuntimeEngineTests` in `tests/test_runtime_engine.py`:

```python
    def test_runtime_accepts_validation_strategy(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            validation="off",
        )

        self.assertEqual(runtime.validation, "off")
        self.assertFalse(runtime.startup_validation_only)

    def test_runtime_maps_startup_validation_only_to_validation_strategy(self) -> None:
        startup_runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            startup_validation_only=True,
        )
        always_runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            startup_validation_only=False,
        )

        self.assertEqual(startup_runtime.validation, "startup")
        self.assertEqual(always_runtime.validation, "always")

    def test_runtime_rejects_conflicting_validation_settings(self) -> None:
        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                startup_validation_only=True,
                validation="always",
            )

        self.assertIn("validation", str(ctx.exception))
        self.assertIn("startup_validation_only", str(ctx.exception))

    def test_runtime_rejects_unknown_validation_strategy(self) -> None:
        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                validation="sometimes",
            )

        self.assertIn("validation", str(ctx.exception))

    def test_async_realtime_preset_builds_async_runtime(self) -> None:
        runtime = infra.InferenceRuntime.async_realtime(
            control_hz=50.0,
            execution_steps=3,
            enable_rtc=True,
        )

        self.assertEqual(runtime.mode, infra.InferenceMode.ASYNC)
        self.assertEqual(runtime.validation, "startup")
        self.assertEqual(runtime.control_hz, 50.0)
        self.assertEqual(runtime.execution_steps, 3)
        self.assertTrue(runtime.enable_rtc)
        self.assertFalse(runtime.profile)
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_runtime_engine.RuntimeEngineTests -q
```

Expected: `FAILED` with `TypeError` or `AttributeError` mentioning
`validation` or `async_realtime`.

- [ ] **Step 3: Leave tests uncommitted for Task 3**

Run:

```bash
git status --short tests/test_runtime_engine.py
```

Expected: `M tests/test_runtime_engine.py`.

## Task 3: Implement Validation Strategy and Realtime Preset

**Files:**
- Create: `src/inferaxis/runtime/inference/validation.py`
- Modify: `src/inferaxis/runtime/inference/engine.py`
- Modify: `src/inferaxis/runtime/inference/engine_config.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/core.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/config.py`
- Modify: `tests/test_runtime_engine.py`

- [ ] **Step 1: Create validation strategy helper**

Create `src/inferaxis/runtime/inference/validation.py`:

```python
"""Validation strategy helpers for realtime inference runtime paths."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from ...core.errors import InterfaceValidationError


UNSET_VALIDATION: Final = object()


class ValidationMode(StrEnum):
    """Runtime validation strategy names."""

    ALWAYS = "always"
    STARTUP = "startup"
    OFF = "off"


def resolve_validation_mode(
    *,
    validation: ValidationMode | str | None,
    startup_validation_only: object,
    field_name: str,
) -> tuple[ValidationMode, bool]:
    """Resolve new validation strategy and legacy startup-only flag."""

    resolved_from_legacy: ValidationMode | None = None
    legacy_was_provided = startup_validation_only is not UNSET_VALIDATION
    if legacy_was_provided:
        if not isinstance(startup_validation_only, bool):
            raise InterfaceValidationError(
                f"{field_name}.startup_validation_only must be a bool."
            )
        resolved_from_legacy = (
            ValidationMode.STARTUP
            if startup_validation_only
            else ValidationMode.ALWAYS
        )

    if validation is None:
        mode = resolved_from_legacy or ValidationMode.STARTUP
    else:
        try:
            mode = ValidationMode(str(validation))
        except ValueError as exc:
            raise InterfaceValidationError(
                f"{field_name}.validation must be 'always', 'startup', or 'off', "
                f"got {validation!r}."
            ) from exc
        if legacy_was_provided and mode is not resolved_from_legacy:
            raise InterfaceValidationError(
                f"{field_name}.validation conflicts with "
                f"{field_name}.startup_validation_only."
            )

    startup_only = mode is ValidationMode.STARTUP
    return mode, startup_only


__all__ = ["UNSET_VALIDATION", "ValidationMode", "resolve_validation_mode"]
```

- [ ] **Step 2: Add fields to `InferenceRuntime`**

Modify `InferenceRuntime` in `src/inferaxis/runtime/inference/engine.py`:

```python
from .validation import UNSET_VALIDATION, ValidationMode
```

Add field:

```python
    validation: ValidationMode | str | None = None
    startup_validation_only: bool | object = UNSET_VALIDATION
```

Remove the old `startup_validation_only: bool = True` field line. The sentinel
lets the runtime distinguish an omitted legacy flag from an explicitly invalid
value such as `None`; `validate_runtime_config(...)` still normalizes the public
attribute back to a bool during `__post_init__`.

- [ ] **Step 3: Add `async_realtime` constructor**

Add this classmethod inside `InferenceRuntime`:

```python
    @classmethod
    def async_realtime(cls, **kwargs: object) -> "InferenceRuntime":
        """Build an async runtime with realtime-friendly defaults."""

        defaults: dict[str, object] = {
            "mode": InferenceMode.ASYNC,
            "validation": ValidationMode.STARTUP,
            "profile": False,
            "interpolation_steps": 0,
            "ensemble_weight": None,
        }
        defaults.update(kwargs)
        return cls(**defaults)  # type: ignore[arg-type]
```

- [ ] **Step 4: Resolve validation in runtime config**

In `src/inferaxis/runtime/inference/engine_config.py`, import:

```python
from .validation import resolve_validation_mode
```

In `validate_runtime_config(...)`, after mode validation and before validating
`profile`, add:

```python
    validation_mode, startup_validation_only = resolve_validation_mode(
        validation=runtime.validation,
        startup_validation_only=runtime.startup_validation_only,
        field_name="InferenceRuntime",
    )
    runtime.validation = validation_mode.value
    runtime.startup_validation_only = startup_validation_only
```

Remove the later old `startup_validation_only` bool type check because
`resolve_validation_mode(...)` now owns it.

- [ ] **Step 5: Pass validation to scheduler**

In `build_chunk_scheduler_kwargs(...)`, add:

```python
        "validation": runtime.validation,
```

In `sync_chunk_scheduler_config(...)`, add:

```python
    scheduler.validation = runtime.validation
```

In `InferenceRuntime._profile_config_snapshot(...)`, include the resolved
strategy so profile output records the trust boundary used for the run:

```python
            "validation": self.validation,
```

- [ ] **Step 6: Add fields to `ChunkScheduler` and validate them**

In `src/inferaxis/runtime/inference/scheduler/core.py`, add field:

```python
    validation: str | None = None
    startup_validation_only: bool | object = UNSET_VALIDATION
```

Remove the old `startup_validation_only: bool = True` field line and import
`UNSET_VALIDATION`.

In `src/inferaxis/runtime/inference/scheduler/config.py`, import
`resolve_validation_mode` and add to `_validate_configuration(...)`:

```python
    validation_mode, startup_validation_only = resolve_validation_mode(
        validation=self.validation,
        startup_validation_only=self.startup_validation_only,
        field_name="ChunkScheduler",
    )
    self.validation = validation_mode.value
    self.startup_validation_only = startup_validation_only
```

Remove the old `startup_validation_only` bool type check.

- [ ] **Step 7: Make runtime validation check honor `"off"`**

In `ChunkScheduler.runtime_validation_enabled(...)`, return `False` when
`self.validation == "off"`:

```python
        if self.validation == "off":
            return False
```

Keep existing startup behavior for `"startup"` and always behavior for
`"always"`.

In `InferenceRuntime._step_validation_enabled(...)`, add the same trust-boundary
check before legacy startup logic:

```python
        if self.validation == "off":
            return False
```

- [ ] **Step 8: Run API tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_runtime_engine.RuntimeEngineTests -q
```

Expected: all `RuntimeEngineTests` pass.

- [ ] **Step 9: Run full tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Expected: unittest passes and ruff reports `All checks passed!`.

Commit:

```bash
git add src/inferaxis/runtime/inference/validation.py src/inferaxis/runtime/inference/engine.py src/inferaxis/runtime/inference/engine_config.py src/inferaxis/runtime/inference/scheduler/core.py src/inferaxis/runtime/inference/scheduler/config.py tests/test_runtime_engine.py
git commit -m "feat: add realtime validation strategy"
```

## Task 4: Add RawChunkBuffer and ExecutionCursor

**Files:**
- Create: `src/inferaxis/runtime/inference/scheduler/buffers.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/actions.py`
- Create: `tests/test_scheduler_components.py`

- [ ] **Step 1: Add focused failing tests**

Create `tests/test_scheduler_components.py`:

```python
"""Focused tests for scheduler v2 internal components."""

from __future__ import annotations

import unittest

from inferaxis.runtime.inference.scheduler.buffers import (
    ExecutionCursor,
    RawChunkBuffer,
)

from helpers import arm_action, arm_value


class RawChunkBufferTests(unittest.TestCase):
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
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components -q
```

Expected: `ModuleNotFoundError` for `scheduler.buffers`.

- [ ] **Step 3: Extract standalone interpolation helper**

In `src/inferaxis/runtime/inference/scheduler/actions.py`, add a scheduler-free
helper and keep the current bound methods as compatibility wrappers. This lets
`ExecutionCursor` interpolate without pretending to be a full `ChunkScheduler`:

```python
def materialize_action(
    *,
    commands: dict[str, Command],
    meta: dict[str, Any],
) -> Action:
    action = object.__new__(Action)
    action.commands = commands
    action.meta = meta
    return action


def materialize_command(
    *,
    command: str,
    value: np.ndarray,
    ref_frame: str | None,
    meta: dict[str, Any],
) -> Command:
    materialized = object.__new__(Command)
    materialized.command = command
    materialized.value = value
    materialized.ref_frame = ref_frame
    materialized.meta = meta
    return materialized


def commands_share_target_layout(
    left_command: Command,
    right_command: Command,
) -> bool:
    if left_command.command != right_command.command:
        return False
    if left_command.ref_frame != right_command.ref_frame:
        return False
    if left_command.meta != right_command.meta:
        return False
    if left_command.value.shape != right_command.value.shape:
        return False
    return True


def interpolate_action(
    left_action: Action,
    right_action: Action,
    *,
    right_weight: float,
) -> Action:
    if right_weight <= 0.0:
        return left_action
    if right_weight >= 1.0:
        return right_action

    interpolated_commands: dict[str, Command] = {}
    has_interpolated_target = False
    for target, left_command in left_action.commands.items():
        right_command = right_action.commands.get(target)
        if (
            right_command is None
            or not commands_share_target_layout(left_command, right_command)
            or left_command.command in _NON_BLENDABLE_OVERLAP_COMMANDS
        ):
            interpolated_commands[target] = left_command
            continue

        has_interpolated_target = True
        interpolated_commands[target] = materialize_command(
            command=left_command.command,
            value=left_command.value * (1.0 - right_weight)
            + right_command.value * right_weight,
            ref_frame=left_command.ref_frame,
            meta=left_command.meta,
        )

    if not has_interpolated_target:
        return left_action

    return materialize_action(
        commands=interpolated_commands,
        meta=left_action.meta,
    )
```

Then update the existing bound wrappers to delegate:

```python
def _materialize_action(self, *, commands: dict[str, Command], meta: dict[str, Any]) -> Action:
    return materialize_action(commands=commands, meta=meta)


def _materialize_command(
    self,
    *,
    command: str,
    value: np.ndarray,
    ref_frame: str | None,
    meta: dict[str, Any],
) -> Command:
    return materialize_command(
        command=command,
        value=value,
        ref_frame=ref_frame,
        meta=meta,
    )


def _commands_share_target_layout(self, left_command: Command, right_command: Command) -> bool:
    return commands_share_target_layout(left_command, right_command)


def _interpolate_action(
    self,
    left_action: Action,
    right_action: Action,
    *,
    right_weight: float,
) -> Action:
    return interpolate_action(
        left_action,
        right_action,
        right_weight=right_weight,
    )
```

- [ ] **Step 4: Implement `buffers.py`**

Create `src/inferaxis/runtime/inference/scheduler/buffers.py`:

```python
"""Raw chunk and execution cursor helpers for realtime scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ....core.errors import InterfaceValidationError
from ....core.schema import Action
from .actions import interpolate_action


@dataclass(slots=True)
class RawChunkBuffer:
    """Track the active raw chunk without rebuilding containers per step."""

    actions: list[Action] = field(default_factory=list)
    start_index: int = 0
    global_step: int = 0
    active_chunk_consumed_steps: int = 0
    active_chunk_waited_raw_steps: int = 0
    active_source_plan_length: int = 0

    @property
    def has_actions(self) -> bool:
        return self.start_index < len(self.actions)

    @property
    def remaining_raw_count(self) -> int:
        return max(len(self.actions) - self.start_index, 0)

    def reset(self) -> None:
        self.actions = []
        self.start_index = 0
        self.global_step = 0
        self.active_chunk_consumed_steps = 0
        self.active_chunk_waited_raw_steps = 0
        self.active_source_plan_length = 0

    def current_action(self) -> Action:
        if not self.has_actions:
            raise InterfaceValidationError("RawChunkBuffer has no action to emit.")
        return self.actions[self.start_index]

    def next_action(self) -> Action | None:
        next_index = self.start_index + 1
        if next_index >= len(self.actions):
            return None
        return self.actions[next_index]

    def remaining_actions(self) -> Sequence[Action]:
        return self.actions[self.start_index :]

    def accept_chunk(
        self,
        *,
        actions: list[Action],
        request_step: int,
        current_raw_step: int,
        source_plan_length: int,
    ) -> int:
        stale_steps = max(current_raw_step - request_step, 0)
        if stale_steps >= len(actions):
            self.actions = actions
            self.start_index = len(actions)
            return stale_steps
        self.actions = actions
        self.start_index = stale_steps
        self.active_chunk_consumed_steps = stale_steps
        self.active_chunk_waited_raw_steps = 0
        self.active_source_plan_length = source_plan_length
        return stale_steps

    def advance_raw_step(self) -> None:
        if not self.has_actions:
            return
        self.start_index += 1
        self.global_step += 1
        self.active_chunk_consumed_steps = min(
            self.active_chunk_consumed_steps + 1,
            len(self.actions),
        )
        self.active_chunk_waited_raw_steps += 1


@dataclass(slots=True)
class ExecutionCursor:
    """Emit control-step actions from a raw chunk buffer."""

    buffer: RawChunkBuffer
    interpolation_steps: int = 0
    _segment_slot: int = 0

    def reset(self) -> None:
        self._segment_slot = 0

    @property
    def at_raw_boundary(self) -> bool:
        return self._segment_slot == 0

    @property
    def remaining_segment_steps(self) -> int:
        if not self.buffer.has_actions:
            return 0
        if self.interpolation_steps <= 0 or self.buffer.next_action() is None:
            return 1
        if self._segment_slot == 0:
            return 1 + self.interpolation_steps
        return max(self.interpolation_steps - self._segment_slot + 1, 0)

    def next_action(self) -> Action:
        left = self.buffer.current_action()
        right = self.buffer.next_action()
        if self.interpolation_steps <= 0 or right is None:
            self.buffer.advance_raw_step()
            self._segment_slot = 0
            return left

        if self._segment_slot == 0:
            self._segment_slot = 1
            return left

        if self._segment_slot <= self.interpolation_steps:
            right_weight = self._segment_slot / float(self.interpolation_steps + 1)
            self._segment_slot += 1
            action = interpolate_action(
                left,
                right,
                right_weight=right_weight,
            )
            if self._segment_slot > self.interpolation_steps:
                self.buffer.advance_raw_step()
                self._segment_slot = 0
            return action

        self.buffer.advance_raw_step()
        self._segment_slot = 0
        return self.next_action()


__all__ = ["ExecutionCursor", "RawChunkBuffer"]
```

- [ ] **Step 5: Run component tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components -q
```

Expected: tests pass.

- [ ] **Step 6: Run ruff and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Expected: `All checks passed!`.

Commit:

```bash
git add src/inferaxis/runtime/inference/scheduler/actions.py src/inferaxis/runtime/inference/scheduler/buffers.py tests/test_scheduler_components.py
git commit -m "feat: add scheduler raw buffer and cursor"
```

## Task 5: Add LatencyTracker

**Files:**
- Modify: `src/inferaxis/runtime/inference/scheduler/latency.py`
- Modify: `tests/test_scheduler_components.py`

- [ ] **Step 1: Add LatencyTracker tests**

Append to `tests/test_scheduler_components.py`:

```python
from inferaxis.runtime.inference.scheduler.latency import LatencyTracker


class LatencyTrackerTests(unittest.TestCase):
    def test_tracker_projects_control_latency_to_raw_steps(self) -> None:
        tracker = LatencyTracker(interpolation_steps=1)

        raw_delay = tracker.project_control_latency_to_raw_steps(
            control_latency_steps=3,
            raw_count=4,
            execution_buffer_steps=0,
        )

        self.assertEqual(raw_delay, 2)

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
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components.LatencyTrackerTests -q
```

Expected: `ImportError` for `LatencyTracker`.

- [ ] **Step 3: Implement `LatencyTracker` while keeping wrappers**

In `src/inferaxis/runtime/inference/scheduler/latency.py`, add a dataclass
above the existing wrapper functions:

```python
@dataclass(slots=True)
class LatencyTracker:
    latency_ema_beta: float = 0.5
    initial_latency_steps: float = 0.0
    fixed_latency_steps: float | None = None
    control_period_s: float | None = None
    warmup_requests: int = 3
    profile_delay_requests: int = 0
    interpolation_steps: int = 0
    latency_steps_offset: int = 0
    estimate: float = field(default=0.0, init=False)
    observation_count: int = field(default=0, init=False)
    bootstrap_complete: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.refresh_mode()

    def refresh_mode(self) -> None:
        if self.fixed_latency_steps is not None:
            self.estimate = self.fixed_latency_steps
            self.bootstrap_complete = True
            return
        self.estimate = self.initial_latency_steps
        self.bootstrap_complete = (
            self.control_period_s is None
            or (self.warmup_requests + self.profile_delay_requests) == 0
        )

    def estimated_latency_steps(self) -> int:
        return max(int(math.ceil(self.estimate)), 0)

    def latency_estimate_ready(self) -> bool:
        return self.bootstrap_complete

    def control_steps_for_raw_count(self, raw_steps: int) -> int:
        if raw_steps <= 0:
            return 0
        return raw_steps + max(raw_steps - 1, 0) * self.interpolation_steps

    def raw_segment_control_steps(self, *, has_successor: bool) -> int:
        if has_successor:
            return 1 + self.interpolation_steps
        return 1

    def project_control_latency_to_raw_steps(
        self,
        *,
        control_latency_steps: int,
        raw_count: int,
        execution_buffer_steps: int,
    ) -> int:
        if raw_count <= 0:
            return max(int(control_latency_steps), 0)
        remaining = max(int(control_latency_steps), 0)
        for raw_offset in range(raw_count):
            if raw_offset == 0 and execution_buffer_steps:
                segment_steps = max(int(execution_buffer_steps), 0)
            else:
                segment_steps = self.raw_segment_control_steps(
                    has_successor=raw_offset < (raw_count - 1),
                )
            if remaining <= segment_steps:
                return raw_offset + 1
            remaining -= segment_steps
        return raw_count

    def estimated_request_latency_steps(
        self,
        *,
        control_latency_steps: int,
        raw_count: int,
        execution_buffer_steps: int,
    ) -> int:
        projected = self.project_control_latency_to_raw_steps(
            control_latency_steps=control_latency_steps,
            raw_count=raw_count,
            execution_buffer_steps=execution_buffer_steps,
        )
        return max(projected + self.latency_steps_offset, 0)

    def update(self, *, waited_steps: int) -> None:
        if self.fixed_latency_steps is not None:
            self.estimate = self.fixed_latency_steps
            self.bootstrap_complete = True
            return
        self.observation_count += 1
        if self.observation_count <= self.warmup_requests:
            return
        if self.observation_count == self.warmup_requests + 1 and self.estimate <= 0:
            self.estimate = float(waited_steps)
            self.bootstrap_complete = True
            return
        self.estimate = (
            (1.0 - self.latency_ema_beta) * self.estimate
            + self.latency_ema_beta * float(waited_steps)
        )
        self.bootstrap_complete = True

    def observed_latency_steps_from_duration(self, inference_time_s: float) -> int:
        if self.control_period_s is None or inference_time_s <= 0.0:
            return 1
        return max(int(math.ceil(inference_time_s / self.control_period_s)), 1)
```

Add required imports:

```python
from dataclasses import dataclass, field
```

Keep existing module-level functions for `ChunkScheduler` compatibility. They
may delegate to `self._latency_tracker` in a later task; do not change
`ChunkScheduler` wiring in this task.

- [ ] **Step 4: Run component tests and existing scheduler tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components tests.test_scheduler -q
```

Expected: tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/inferaxis/runtime/inference/scheduler/latency.py tests/test_scheduler_components.py
git commit -m "feat: add scheduler latency tracker"
```

## Task 6: Add RtcWindowBuilder and Slow Bootstrap Policy Tests

**Files:**
- Modify: `src/inferaxis/runtime/inference/scheduler/rtc.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/bootstrap.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/core.py`
- Modify: `src/inferaxis/runtime/inference/engine.py`
- Modify: `src/inferaxis/runtime/inference/engine_config.py`
- Modify: `tests/test_scheduler_components.py`
- Modify: `tests/test_runtime_engine.py`

- [ ] **Step 1: Add RtcWindowBuilder tests**

Add `import inferaxis as infra` near the top of
`tests/test_scheduler_components.py`, then append:

```python
from inferaxis.runtime.inference.scheduler.rtc import RtcWindowBuilder


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
```

- [ ] **Step 2: Add slow RTC policy tests**

Append to `RuntimeEngineTests` in `tests/test_runtime_engine.py`:

```python
    def test_runtime_accepts_slow_rtc_bootstrap_policy(self) -> None:
        runtime = infra.InferenceRuntime(
            mode=infra.InferenceMode.ASYNC,
            execution_steps=2,
            enable_rtc=True,
            slow_rtc_bootstrap="error",
        )

        self.assertEqual(runtime.slow_rtc_bootstrap, "error")

    def test_runtime_rejects_invalid_slow_rtc_bootstrap_policy(self) -> None:
        with self.assertRaises(infra.InterfaceValidationError) as ctx:
            infra.InferenceRuntime(
                mode=infra.InferenceMode.ASYNC,
                execution_steps=2,
                enable_rtc=True,
                slow_rtc_bootstrap="ask-politely",
            )

        self.assertIn("slow_rtc_bootstrap", str(ctx.exception))
```

- [ ] **Step 3: Verify tests fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components.RtcWindowBuilderTests tests.test_runtime_engine.RuntimeEngineTests -q
```

Expected: `ImportError` for `RtcWindowBuilder` and/or `TypeError` for
`slow_rtc_bootstrap`.

- [ ] **Step 4: Implement RtcWindowBuilder**

In `src/inferaxis/runtime/inference/scheduler/rtc.py`, add:

```python
@dataclass(slots=True)
class RtcWindowBuilder:
    enabled: bool = False
    execution_steps: int | None = None
    steps_before_request: int = 0
    locked_chunk_total_length: int | None = None

    def reset(self) -> None:
        self.locked_chunk_total_length = None

    def lock_chunk_total_length(self, chunk_length: int) -> None:
        if not self.enabled:
            return
        if self.execution_steps is None:
            raise InterfaceValidationError(
                "RTC chunk length locking requires execution_steps."
            )
        if chunk_length <= 0:
            raise InterfaceValidationError(
                f"RTC chunk_total_length must be > 0, got {chunk_length!r}."
            )
        if self.locked_chunk_total_length is None:
            self.locked_chunk_total_length = chunk_length
            self.validate_execution_window_structure(chunk_length)
            return
        if chunk_length != self.locked_chunk_total_length:
            raise InterfaceValidationError(
                "RTC requires a stable source raw chunk length once the first "
                "chunk is accepted. Got "
                f"chunk_total_length={chunk_length!r}, "
                f"locked_chunk_total_length={self.locked_chunk_total_length!r}."
            )

    def validate_execution_window_structure(self, chunk_total_length: int) -> None:
        if not self.enabled or self.execution_steps is None:
            return
        if self.execution_steps >= (chunk_total_length - self.steps_before_request):
            raise InterfaceValidationError(
                "RTC requires execution_steps < chunk_total_length - "
                "steps_before_request, got "
                f"execution_steps={self.execution_steps!r}, "
                f"chunk_total_length={chunk_total_length!r}, "
                f"steps_before_request={self.steps_before_request!r}."
            )

    def build_prev_action_chunk(
        self,
        *,
        source_chunk: Sequence[Action],
    ) -> tuple[list[Action], int]:
        if not source_chunk:
            raise InterfaceValidationError(
                "RTC prev_action_chunk source must contain at least one action."
            )
        if self.execution_steps is None:
            raise InterfaceValidationError(
                "RTC prev_action_chunk construction requires execution_steps."
            )
        total_length = (
            self.locked_chunk_total_length
            if self.locked_chunk_total_length is not None
            else len(source_chunk)
        )
        execute_horizon = self.execution_steps
        if total_length < execute_horizon:
            raise InterfaceValidationError(
                "RTC locked chunk_total_length must be >= execution_steps, got "
                f"chunk_total_length={total_length!r}, "
                f"execution_steps={execute_horizon!r}."
            )
        window_limit = min(len(source_chunk), execute_horizon, total_length)
        window = [source_chunk[index] for index in range(window_limit)]
        total_pad_count = total_length - len(window)
        if total_pad_count > 0:
            pad_action = window[-1]
            window.extend(pad_action for _ in range(total_pad_count))
        return window, execute_horizon

    def build_args(
        self,
        *,
        remaining_chunk: Sequence[Action],
        inference_delay: int,
        rtc_seed_chunk: Sequence[Action] | None = None,
    ) -> RtcArgs | None:
        if not self.enabled:
            return None
        source_chunk = remaining_chunk if remaining_chunk else rtc_seed_chunk
        if not source_chunk:
            return None
        prev_action_chunk, execute_horizon = self.build_prev_action_chunk(
            source_chunk=source_chunk,
        )
        return RtcArgs(
            prev_action_chunk=prev_action_chunk,
            inference_delay=min(max(int(inference_delay), 1), execute_horizon),
            execute_horizon=execute_horizon,
        )
```

Add:

```python
from dataclasses import dataclass
```

Keep existing wrapper functions. They may delegate to
`self._rtc_window_builder` in a later task.

- [ ] **Step 5: Add slow policy field and validation**

In `InferenceRuntime`, add field:

```python
    slow_rtc_bootstrap: str = "warn"
```

In `ChunkScheduler`, add field:

```python
    slow_rtc_bootstrap: str = "warn"
```

In `engine_config.validate_runtime_config(...)` and
`scheduler/config._validate_configuration(...)`, validate:

```python
    if runtime.slow_rtc_bootstrap not in {"warn", "error", "confirm"}:
        raise InterfaceValidationError(
            "InferenceRuntime.slow_rtc_bootstrap must be 'warn', 'error', or "
            f"'confirm', got {runtime.slow_rtc_bootstrap!r}."
        )
```

Use `self` and `ChunkScheduler` in the scheduler-side message.

Pass/sync `slow_rtc_bootstrap` through `build_chunk_scheduler_kwargs(...)` and
`sync_chunk_scheduler_config(...)`.

- [ ] **Step 6: Implement policy behavior in bootstrap**

In `_confirm_slow_rtc_bootstrap_request(...)`, replace the current unconditional
`warnings.warn(...)` plus `input(...)` block with a single policy branch:

```python
    if self.slow_rtc_bootstrap == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    if self.slow_rtc_bootstrap == "error":
        raise InterfaceValidationError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
```

Keep `"confirm"` using the existing prompt path after that final warning. This
preserves the old interactive behavior without double-warning in `"warn"` mode.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components tests.test_runtime_engine -q
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: all tests pass.

Commit:

```bash
git add src/inferaxis/runtime/inference/scheduler/rtc.py src/inferaxis/runtime/inference/scheduler/bootstrap.py src/inferaxis/runtime/inference/scheduler/core.py src/inferaxis/runtime/inference/scheduler/config.py src/inferaxis/runtime/inference/engine.py src/inferaxis/runtime/inference/engine_config.py tests/test_scheduler_components.py tests/test_runtime_engine.py
git commit -m "feat: add scheduler rtc window builder"
```

## Task 7: Introduce RequestPipeline Skeleton

**Files:**
- Create: `src/inferaxis/runtime/inference/scheduler/pipeline.py`
- Modify: `tests/test_scheduler_components.py`

- [ ] **Step 1: Add RequestPipeline smoke tests**

Append to `tests/test_scheduler_components.py`:

```python
from concurrent.futures import Future

from inferaxis.runtime.inference.scheduler.pipeline import RequestPipeline


class RequestPipelineTests(unittest.TestCase):
    def test_pipeline_tracks_pending_future(self) -> None:
        pipeline = RequestPipeline()

        self.assertFalse(pipeline.has_pending)
        pipeline.pending = Future()
        self.assertTrue(pipeline.has_pending)
        pipeline.clear_pending()
        self.assertFalse(pipeline.has_pending)
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components.RequestPipelineTests -q
```

Expected: `ModuleNotFoundError` for `scheduler.pipeline`.

- [ ] **Step 3: Create pipeline skeleton**

Create `src/inferaxis/runtime/inference/scheduler/pipeline.py`:

```python
"""Request pipeline helpers for async chunk scheduling."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass

from .state import _CompletedChunk


@dataclass(slots=True)
class RequestPipeline:
    """Track pending request state for the scheduler orchestrator."""

    pending: Future[_CompletedChunk] | None = None

    @property
    def has_pending(self) -> bool:
        return self.pending is not None

    def clear_pending(self) -> None:
        self.pending = None


__all__ = ["RequestPipeline"]
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Expected: tests pass and ruff is clean.

Commit:

```bash
git add src/inferaxis/runtime/inference/scheduler/pipeline.py tests/test_scheduler_components.py
git commit -m "feat: add scheduler request pipeline"
```

## Task 8: Wire Buffer and Cursor into ChunkScheduler

**Files:**
- Modify: `src/inferaxis/runtime/inference/scheduler/core.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/actions.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/execution.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/requests.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/latency.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/bootstrap.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Replace raw-buffer storage fields**

In `src/inferaxis/runtime/inference/scheduler/core.py`, remove these dataclass
fields:

```python
    _buffer: deque[Action] = field(default_factory=deque, init=False, repr=False)
    _global_step: int = field(default=0, init=False, repr=False)
    _active_chunk_snapshot: list[Action] = field(default_factory=list, init=False, repr=False)
    _active_chunk_consumed_steps: int = field(default=0, init=False, repr=False)
    _active_chunk_waited_raw_steps: int = field(default=0, init=False, repr=False)
    _active_source_plan_length: int = field(default=0, init=False, repr=False)
    _execution_buffer: deque[Action] = field(default_factory=deque, init=False, repr=False)
```

Add component fields and import from `.buffers`:

```python
from .buffers import ExecutionCursor, RawChunkBuffer

    _raw_buffer: RawChunkBuffer = field(init=False, repr=False)
    _execution_cursor: ExecutionCursor = field(init=False, repr=False)
```

In `ChunkScheduler.__post_init__(...)`, after `_validate_configuration()`, add:

```python
        self._raw_buffer = RawChunkBuffer()
        self._execution_cursor = ExecutionCursor(
            buffer=self._raw_buffer,
            interpolation_steps=self.interpolation_steps,
        )
```

- [ ] **Step 2: Add cold-path compatibility accessors**

In `ChunkScheduler`, add these properties so existing tests and debugging code
keep working while hot-path code uses explicit components:

```python
    @property
    def _buffer(self) -> deque[Action]:
        return deque(self._raw_buffer.remaining_actions())

    @_buffer.setter
    def _buffer(self, value: deque[Action]) -> None:
        self._raw_buffer.accept_chunk(
            actions=list(value),
            request_step=self._raw_buffer.global_step,
            current_raw_step=self._raw_buffer.global_step,
            source_plan_length=len(value),
        )
        self._execution_cursor.reset()

    @property
    def _execution_buffer(self) -> deque[Action]:
        if not self._raw_buffer.has_actions:
            return deque()
        return deque(
            self._raw_buffer.current_action()
            for _ in range(self._execution_cursor.remaining_segment_steps)
        )

    @_execution_buffer.setter
    def _execution_buffer(self, value: deque[Action]) -> None:
        if value:
            self._buffer = value
        else:
            self._execution_cursor.reset()

    @property
    def _global_step(self) -> int:
        return self._raw_buffer.global_step

    @_global_step.setter
    def _global_step(self, value: int) -> None:
        self._raw_buffer.global_step = value

    @property
    def _active_chunk_consumed_steps(self) -> int:
        return self._raw_buffer.active_chunk_consumed_steps

    @_active_chunk_consumed_steps.setter
    def _active_chunk_consumed_steps(self, value: int) -> None:
        self._raw_buffer.active_chunk_consumed_steps = value

    @property
    def _active_chunk_waited_raw_steps(self) -> int:
        return self._raw_buffer.active_chunk_waited_raw_steps

    @_active_chunk_waited_raw_steps.setter
    def _active_chunk_waited_raw_steps(self, value: int) -> None:
        self._raw_buffer.active_chunk_waited_raw_steps = value

    @property
    def _active_source_plan_length(self) -> int:
        return self._raw_buffer.active_source_plan_length

    @_active_source_plan_length.setter
    def _active_source_plan_length(self, value: int) -> None:
        self._raw_buffer.active_source_plan_length = value
```

Keep `collections.deque` and `Action` imported in `core.py` because these
compatibility properties use both names.

- [ ] **Step 3: Update reset and active_source_plan_length**

In `reset(...)`, replace direct `_buffer` and `_execution_buffer` clearing with:

```python
        self._raw_buffer.reset()
        self._execution_cursor.reset()
```

Remove assignments to `_global_step`, `_active_chunk_snapshot`,
`_active_chunk_consumed_steps`, `_active_chunk_waited_raw_steps`, and
`_active_source_plan_length` from `reset(...)`; `RawChunkBuffer.reset()` owns
that state now.

Return active source plan length from:

```python
        return self._raw_buffer.active_source_plan_length
```

- [ ] **Step 4: Update integration**

In `_integrate_completed_chunk(...)`, after profiler handling, replace
`next_buffer = deque(...)`, `_active_chunk_snapshot`, `_active_chunk_*`, and
`_active_source_plan_length` assignments with:

```python
    self._raw_buffer.accept_chunk(
        actions=completed.prepared_actions,
        request_step=completed.request.request_step,
        current_raw_step=integration_step,
        source_plan_length=completed.source_plan_length,
    )
    self._execution_cursor.reset()
```

Keep profiler stale/accepted length calculations unchanged.

- [ ] **Step 5: Update executable-action flow**

In `src/inferaxis/runtime/inference/scheduler/execution.py`, replace checks
against `_execution_buffer` with cursor/buffer checks:

```python
def _accept_ready_pending_chunk(self) -> bool:
    """Try to accept one already-finished async request."""

    if not self._execution_cursor.at_raw_boundary:
        return False
    return self._accept_pending_chunk(block=False)


def _accept_blocking_pending_chunk(self) -> bool:
    """Block until the in-flight request finishes and integrate it."""

    if self._pending_future is None:
        return False
    return self._accept_pending_chunk(block=True)
```

Update `_request_until_execution_buffer_ready(...)` to loop on raw availability:

```python
    while not self._raw_buffer.has_actions:
```

After `_integrate_completed_chunk(completed)`, remove the call to
`self._ensure_execution_buffer()`.

Update `_ensure_executable_actions(...)`:

```python
    if self._raw_buffer.has_actions:
        return False

    plan_refreshed = False
    if prefetch_async and self._pending_future is not None:
        plan_refreshed = self._accept_blocking_pending_chunk()
    if self._raw_buffer.has_actions:
        return plan_refreshed

    return (
        self._request_until_execution_buffer_ready(
            frame,
            include_latency=prefetch_async,
        )
        or plan_refreshed
    )
```

Update `_maybe_launch_next_request(...)`:

```python
    if not self._raw_buffer.has_actions or self._active_source_plan_length == 1:
        return plan_refreshed
```

- [ ] **Step 6: Update action emission**

In `_pop_next_action(...)`, replace `_execution_buffer` popleft logic with:

```python
    if not self._raw_buffer.has_actions:
        raise InterfaceValidationError("ChunkScheduler has no buffered action to emit.")
    action = self._execution_cursor.next_action()
    self._control_step += 1
    return action
```

- [ ] **Step 7: Update request, latency, and bootstrap helpers**

In `src/inferaxis/runtime/inference/scheduler/requests.py`, build request jobs
from the raw buffer:

```python
    buffer_actions = self._raw_buffer.remaining_actions()
    buffer_length = self._raw_buffer.remaining_raw_count
```

Pass cursor segment length into latency projection:

```python
            execution_buffer_steps=self._execution_cursor.remaining_segment_steps,
```

In `src/inferaxis/runtime/inference/scheduler/latency.py`, update
`_remaining_control_steps(...)`:

```python
    current_segment_steps = self._execution_cursor.remaining_segment_steps
    if current_segment_steps:
        remaining_raw_after_segment = max(
            self._raw_buffer.remaining_raw_count - 1,
            0,
        )
        return current_segment_steps + self._control_steps_for_raw_count(
            remaining_raw_after_segment,
        )
    return self._control_steps_for_raw_count(self._raw_buffer.remaining_raw_count)
```

In `_project_control_latency_to_raw_steps(...)`, use the raw-buffer window when
`buffer_actions` is omitted:

```python
    active_buffer = (
        self._raw_buffer.remaining_actions()
        if buffer_actions is None
        else buffer_actions
    )
```

In `src/inferaxis/runtime/inference/scheduler/bootstrap.py`, replace startup
checks that read `_buffer` or `_execution_buffer`:

```python
    if self._raw_buffer.has_actions:
        return False
```

and:

```python
    if self._raw_buffer.has_actions or self._pending_future is not None:
        return False
```

In `src/inferaxis/runtime/inference/scheduler/actions.py`, make the old
execution-buffer helpers compatibility-only wrappers:

```python
def _ensure_execution_buffer(self) -> None:
    return None


def _advance_raw_step(self) -> None:
    self._raw_buffer.advance_raw_step()
    self._execution_cursor.reset()
```

- [ ] **Step 8: Run scheduler tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler -q
```

Expected: scheduler tests pass.

- [ ] **Step 9: Run full tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected: all tests pass.

Commit:

```bash
git add src/inferaxis/runtime/inference/scheduler/core.py src/inferaxis/runtime/inference/scheduler/actions.py src/inferaxis/runtime/inference/scheduler/execution.py src/inferaxis/runtime/inference/scheduler/requests.py src/inferaxis/runtime/inference/scheduler/latency.py src/inferaxis/runtime/inference/scheduler/bootstrap.py tests/test_scheduler.py
git commit -m "refactor: wire scheduler raw buffer cursor"
```

## Task 9: Wire LatencyTracker and RtcWindowBuilder into ChunkScheduler

**Files:**
- Modify: `src/inferaxis/runtime/inference/scheduler/core.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/config.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/latency.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/rtc.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/requests.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/bootstrap.py`

- [ ] **Step 1: Initialize tracker and RTC builder**

In `src/inferaxis/runtime/inference/scheduler/core.py`, remove these dataclass
fields because the new components own the state:

```python
    _latency_steps_estimate: float = field(default=0.0, init=False, repr=False)
    _latency_observation_count: int = field(default=0, init=False, repr=False)
    _startup_latency_bootstrap_complete: bool = field(default=False, init=False, repr=False)
    _rtc_chunk_total_length: int | None = field(default=None, init=False, repr=False)
```

Import `LatencyTracker` and `RtcWindowBuilder`, add component fields with
`init=False`, then initialize them in `ChunkScheduler.__post_init__(...)` after
`_validate_configuration()` and after the raw buffer/cursor:

```python
        self._latency_tracker = LatencyTracker(
            latency_ema_beta=self.latency_ema_beta,
            initial_latency_steps=self.initial_latency_steps,
            fixed_latency_steps=self.fixed_latency_steps,
            control_period_s=self.control_period_s,
            warmup_requests=self.warmup_requests,
            profile_delay_requests=self.profile_delay_requests,
            interpolation_steps=self.interpolation_steps,
            latency_steps_offset=self.latency_steps_offset,
        )
        self._rtc_window_builder = RtcWindowBuilder(
            enabled=self.enable_rtc,
            execution_steps=self.execution_steps,
            steps_before_request=self.steps_before_request,
        )
```

Add fields:

```python
    _latency_tracker: LatencyTracker = field(init=False, repr=False)
    _rtc_window_builder: RtcWindowBuilder = field(init=False, repr=False)
```

- [ ] **Step 2: Add latency and RTC compatibility accessors**

Add these cold-path properties to `ChunkScheduler` for existing tests and
debugging code:

```python
    @property
    def _latency_steps_estimate(self) -> float:
        return self._latency_tracker.estimate

    @_latency_steps_estimate.setter
    def _latency_steps_estimate(self, value: float) -> None:
        self._latency_tracker.estimate = float(value)

    @property
    def _latency_observation_count(self) -> int:
        return self._latency_tracker.observation_count

    @_latency_observation_count.setter
    def _latency_observation_count(self, value: int) -> None:
        self._latency_tracker.observation_count = int(value)

    @property
    def _startup_latency_bootstrap_complete(self) -> bool:
        return self._latency_tracker.bootstrap_complete

    @_startup_latency_bootstrap_complete.setter
    def _startup_latency_bootstrap_complete(self, value: bool) -> None:
        self._latency_tracker.bootstrap_complete = bool(value)

    @property
    def _rtc_chunk_total_length(self) -> int | None:
        return self._rtc_window_builder.locked_chunk_total_length

    @_rtc_chunk_total_length.setter
    def _rtc_chunk_total_length(self, value: int | None) -> None:
        self._rtc_window_builder.locked_chunk_total_length = value
```

- [ ] **Step 3: Sync component settings after config changes**

In `refresh_latency_mode(...)`, remove the old direct assignments to
`_latency_steps_estimate` and `_startup_latency_bootstrap_complete`. Then update
the tracker from scheduler fields when it exists:

```python
    if not hasattr(self, "_latency_tracker"):
        return
    self._latency_tracker.latency_ema_beta = self.latency_ema_beta
    self._latency_tracker.initial_latency_steps = self.initial_latency_steps
    self._latency_tracker.fixed_latency_steps = self.fixed_latency_steps
    self._latency_tracker.control_period_s = self.control_period_s
    self._latency_tracker.warmup_requests = self.warmup_requests
    self._latency_tracker.profile_delay_requests = self.profile_delay_requests
    self._latency_tracker.interpolation_steps = self.interpolation_steps
    self._latency_tracker.latency_steps_offset = self.latency_steps_offset
    self._latency_tracker.refresh_mode()
```

The first `_validate_configuration()` call in `__post_init__` validates public
settings; the tracker itself is constructed immediately after that with the
validated values.

In `_validate_configuration(...)`, update RTC builder settings when it already
exists:

```python
    if hasattr(self, "_rtc_window_builder"):
        self._rtc_window_builder.enabled = self.enable_rtc
        self._rtc_window_builder.execution_steps = self.execution_steps
        self._rtc_window_builder.steps_before_request = self.steps_before_request
```

- [ ] **Step 4: Delegate latency wrappers**

Update existing functions in `latency.py` to delegate to
`self._latency_tracker`:

```python
def estimated_latency_steps(self) -> int:
    return self._latency_tracker.estimated_latency_steps()
```

Apply the same delegation for `_control_steps_for_raw_count`,
`_estimated_request_latency_steps`, `_update_latency_estimate`, and
`_observed_latency_steps_from_duration`.

- [ ] **Step 5: Delegate RTC wrappers**

Update existing functions in `rtc.py` to delegate to `self._rtc_window_builder`:

```python
def _build_rtc_args(self, *, remaining_chunk, inference_delay, rtc_seed_chunk=None):
    return self._rtc_window_builder.build_args(
        remaining_chunk=remaining_chunk,
        inference_delay=inference_delay,
        rtc_seed_chunk=rtc_seed_chunk,
    )
```

Delegate `_lock_rtc_chunk_total_length`,
`_validate_rtc_execution_window_structure`, and `_build_prev_action_chunk`.

- [ ] **Step 6: Run RTC and latency tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler_components tests.test_scheduler -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/inferaxis/runtime/inference/scheduler/core.py src/inferaxis/runtime/inference/scheduler/config.py src/inferaxis/runtime/inference/scheduler/latency.py src/inferaxis/runtime/inference/scheduler/rtc.py src/inferaxis/runtime/inference/scheduler/requests.py src/inferaxis/runtime/inference/scheduler/bootstrap.py
git commit -m "refactor: wire scheduler latency and rtc components"
```

## Task 10: Move Request Lifecycle into RequestPipeline

**Files:**
- Modify: `src/inferaxis/runtime/inference/scheduler/pipeline.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/core.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/requests.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/execution.py`

- [ ] **Step 1: Move pending state into pipeline**

Extend `RequestPipeline` with:

```python
from concurrent.futures import Future, ThreadPoolExecutor

from .state import _CompletedChunk

    pending: Future[_CompletedChunk] | None = None
    executor: ThreadPoolExecutor | None = None

    @property
    def has_ready_pending(self) -> bool:
        return self.pending is not None and self.pending.done()

    def close(self) -> None:
        if self.pending is not None:
            self.pending.cancel()
            self.pending = None
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def ensure_executor(self) -> ThreadPoolExecutor:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="inferaxis-async-inference",
            )
        return self.executor
```

- [ ] **Step 2: Use pipeline in `ChunkScheduler`**

In `ChunkScheduler`, remove `_pending_future` and `_executor` fields, add:

```python
    _pipeline: RequestPipeline = field(default_factory=RequestPipeline, init=False, repr=False)
```

In `close(...)`, call:

```python
        self._record_completed_pending_profile_request()
        self.reset()
        self._pipeline.close()
```

In `reset(...)`, cancel and clear pending through pipeline but keep executor
alive:

```python
        if self._pipeline.pending is not None:
            self._pipeline.pending.cancel()
        self._pipeline.clear_pending()
```

- [ ] **Step 3: Add pending/executor compatibility accessors**

Add these properties to `ChunkScheduler` so existing private tests can keep
inspecting scheduler state without owning the storage:

```python
    @property
    def _pending_future(self) -> Future[_CompletedChunk] | None:
        return self._pipeline.pending

    @_pending_future.setter
    def _pending_future(self, value: Future[_CompletedChunk] | None) -> None:
        self._pipeline.pending = value

    @property
    def _executor(self) -> ThreadPoolExecutor | None:
        return self._pipeline.executor

    @_executor.setter
    def _executor(self, value: ThreadPoolExecutor | None) -> None:
        self._pipeline.executor = value
```

Keep `Future`, `ThreadPoolExecutor`, and `_CompletedChunk` imported in
`core.py` for the property annotations.

- [ ] **Step 4: Update request/execution helpers**

In `requests.py` and `execution.py`, replace `self._pending_future` with
`self._pipeline.pending` and `_ensure_executor()` with
`self._pipeline.ensure_executor()`.

Keep method names on `ChunkScheduler` for private-test compatibility:

```python
def _ensure_executor(self):
    return self._pipeline.ensure_executor()
```

- [ ] **Step 5: Run async scheduler tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_scheduler.SchedulerTests -q
```

Expected: scheduler tests pass.

- [ ] **Step 6: Run full tests and commit**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected: all tests pass.

Commit:

```bash
git add src/inferaxis/runtime/inference/scheduler/pipeline.py src/inferaxis/runtime/inference/scheduler/core.py src/inferaxis/runtime/inference/scheduler/requests.py src/inferaxis/runtime/inference/scheduler/execution.py
git commit -m "refactor: move scheduler requests into pipeline"
```

## Task 11: Polish Examples for Human Realtime Usage

**Files:**
- Modify: `examples/02_async_inference.py`
- Modify: `examples/06_async_inference_with_rtc.py`
- Modify: `examples/07_benchmark_runtime.py`

- [ ] **Step 1: Update async example to use preset**

In `examples/02_async_inference.py`, replace the verbose runtime constructor
with:

```python
    runtime = infra.InferenceRuntime.async_realtime(
        control_hz=50.0,
        warmup_requests=0,
        profile_delay_requests=0,
    )
```

Keep printed output fields unchanged.

- [ ] **Step 2: Update RTC example to use preset**

In `examples/06_async_inference_with_rtc.py`, replace the runtime constructor
with:

```python
    runtime = infra.InferenceRuntime.async_realtime(
        control_hz=50.0,
        execution_steps=3,
        enable_rtc=True,
        slow_rtc_bootstrap="warn",
    )
```

- [ ] **Step 3: Run examples**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/02_async_inference.py
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/06_async_inference_with_rtc.py
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/07_benchmark_runtime.py
```

Expected: each command exits `0`.

- [ ] **Step 4: Commit**

Run:

```bash
git add examples/02_async_inference.py examples/06_async_inference_with_rtc.py examples/07_benchmark_runtime.py
git commit -m "docs: polish realtime runtime examples"
```

## Task 12: Final Verification and Performance Comparison

**Files:**
- Modify only files touched by earlier tasks if verification exposes a
  regression.

- [ ] **Step 1: Run compile verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall -q src tests examples
```

Expected: exits `0`.

- [ ] **Step 2: Run unittest**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: all tests pass.

- [ ] **Step 3: Run pytest**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Run ruff**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests examples
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests examples
```

Expected: both commands exit `0`.

- [ ] **Step 5: Run benchmark**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/07_benchmark_runtime.py
```

Expected: exits `0` and prints three benchmark sections.

- [ ] **Step 6: Commit verification fixes if needed**

If any previous step required fixes:

```bash
git add src tests examples
git commit -m "fix: resolve scheduler v2 verification issues"
```

If no fixes were needed, skip this commit.

## Task 13: Final Review Checklist

**Files:**
- No planned file changes.

- [ ] **Step 1: Inspect hot-path module sizes**

Run:

```bash
wc -l src/inferaxis/runtime/inference/scheduler/core.py src/inferaxis/runtime/inference/scheduler/buffers.py src/inferaxis/runtime/inference/scheduler/pipeline.py src/inferaxis/runtime/inference/scheduler/latency.py src/inferaxis/runtime/inference/scheduler/rtc.py src/inferaxis/runtime/inference/scheduler/actions.py src/inferaxis/runtime/inference/scheduler/execution.py src/inferaxis/runtime/inference/scheduler/requests.py
```

Expected: `core.py` is smaller than before or clearly orchestrational, and new
component modules have focused responsibilities.

- [ ] **Step 2: Inspect public API diff**

Run:

```bash
git diff --stat HEAD~12..HEAD
git diff HEAD~12..HEAD -- src/inferaxis/__init__.py src/inferaxis/runtime/inference/__init__.py src/inferaxis/runtime/inference/contracts.py
```

Expected: public API changes are limited to documented additions. `ChunkRequest`
field semantics are unchanged.

- [ ] **Step 3: Confirm benchmark exists and is not a CI gate**

Run:

```bash
sed -n '1,240p' examples/07_benchmark_runtime.py
```

Expected: benchmark prints local timing summaries and does not assert absolute
performance thresholds.

- [ ] **Step 4: Report final status**

Summarize:

```text
Implemented Scheduler v2 realtime optimization.
Public runtime usage preserved with small additions:
- validation
- InferenceRuntime.async_realtime(...)
- slow_rtc_bootstrap

Verification passed:
- compileall
- unittest
- pytest
- ruff check
- ruff format --check
- benchmark example
```

Do not claim completion unless every command in Task 12 passed in the current
session.
