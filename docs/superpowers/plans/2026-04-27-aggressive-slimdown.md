# Aggressive Slimdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce inferaxis to the approved minimal root API and remove legacy compatibility code without changing runtime scheduling behavior.

**Architecture:** Keep current runtime, scheduler, schema, and profiling module boundaries where they still serve current behavior. Remove root-level legacy exports first, then remove test-only compatibility helpers and scheduler private proxy accessors. Use tests to lock the smaller contract before deleting implementation code.

**Tech Stack:** Python 3.12, dataclasses, numpy, pytest, ruff.

---

## File Structure

- Modify `src/inferaxis/__init__.py`: root package exports only the approved minimal API.
- Modify examples `examples/01_sync_inference.py`, `examples/02_async_inference.py`, `examples/03_data_collection.py`, `examples/04_replay_collected_data.py`, `examples/06_async_inference_with_rtc.py`: import transform helpers from `inferaxis.core.transform` instead of `inferaxis`.
- Modify `tests/test_interfaces.py`: import non-root helpers from submodules and assert the minimal root API.
- Modify `tests/helpers.py`: stop silently ignoring removed `ChunkRequest` fields; add small scheduler test helper functions.
- Modify `tests/test_internal_architecture.py`: remove tests that protect legacy module export completeness.
- Modify `tests/test_scheduler.py`, `tests/test_runtime_engine.py`, `tests/test_profiling.py`: replace old scheduler private proxy usage with real scheduler components.
- Modify `src/inferaxis/runtime/inference/scheduler/core.py`: remove `_buffer`, `_execution_buffer`, `_pending_future`, and `_executor`.
- Modify `src/inferaxis/runtime/inference/scheduler/actions.py`, `src/inferaxis/runtime/inference/scheduler/bootstrap.py`: use real scheduler components after proxy removal.
- Modify README/docs only where they still describe removed root exports or compatibility behavior.

---

### Task 1: Lock the Minimal Root API

**Files:**
- Modify: `tests/test_interfaces.py`

- [ ] **Step 1: Write the failing root API test and move existing helper imports**

At the top of `tests/test_interfaces.py`, add explicit submodule imports for helpers that will no longer be root exports:

```python
from inferaxis.core.errors import InterfaceValidationError
from inferaxis.core.schema import (
    Command,
    CommandKindSpec,
    PolicyOutputSpec,
    PolicySpec,
    RobotSpec,
    get_command_kind_spec,
    is_known_command_kind,
    register_command_kind,
    validate_command,
)
from inferaxis.core.transform import (
    action_to_dict,
    coerce_action,
    coerce_frame,
    frame_to_dict,
)
from inferaxis.runtime.checks import check_pair, check_policy
```

Then add this test near the top of `InterfaceTests`:

```python
def test_root_public_api_is_minimal(self) -> None:
    expected_names = {
        "Action",
        "BuiltinCommandKind",
        "ChunkRequest",
        "Command",
        "Frame",
        "InferenceMode",
        "InferenceRuntime",
        "RealtimeController",
        "run_step",
    }

    self.assertEqual(set(infra.__all__), expected_names)
    for removed_name in (
        "InterfaceValidationError",
        "PolicySpec",
        "RobotSpec",
        "action_to_dict",
        "check_pair",
        "coerce_action",
        "coerce_frame",
        "frame_to_dict",
    ):
        self.assertFalse(hasattr(infra, removed_name), removed_name)
```

Update existing `tests/test_interfaces.py` references in the same edit:

```python
infra.coerce_action(...)      -> coerce_action(...)
infra.coerce_frame(...)       -> coerce_frame(...)
infra.action_to_dict(...)     -> action_to_dict(...)
infra.frame_to_dict(...)      -> frame_to_dict(...)
infra.check_pair(...)         -> check_pair(...)
infra.InterfaceValidationError -> InterfaceValidationError
infra.PolicySpec              -> PolicySpec
infra.RobotSpec               -> RobotSpec
```

- [ ] **Step 2: Run the targeted red test**

Run:

```bash
uv run pytest tests/test_interfaces.py::InterfaceTests::test_root_public_api_is_minimal -q
```

Expected: FAIL because `inferaxis.__all__` still contains non-minimal root exports such as `InterfaceValidationError`, `PolicySpec`, `RobotSpec`, `action_to_dict`, `coerce_action`, `coerce_frame`, `frame_to_dict`, and `check_pair`.

- [ ] **Step 3: Keep the red test uncommitted**

Do not commit this red state. Task 2 turns the test green and commits the root
API cleanup together.

---

### Task 2: Remove Non-Minimal Root Exports and Fix Example Imports

**Files:**
- Modify: `src/inferaxis/__init__.py`
- Modify: `examples/01_sync_inference.py`
- Modify: `examples/02_async_inference.py`
- Modify: `examples/03_data_collection.py`
- Modify: `examples/04_replay_collected_data.py`
- Modify: `examples/06_async_inference_with_rtc.py`
- Modify: `tests/test_interfaces.py`
- Modify: `tests/test_profiling.py`
- Modify: `tests/test_runtime_engine.py`
- Modify: `tests/test_scheduler.py`
- Modify: `tests/test_scheduler_components.py`

- [ ] **Step 1: Implement minimal root exports**

Replace `src/inferaxis/__init__.py` imports and `__all__` with:

```python
"""Public package exports for inferaxis."""

from .core.schema import (
    Action,
    BuiltinCommandKind,
    Command,
    Frame,
)
from .runtime.flow import run_step
from .runtime.inference import (
    ChunkRequest,
    InferenceMode,
    InferenceRuntime,
    RealtimeController,
)

__all__ = [
    "Action",
    "BuiltinCommandKind",
    "ChunkRequest",
    "Command",
    "Frame",
    "InferenceMode",
    "InferenceRuntime",
    "RealtimeController",
    "run_step",
]
```

- [ ] **Step 2: Move transform helper imports in examples**

In `examples/01_sync_inference.py`, `examples/02_async_inference.py`, and `examples/06_async_inference_with_rtc.py`, add:

```python
from inferaxis.core.transform import action_to_dict
```

Then replace:

```python
infra.action_to_dict(result.action)
```

with:

```python
action_to_dict(result.action)
```

In `examples/03_data_collection.py`, add:

```python
from inferaxis.core.transform import action_to_dict, frame_to_dict
```

Then replace:

```python
infra.frame_to_dict(...)
infra.action_to_dict(...)
```

with:

```python
frame_to_dict(...)
action_to_dict(...)
```

In `examples/04_replay_collected_data.py`, add:

```python
from inferaxis.core.transform import (
    action_to_dict,
    coerce_action,
    coerce_frame,
    frame_to_dict,
)
```

Then replace:

```python
infra.frame_to_dict(...)
infra.action_to_dict(...)
infra.coerce_frame(...)
infra.coerce_action(...)
```

with:

```python
frame_to_dict(...)
action_to_dict(...)
coerce_frame(...)
coerce_action(...)
```

- [ ] **Step 3: Move error/spec imports in tests outside `tests/test_interfaces.py`**

In tests that use `infra.InterfaceValidationError`, add:

```python
from inferaxis.core.errors import InterfaceValidationError
```

Then replace:

```python
infra.InterfaceValidationError
```

with:

```python
InterfaceValidationError
```

In `tests/helpers.py`, add:

```python
from inferaxis.core.schema import PolicySpec, RobotSpec
```

Then replace:

```python
infra.PolicySpec
infra.RobotSpec
```

with:

```python
PolicySpec
RobotSpec
```

- [ ] **Step 4: Run targeted and broad tests**

Run:

```bash
uv run pytest tests/test_interfaces.py -q
uv run pytest tests/test_profiling.py tests/test_runtime_engine.py tests/test_scheduler.py tests/test_scheduler_components.py -q
```

Expected: PASS after all imports are updated.

- [ ] **Step 5: Run example smoke checks**

Run:

```bash
uv run python examples/01_sync_inference.py
uv run python examples/02_async_inference.py
uv run python examples/06_async_inference_with_rtc.py
```

Expected: each command exits with code 0 and prints `example N passed.`.

- [ ] **Step 6: Commit**

```bash
git add src/inferaxis/__init__.py examples tests
git commit -m "refactor: slim root public api"
```

---

### Task 3: Remove Legacy Test-Helper Compatibility

**Files:**
- Modify: `tests/helpers.py`
- Modify: `tests/test_interfaces.py`
- Modify: `tests/test_internal_architecture.py`

- [ ] **Step 1: Write the failing helper behavior test**

Add this import to `tests/test_interfaces.py` if it is not already present:

```python
from helpers import make_chunk_request
```

Add this test to `InterfaceTests`:

```python
def test_make_chunk_request_rejects_removed_legacy_fields(self) -> None:
    with self.assertRaises(AssertionError) as ctx:
        make_chunk_request(history_start=0)

    self.assertIn("Unexpected ChunkRequest test fields", str(ctx.exception))
```

- [ ] **Step 2: Run the red test**

Run:

```bash
uv run pytest tests/test_interfaces.py::InterfaceTests::test_make_chunk_request_rejects_removed_legacy_fields -q
```

Expected: FAIL because `make_chunk_request(...)` currently discards `history_start` silently.

- [ ] **Step 3: Remove the legacy-field discard loop**

In `tests/helpers.py`, replace the `make_chunk_request` docstring:

```python
"""Build one modern ChunkRequest for tests."""
```

Delete this whole block:

```python
for legacy_field in (
    "history_start",
    "history_end",
    "overlap_steps",
    "request_trigger_steps",
    "plan_start_step",
    "history_actions",
):
    kwargs.pop(legacy_field, None)
```

Leave the existing unknown-field assertion in place:

```python
if kwargs:
    raise AssertionError(
        f"Unexpected ChunkRequest test fields: {sorted(kwargs.keys())!r}"
    )
```

- [ ] **Step 4: Remove legacy internal-architecture tests**

In `tests/test_internal_architecture.py`, delete:

```python
def test_core_schema_legacy_module_exports_remain_complete(self) -> None:
    ...
```

Keep the tests that check useful current re-export identity and profiling render helpers.

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_interfaces.py::InterfaceTests::test_make_chunk_request_rejects_removed_legacy_fields -q
uv run pytest tests/test_internal_architecture.py tests/test_interfaces.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/helpers.py tests/test_interfaces.py tests/test_internal_architecture.py
git commit -m "refactor: remove legacy test compatibility"
```

---

### Task 4: Rewrite Tests Away From Scheduler Private Proxy Accessors

**Files:**
- Modify: `tests/helpers.py`
- Modify: `tests/test_scheduler.py`
- Modify: `tests/test_runtime_engine.py`
- Modify: `tests/test_profiling.py`

- [ ] **Step 1: Add explicit scheduler test helpers**

Add these helpers to `tests/helpers.py` near `arm_value(...)`:

```python
def accept_scheduler_chunk(
    scheduler: object,
    actions: list[infra.Action],
    *,
    request_step: int | None = None,
    current_raw_step: int | None = None,
    source_plan_length: int | None = None,
) -> None:
    raw_buffer = scheduler._raw_buffer  # type: ignore[attr-defined]
    raw_step = raw_buffer.global_step
    raw_buffer.accept_chunk(
        actions=actions,
        request_step=raw_step if request_step is None else request_step,
        current_raw_step=raw_step if current_raw_step is None else current_raw_step,
        source_plan_length=len(actions)
        if source_plan_length is None
        else source_plan_length,
    )
    scheduler._execution_cursor.reset()  # type: ignore[attr-defined]


def scheduler_raw_actions(scheduler: object) -> list[infra.Action]:
    return scheduler._raw_buffer.remaining_actions()  # type: ignore[attr-defined]


def scheduler_execution_actions(scheduler: object) -> list[infra.Action]:
    return scheduler._execution_cursor.remaining_segment_actions()  # type: ignore[attr-defined]
```

Update the `tests/test_scheduler.py` helper imports to include:

```python
from helpers import (
    DeterministicClock,
    RtcLoggingChunkPolicy,
    RuntimeRobot,
    accept_scheduler_chunk,
    arm_action,
    arm_and_gripper_action,
    arm_value,
    assert_array_equal,
    gripper_value,
    make_chunk_request,
    scheduler_execution_actions,
    scheduler_raw_actions,
)
```

- [ ] **Step 2: Replace test setup that assigns `_buffer`**

In `tests/test_scheduler.py`, replace patterns like:

```python
scheduler._buffer = deque([arm_action(1.0), arm_action(2.0)])
```

with:

```python
accept_scheduler_chunk(scheduler, [arm_action(1.0), arm_action(2.0)])
```

If a test currently sets `_global_step` and `_active_source_plan_length` after `_buffer`, move those values into `accept_scheduler_chunk(...)`:

```python
accept_scheduler_chunk(
    scheduler,
    [arm_action(1.0), arm_action(2.0), arm_action(3.0), arm_action(4.0)],
    request_step=0,
    current_raw_step=2,
    source_plan_length=4,
)
```

- [ ] **Step 3: Replace test reads of `_buffer` and `_execution_buffer`**

Replace:

```python
[arm_value(action) for action in scheduler._buffer]
```

with:

```python
[arm_value(action) for action in scheduler_raw_actions(scheduler)]
```

Replace:

```python
[arm_value(action) for action in scheduler._execution_buffer]
```

with:

```python
[arm_value(action) for action in scheduler_execution_actions(scheduler)]
```

Replace:

```python
while scheduler._buffer or scheduler._execution_buffer:
```

with:

```python
while scheduler._raw_buffer.has_actions or scheduler._execution_cursor.remaining_segment_steps:
```

Replace:

```python
len(scheduler._buffer)
```

with:

```python
scheduler._raw_buffer.remaining_raw_count
```

Replace:

```python
len(scheduler._execution_buffer)
```

with:

```python
scheduler._execution_cursor.remaining_segment_steps
```

- [ ] **Step 4: Replace pending/executor proxy usage in tests**

Replace:

```python
scheduler._pending_future
```

with:

```python
scheduler._pipeline.pending
```

Replace:

```python
scheduler._pending_future = future
```

with:

```python
scheduler._pipeline.pending = future
```

Replace:

```python
first_scheduler._executor
```

with:

```python
first_scheduler._pipeline.executor
```

If `tests/test_scheduler.py` no longer uses `deque` after these replacements,
remove this import:

```python
from collections import deque
```

- [ ] **Step 5: Run the scheduler tests before production deletion**

Run:

```bash
uv run pytest tests/test_scheduler.py tests/test_runtime_engine.py tests/test_profiling.py -q
```

Expected: PASS while compatibility properties still exist, proving test rewrites did not change behavior.

- [ ] **Step 6: Commit**

```bash
git add tests/helpers.py tests/test_scheduler.py tests/test_runtime_engine.py tests/test_profiling.py
git commit -m "test: inspect scheduler real components"
```

---

### Task 5: Delete Scheduler Proxy Accessors

**Files:**
- Modify: `src/inferaxis/runtime/inference/scheduler/core.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/actions.py`
- Modify: `src/inferaxis/runtime/inference/scheduler/bootstrap.py`
- Modify: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing no-proxy test**

Add this test to `tests/test_scheduler.py`:

```python
def test_chunk_scheduler_no_longer_exposes_private_proxy_accessors(self) -> None:
    for name in ("_buffer", "_execution_buffer", "_pending_future", "_executor"):
        self.assertFalse(hasattr(ChunkScheduler, name))
```

- [ ] **Step 2: Run the red test**

Run:

```bash
uv run pytest tests/test_scheduler.py::SchedulerTests::test_chunk_scheduler_no_longer_exposes_private_proxy_accessors -q
```

Expected: FAIL because `ChunkScheduler` still exposes `_buffer`, `_execution_buffer`, `_pending_future`, and `_executor`.

- [ ] **Step 3: Remove proxy properties from scheduler core**

In `src/inferaxis/runtime/inference/scheduler/core.py`, delete the property blocks for:

```python
_buffer
_execution_buffer
_pending_future
_executor
```

Also remove now-unused imports:

```python
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
```

Keep the properties that are still used by production modules:

```python
_global_step
_active_chunk_consumed_steps
_active_chunk_waited_raw_steps
_active_source_plan_length
```

- [ ] **Step 4: Replace production `_buffer` usage**

In `src/inferaxis/runtime/inference/scheduler/actions.py`, replace `_build_execution_segment(...)` with:

```python
def _build_execution_segment(self) -> deque[Action]:
    """Expand the current raw step into execution actions."""

    remaining_actions = self._raw_buffer.remaining_actions()
    if not remaining_actions:
        return deque()

    left_action = remaining_actions[0]
    if self.interpolation_steps <= 0 or len(remaining_actions) <= 1:
        return deque([left_action])

    segment: list[Action] = [left_action]
    right_action = remaining_actions[1]
    for interpolation_index in range(1, self.interpolation_steps + 1):
        right_weight = interpolation_index / float(self.interpolation_steps + 1)
        segment.append(
            self._interpolate_action(
                left_action,
                right_action,
                right_weight=right_weight,
            )
        )
    return deque(segment)
```

- [ ] **Step 5: Replace production `_pending_future` usage**

In `src/inferaxis/runtime/inference/scheduler/bootstrap.py`, replace:

```python
if self._raw_buffer.has_actions or self._pending_future is not None:
```

with:

```python
if self._raw_buffer.has_actions or self._pipeline.pending is not None:
```

- [ ] **Step 6: Run targeted scheduler tests**

Run:

```bash
uv run pytest tests/test_scheduler.py tests/test_scheduler_components.py -q
```

Expected: PASS.

- [ ] **Step 7: Search for removed proxy names**

Run:

```bash
rg -n "\\._buffer|\\._execution_buffer|\\._pending_future|\\._executor" src tests
```

Expected: no matches for the four removed proxy names. Matches for `_raw_buffer`, `_execution_cursor`, or `_pipeline` are expected and acceptable.

- [ ] **Step 8: Commit**

```bash
git add src/inferaxis/runtime/inference/scheduler tests
git commit -m "refactor: remove scheduler proxy accessors"
```

---

### Task 6: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `docs/examples_guide.md`
- Modify: `docs/plain_objects_guide.md`

- [ ] **Step 1: Update docs for the minimal root API**

In README files, make the public API list match:

```markdown
- `Frame`
- `Action`
- `Command`
- `BuiltinCommandKind`
- `ChunkRequest`
- `run_step(...)`
- `InferenceRuntime(...)`
- `InferenceMode`
- `RealtimeController`
```

When transform helpers are mentioned, refer to:

```python
from inferaxis.core.transform import action_to_dict, frame_to_dict
```

Do not describe `coerce_action`, `coerce_frame`, `action_to_dict`, `frame_to_dict`,
`check_pair`, `PolicySpec`, `RobotSpec`, or `InterfaceValidationError` as root
exports.

- [ ] **Step 2: Verify removed old profiling names are absent from current docs**

Run:

```bash
rg -n "profile_sync_inference|recommend_inference_mode" README.md README_CN.md docs/examples_guide.md docs/plain_objects_guide.md src examples tests
```

Expected: only negative-public-API assertions in tests may remain. No README,
example, production, or guide text should mention those old entrypoints.

- [ ] **Step 3: Run ruff**

Run:

```bash
uv run ruff check src/inferaxis examples tests
uv run ruff format --diff src/inferaxis examples tests
```

Expected: `All checks passed!` and no Python files would be reformatted.

- [ ] **Step 4: Run full tests**

Run:

```bash
uv run pytest
```

Expected: all tests pass. Existing warning tests may still emit the current RTC latency warnings.

- [ ] **Step 5: Run example smoke checks**

Run:

```bash
uv run python examples/01_sync_inference.py
uv run python examples/02_async_inference.py
uv run python examples/05_profile_inference_latency.py --target-hz 50 --policy-latency-ms 1 --stable-sample-count 2 --output-dir tmp/example5-slimdown-check
uv run python examples/06_async_inference_with_rtc.py
uv run python examples/07_benchmark_runtime.py
```

Expected: examples exit with code 0. Example 5 writes `runtime_profile.json` and
`runtime_profile.html` under `tmp/example5-slimdown-check`.

- [ ] **Step 6: Commit final docs and verification updates**

```bash
git add README.md README_CN.md docs examples tests src
git commit -m "docs: document slim public api"
```
