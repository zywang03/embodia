# Inferaxis Scheduler v2 Realtime Optimization Design

## Context

`inferaxis` is a runtime layer for local robot/action-source inference loops.
The async runtime is latency-sensitive: each control step should spend as
little time as possible on framework bookkeeping so slow model inference can
happen in the background without adding avoidable jitter to action execution.

The project now has a cleaner internal architecture, but the scheduler core is
still organized as one `ChunkScheduler` dataclass wired to many helper
functions. That shape is testable, yet the hot path still pays costs that are
easy to lose sight of:

- Multiple list/deque conversions when accepting chunks and expanding
  execution actions.
- Per-step validation decisions spread across runtime, scheduler, and action
  normalization.
- Request lifecycle, latency estimation, RTC window construction, and execution
  buffer movement sharing the same state object.
- Interactive RTC startup confirmation that can block unattended realtime
  processes.
- No dedicated benchmark entrypoint for comparing runtime latency before and
  after scheduler changes.

## Goals

1. Reduce average latency and jitter in `run_step(...)`,
   `InferenceRuntime._run_step_impl(...)`, and `ChunkScheduler.next_action(...)`.
2. Rebuild the scheduler internals around explicit buffer, cursor, pipeline,
   latency, and RTC components.
3. Keep public runtime usage mostly compatible while allowing small API
   additions that make realtime intent clearer.
4. Add a lightweight benchmark example so optimization claims can be measured.
5. Keep existing scheduler, RTC, profiling, and example behavior compatible
   unless a small API addition is explicitly documented.
6. Make the implementation easier to reason about by giving each internal
   component one job.

## Non-Goals

1. Do not change the `run_step(...)` calling model.
2. Do not replace `policy.infer(frame, request)` or the plain-object adapter
   model.
3. Do not remove `startup_validation_only`; keep it as a compatibility alias.
4. Do not introduce runtime dependencies.
5. Do not turn benchmark results into strict CI pass/fail thresholds.
6. Do not optimize profiling renderers; they are cold-path reporting code.

## Compatibility Contract

These calls must continue to work:

```python
import inferaxis as infra

runtime = infra.InferenceRuntime(
    mode=infra.InferenceMode.ASYNC,
    startup_validation_only=True,
)

result = infra.run_step(
    observe_fn=robot.get_obs,
    act_fn=robot.send_action,
    act_src_fn=policy.infer,
    runtime=runtime,
)
```

`ChunkRequest` fields keep their current meaning:

- `request_step`
- `active_chunk_length`
- `remaining_steps`
- `latency_steps`
- `rtc_args`
- `prev_action_chunk`
- `inference_delay`
- `execute_horizon`

The runtime may add new public options, but existing examples and tests should
continue to run without rewriting user code.

## Public API Additions

### Validation Strategy

Add `validation` to `InferenceRuntime` and `ChunkScheduler`:

```python
runtime = infra.InferenceRuntime(
    mode=infra.InferenceMode.ASYNC,
    validation="startup",
)
```

Accepted values:

- `"always"`: validate frame/action payloads on every step.
- `"startup"`: validate configuration, startup frame, startup chunk, and chunk
  admission; skip full validation in stable hot-path steps.
- `"off"`: trust pre-normalized `Frame` and `Action` objects and keep only
  essential scheduler invariants.

Compatibility:

- `startup_validation_only=True` maps to `validation="startup"`.
- `startup_validation_only=False` maps to `validation="always"`.
- Passing both `validation` and `startup_validation_only` is allowed only when
  they agree; conflicts raise `InterfaceValidationError` with a direct message.

Recommended default: preserve current behavior by mapping the current default
`startup_validation_only=True` to `validation="startup"`.

### Async Realtime Preset

Add a small constructor helper:

```python
runtime = infra.InferenceRuntime.async_realtime(
    control_hz=50.0,
    execution_steps=3,
    enable_rtc=True,
)
```

The preset sets visible, documented defaults rather than hidden magic:

- `mode=InferenceMode.ASYNC`
- `validation="startup"`
- `profile=False`
- `interpolation_steps=0`
- `ensemble_weight=None`

Callers can override any supported option through keyword arguments.

### Slow RTC Bootstrap Policy

Replace the current unconditional interactive confirmation path with:

```python
runtime = infra.InferenceRuntime(
    mode=infra.InferenceMode.ASYNC,
    enable_rtc=True,
    slow_rtc_bootstrap="warn",
)
```

Accepted values:

- `"warn"`: emit `RuntimeWarning` and continue.
- `"error"`: raise `InterfaceValidationError`.
- `"confirm"`: preserve the old interactive prompt behavior.

The default should avoid surprise blocking in unattended realtime processes.
Use `"warn"` for a compatibility-friendly default.

## Scheduler v2 Architecture

Keep `ChunkScheduler` as the public entrypoint. Internally, split its state and
hot-path responsibilities into focused units.

### RawChunkBuffer

Owns the active raw action chunk and raw-step position.

Responsibilities:

- Store the current chunk as a list of shared read-only `Action` references.
- Track raw index, consumed raw steps, waited raw steps, and global raw step.
- Expose the current action without copying.
- Drop stale prefixes by moving an index rather than rebuilding a deque.
- Provide read-only windows for request launch and RTC construction.

`RawChunkBuffer` does not know about futures, latency math, profiling, or
policy calls.

### ExecutionCursor

Maps raw actions to emitted control-step actions.

Responsibilities:

- Fast path: when `interpolation_steps == 0`, return the current raw action and
  advance with no per-step container allocation.
- Interpolation path: generate only the current interpolated action when needed.
- Track the current interpolation slot for one raw segment.
- Advance the raw buffer only when the current segment is fully consumed.

This replaces the current `_execution_buffer` pre-expansion in the common no
interpolation case.

### RequestPipeline

Owns request launch and completion flow.

Responsibilities:

- Build request jobs from the current buffer, latency hint, RTC window, and
  launch control step.
- Manage the pending future and background executor.
- Accept ready completed chunks without blocking.
- Block only when no executable action is available and a pending request must
  be consumed.
- Record launch/reply/accept/error profiler events when profiling is enabled.
- Admit completed chunks through `RawChunkBuffer` and reset the
  `ExecutionCursor`.

`RequestPipeline` does not blend action values itself; it delegates action
normalization and blending to action helpers.

### LatencyTracker

Owns latency estimate state and control/raw projection.

Responsibilities:

- Track warmup requests, profile delay requests, fixed latency overrides, EMA
  beta, and observation counts.
- Convert measured request durations into control-step estimates.
- Project control-step latency into raw-step request hints.
- Apply signed latency offset after projection.

This reduces scattered latency state inside `ChunkScheduler`.

### RtcWindowBuilder

Owns RTC request-window construction and constraints.

Responsibilities:

- Lock and validate stable source raw chunk length.
- Build `prev_action_chunk`, `inference_delay`, and `execute_horizon`.
- Preserve fixed-length, right-padded, shared-reference semantics.
- Validate `execution_steps < chunk_total_length - steps_before_request`.
- Handle slow bootstrap policy through `slow_rtc_bootstrap`.

### ChunkScheduler

Becomes a small orchestrator that keeps the existing public methods:

- `next_action(...)`
- `bootstrap(...)`
- `reset()`
- `close()`
- `estimated_latency_steps()`
- `latency_estimate_ready()`

The hot path should read like a control loop:

```python
pipeline.accept_ready()
action = cursor.next()
buffer.advance_if_needed()
pipeline.launch_if_needed(frame)
```

The implementation may differ in details, but the code should make the same
sequence obvious.

## Data Flow

### Startup

1. `InferenceRuntime` validates configuration.
2. Validation policy is resolved from `validation` and
   `startup_validation_only`.
3. `ChunkScheduler` creates buffer, cursor, latency tracker, RTC builder, and
   request pipeline.
4. Startup frame and first chunk are validated unless validation is `"off"`.
5. Warmup/profile requests seed latency estimates and initial executable chunk.

### Stable Hot Path

1. `run_step(...)` resolves the frame.
2. Runtime decides whether this step needs validation.
3. Scheduler accepts completed background work only if ready.
4. Scheduler emits the current executable action.
5. Scheduler advances raw/control counters.
6. Scheduler launches the next request only when trigger conditions are met.
7. Action execution happens through the existing `act_fn` path.
8. Profiling work is skipped unless a live profiler is attached.

The no-profiling, no-RTC, no-interpolation path should allocate as little as
practically possible per step.

## Error Handling

- Configuration errors still raise `InterfaceValidationError`.
- Background source exceptions still surface as `InterfaceValidationError` when
  the future is consumed.
- `validation="off"` does not suppress essential scheduler invariants such as
  empty chunks, invalid RTC chunk length, or impossible execution windows.
- Slow RTC bootstrap follows `slow_rtc_bootstrap`.
- Error messages should name the runtime boundary in plain language:
  `act_src_fn(frame, request)`, `ChunkScheduler`, `RtcWindowBuilder`, or
  `InferenceRuntime.validation`.

## Benchmark Design

Add:

```text
examples/07_benchmark_runtime.py
```

The benchmark should run deterministic local policies and report:

- sync `run_step(...)`
- async scheduler without RTC
- async scheduler with RTC

Metrics:

- warmup iteration count
- measured iteration count
- p50 latency
- p95 latency
- max latency
- mean latency

The script should use `time.perf_counter()` and keep output compact. It should
not fail CI based on timing.

Example output:

```text
sync run_step:
  mean: 0.041 ms
  p50:  0.039 ms
  p95:  0.058 ms
  max:  0.083 ms
```

## Testing Strategy

1. Keep all existing tests passing.
2. Add focused unit tests for:
   - `RawChunkBuffer`
   - `ExecutionCursor`
   - `RequestPipeline`
   - `LatencyTracker`
   - `RtcWindowBuilder`
3. Add API tests for:
   - `validation="always"`, `"startup"`, and `"off"`
   - `startup_validation_only` compatibility
   - conflicting validation parameters
   - `InferenceRuntime.async_realtime(...)`
   - `slow_rtc_bootstrap`
4. Keep scheduler behavior tests for:
   - overlap blending
   - interpolation
   - latency offset
   - RTC request metadata
   - stale chunk handling
   - async background request completion
5. Add benchmark smoke test only if it is deterministic and fast enough; the
   benchmark itself remains an example script.

Required verification after implementation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall -q src tests examples
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests examples
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests examples
UV_CACHE_DIR=/tmp/uv-cache uv run python examples/07_benchmark_runtime.py
```

## Risks and Mitigations

- **Risk: scheduler rewrite changes subtle RTC timing behavior.**
  Mitigation: keep existing RTC tests, add direct `RtcWindowBuilder` tests, and
  compare request metadata before/after.

- **Risk: validation skipping hides malformed user inputs.**
  Mitigation: keep `"always"` available, keep `"startup"` as validated default,
  document `"off"` as a trust boundary, and preserve essential invariants.

- **Risk: object reuse invites accidental mutation.**
  Mitigation: preserve shared read-only action convention and avoid reusing
  mutable `ChunkRequest` objects across policy calls.

- **Risk: benchmark is noisy.**
  Mitigation: present benchmark as a local comparison tool, not a CI threshold.

- **Risk: too much changes in one step.**
  Mitigation: implement in phases: benchmark first, internal components behind
  existing tests second, API additions third, scheduler integration last.

## Implementation Order

1. Add benchmark example and baseline scheduler performance smoke coverage.
2. Add validation API tests and compatibility tests.
3. Introduce `ValidationPolicy` resolution while preserving existing behavior.
4. Add `RawChunkBuffer` and `ExecutionCursor` with focused tests.
5. Move latency state into `LatencyTracker` with focused tests.
6. Move RTC window logic into `RtcWindowBuilder` with focused tests.
7. Introduce `RequestPipeline` and wire it into `ChunkScheduler`.
8. Add `InferenceRuntime.async_realtime(...)`.
9. Add `slow_rtc_bootstrap` handling.
10. Run full verification and benchmark.

