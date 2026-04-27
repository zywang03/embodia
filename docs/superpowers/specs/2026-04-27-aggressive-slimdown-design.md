# Aggressive Slimdown Design

Date: 2026-04-27

## Goal

Slim inferaxis aggressively while preserving the runtime contract needed by the
examples and current async scheduling behavior.

The root public API is reduced to:

- `Frame`
- `Action`
- `Command`
- `BuiltinCommandKind`
- `ChunkRequest`
- `InferenceRuntime`
- `InferenceMode`
- `run_step`
- `RealtimeController`

Everything outside that root API may remain available from deeper modules only
when still needed internally, or may be removed when it exists only for legacy
compatibility.

## Non-Goals

This cleanup will not rewrite the scheduler algorithm, change RTC semantics, or
change the live runtime profile JSON/HTML format unless that format depends on
code being removed.

## Approach

Use a slice-by-slice cleanup with tests as the gate. Each slice starts by
updating tests to encode the new smaller surface, then implementation removes
the now-unwanted compatibility code.

### Slice 1: Public API Surface

Root exports keep only the approved minimal API. Items such as `PolicySpec`,
`RobotSpec`, `check_pair`, transform helpers, and serialization helpers stop
being root-level day-to-day exports. If still useful, they can remain in their
own submodules.

Docs and tests should stop describing the larger legacy root surface as stable.

### Slice 2: Legacy Compatibility

Remove compatibility code and tests that exist only for old request or module
shapes. Examples include helper logic that silently accepts removed
`ChunkRequest` fields and internal-architecture tests that protect historical
module boundaries instead of current behavior.

Schema and validation code that the runtime still actively uses stays unless a
test proves it is unreachable.

### Slice 3: Scheduler Private Compatibility

Remove private compatibility accessors on `ChunkScheduler`, especially old
buffer and executor proxies that only tests use:

- `_buffer`
- `_execution_buffer`
- `_pending_future`
- `_executor`

Tests should inspect the actual components directly (`_raw_buffer`,
`_execution_cursor`, `_pipeline`) where internal assertions are still worth
keeping. The scheduler algorithm and state transitions remain unchanged.

### Slice 4: Profiling, Docs, and Examples

Keep profiling centered on `InferenceRuntime.async_realtime(profile=True)`.
Remove remaining tests, docs, and examples that imply standalone sync profiling
or mode recommendation are part of the supported surface.

## Testing Strategy

For each slice:

1. Add or update tests for the smaller contract.
2. Run the targeted tests and confirm the expected red state when appropriate.
3. Remove code until the targeted tests pass.
4. Run `uv run ruff check src/inferaxis examples tests`.
5. Run `uv run pytest`.

## Risks

The main risk is deleting an internal helper that examples or downstream imports
still use. The mitigation is to preserve deeper-module imports when they serve a
real current purpose and only shrink the root public API aggressively.

Scheduler private cleanup is the riskiest slice because many tests inspect
private state. That work should be done after the root/API cleanup and should
avoid changing scheduling behavior.
