# Inferaxis Internal Architecture Refactor Design

## Context

`inferaxis` is a Python package for unified runtime data contracts and
adaptive inference scheduling. The public package surface is intentionally
small: users primarily import `inferaxis as infra` and work with `Frame`,
`Action`, `Command`, `run_step(...)`, `InferenceRuntime(...)`, and profiling
helpers.

The current test baseline passes with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

The suite currently contains 143 tests. The implementation is functionally
healthy, but several files carry too many responsibilities:

- `src/inferaxis/core/schema.py` contains schema models, command kind
  registration, primitive validation helpers, object validation, and
  robot-policy compatibility checks.
- `src/inferaxis/runtime/inference/profiling/render.py` contains multiple SVG
  and HTML renderers in one large module.
- `tests/test_inference_runtime.py` combines runtime, scheduler, RTC,
  interpolation, and profiling tests in one large file.
- `pyproject.toml` has project metadata and dependencies, but only minimal
  development tooling configuration.

## Goals

1. Keep the public API fully compatible.
2. Reduce module size and responsibility overlap.
3. Make future schema, scheduler, and profiling work easier to extend.
4. Keep each refactor step behavior-preserving and testable.
5. Standardize development commands around `uv`.

## Non-Goals

1. Do not rename public classes, functions, or root package exports.
2. Do not change runtime scheduling algorithms or profiling semantics.
3. Do not introduce new runtime dependencies.
4. Do not replace the plain-object integration model.
5. Do not rewrite tests into a different framework as part of the same change.

## Compatibility Contract

The following imports and behaviors must keep working:

```python
import inferaxis as infra

infra.Frame
infra.Command
infra.Action
infra.BuiltinCommandKind
infra.run_step
infra.InferenceRuntime
infra.RealtimeController
infra.profile_sync_inference
infra.recommend_inference_mode
```

The existing `src/inferaxis/core/schema.py` and
`src/inferaxis/runtime/inference/profiling/render.py` modules must remain
importable. They will become compatibility-facing modules that re-export or
delegate to smaller internal modules.

## Proposed Architecture

### Core Schema

Keep `src/inferaxis/core/schema.py` as the compatibility module. Move internal
responsibilities into focused modules:

- `src/inferaxis/core/schema_models.py`
  - Owns dataclasses and schema object methods:
    `Frame`, `Command`, `Action`, `ComponentSpec`, `RobotSpec`,
    `PolicyOutputSpec`, and `PolicySpec`.
  - Uses validation helpers imported from the validation module.
  - Keeps constructor behavior and array-copy behavior unchanged.

- `src/inferaxis/core/command_kinds.py`
  - Owns command kind constants and registry:
    `BuiltinCommandKind`, `CommandKindSpec`, `COMMAND_KIND_REGISTRY`,
    `CUSTOM_COMMAND_KIND_PREFIX`, `register_command_kind(...)`,
    `get_command_kind_spec(...)`, `is_known_command_kind(...)`, and
    `is_custom_command_kind_name(...)`.
  - Registers built-in command kinds at import time as today.

- `src/inferaxis/core/schema_validation.py`
  - Owns primitive validation helpers and structural validators:
    `_ensure_non_empty_string`, `_ensure_string_key_dict`,
    `_ensure_string_list`, `_ensure_ndarray`, `_ensure_positive_int`,
    `_ensure_bool`, `validate_frame(...)`, `validate_command(...)`,
    `validate_action(...)`, `validate_component_spec(...)`,
    `validate_robot_spec(...)`, `validate_policy_output_spec(...)`, and
    `validate_policy_spec(...)`.

- `src/inferaxis/core/schema_compat.py`
  - Owns cross-object compatibility checks:
    `ensure_action_supported_by_robot(...)` and
    `ensure_action_matches_policy_spec(...)`.

This keeps object definitions, registry behavior, primitive validation, and
cross-object compatibility separate while preserving the existing import path.

### Profiling Rendering

Keep `src/inferaxis/runtime/inference/profiling/render.py` as the compatibility
module. Move rendering responsibilities into focused modules:

- `profiling/render_common.py`
  - Formatting helpers and small SVG utilities shared by the renderers.

- `profiling/render_async_trace.py`
  - `_step_plot_points(...)` and `_async_buffer_trace_svg(...)`.

- `profiling/render_runtime_svg.py`
  - Runtime SVG chart helpers and `_runtime_profile_svg(...)`.

- `profiling/render_runtime_html.py`
  - `_runtime_request_status(...)`, `_seconds_to_ms(...)`, and
    `_runtime_profile_html(...)`.

The rendering modules should keep private helper names private. Existing code
that imports helpers from `profiling.render` should continue to work by
re-exporting the same names from the compatibility module.

### Tests

Keep the current test behavior. Split the large runtime test module by domain
after the core and render refactors are stable:

- `tests/test_runtime_engine.py`
  - `InferenceRuntime`, `run_step`, source switching, public exports.

- `tests/test_scheduler.py`
  - `ChunkScheduler`, overlap blending, interpolation, latency estimates, RTC
    request metadata.

- `tests/test_profiling.py`
  - Sync profiling, async buffer trace, SVG/HTML report behavior, live profile
    output.

- `tests/test_interfaces.py`
  - Keep schema, coercion, public interface, and function-first flow coverage.

Shared helper classes and action builders should move into `tests/helpers.py`
when used by multiple test files.

### Development Tooling

Use `uv` as the standard project entrypoint. Keep the project-level Tsinghua
index setting:

```toml
[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
```

Add a development dependency group with test and style tooling:

```toml
[dependency-groups]
dev = [
    "pytest>=8,<10",
    "ruff>=0.14,<1",
]
```

Add conservative tool configuration:

- `pytest` discovers `tests`.
- `ruff` checks syntax, unused imports, import ordering, and common bug risks.
- `ruff format` can be introduced in check mode first to avoid broad unrelated
  formatting churn.

## Data and Control Flow

The runtime data flow stays unchanged:

1. User code produces or provides a `Frame`.
2. `run_step(...)` resolves and validates the frame.
3. The action source returns an `Action` or `list[Action]`.
4. Sync mode executes one action directly.
5. Async mode delegates chunk management to `ChunkScheduler`.
6. Profiling records requests, chunks, and emitted actions.
7. Rendering turns profile models into SVG or HTML reports.

The refactor changes module boundaries only. It does not alter frame/action
normalization, validation rules, scheduler state transitions, or profiling
model content.

## Error Handling

All existing exception types and message intent remain unchanged:

- Public validation errors continue to use `InterfaceValidationError`.
- Duplicate command kind registration continues to raise `ValueError`.
- Unknown command kind lookup continues to raise `KeyError`.
- Type and shape validation keep the same field names where practical.

Where helpers move to new modules, compatibility imports prevent consumers from
seeing a new error source or public import path.

## Implementation Order

1. Add development tooling configuration and confirm both unittest and pytest
   can run through `uv`.
2. Add API compatibility tests around root exports and legacy module imports.
3. Split command kind registry from `schema.py`.
4. Split schema dataclasses from `schema.py`.
5. Split validation and compatibility checks from `schema.py`.
6. Split profiling rendering into common, async trace, runtime SVG, and runtime
   HTML modules.
7. Split the large runtime test module by domain.
8. Run the full verification suite after each major extraction.

Each step must preserve passing behavior before moving on.

## Verification

Required verification commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests
```

If `ruff format --check` reports broad existing formatting differences, the
first implementation pass should either narrow formatting scope or introduce
formatting in a dedicated final step after behavior-preserving refactors pass.

## Risks and Mitigations

- Import cycles between models, validation, and command kind registry.
  Mitigation: keep primitive helpers in `schema_validation.py`, avoid importing
  compatibility checks from model modules, and centralize re-exports in
  `schema.py`.

- Accidentally changing public imports.
  Mitigation: add explicit compatibility tests before moving code.

- Large render extraction changing report output.
  Mitigation: preserve helper names, move code mechanically first, and rely on
  existing profiling SVG/HTML tests before cleanup.

- Test split hiding shared state assumptions.
  Mitigation: split tests after code extractions, run the full test suite after
  each file move, and keep shared setup in `tests/helpers.py`.

## Success Criteria

The refactor is successful when:

1. Existing public API imports continue to work.
2. Existing examples remain valid.
3. Full unittest and pytest suites pass through `uv`.
4. Ruff checks pass or explicitly documented format-only follow-up remains.
5. `schema.py` and `profiling/render.py` are reduced to compatibility-oriented
   modules with focused internal modules handling the real responsibilities.
6. Tests are grouped by domain so future scheduler, runtime, and profiling
   changes can be developed independently.
