# Inferaxis Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor inferaxis internals into smaller, focused modules while keeping every public import, behavior, example, and existing test compatible.

**Architecture:** Keep existing public modules as compatibility facades. Move schema models, command kind registry, schema validation, schema compatibility checks, and profiling renderers into focused internal modules. Add conservative `uv`-managed development tooling before moving code so each extraction can be verified independently.

**Tech Stack:** Python 3.11+, dataclasses, numpy, plotly, uv, unittest, pytest, ruff.

---

## Current State and Safety Notes

- The design spec is `docs/superpowers/specs/2026-04-26-inferaxis-refactor-design.md`.
- Current baseline command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: `Ran 143 tests` and `OK`.

- The working tree may already contain uncommitted `pyproject.toml` and
  `uv.lock` changes for the Tsinghua uv index. Preserve those changes.
- `.codex` is an unrelated untracked file. Do not add or modify it.
- Public compatibility is mandatory. `import inferaxis as infra`,
  `inferaxis.core.schema`, and
  `inferaxis.runtime.inference.profiling.render` must continue to work.

## Target File Structure

- Modify: `pyproject.toml`
  - Keeps package metadata.
  - Adds dev dependency group plus pytest and ruff settings.

- Modify: `uv.lock`
  - Updated by `uv lock` after dev dependencies are added.

- Create: `tests/test_internal_architecture.py`
  - Adds failing compatibility tests for the new internal module boundaries.

- Create: `src/inferaxis/core/command_kinds.py`
  - Owns `BuiltinCommandKind`, `CommandKindSpec`,
    `COMMAND_KIND_REGISTRY`, `CUSTOM_COMMAND_KIND_PREFIX`,
    `register_command_kind`, `get_command_kind_spec`,
    `is_known_command_kind`, and `is_custom_command_kind_name`.

- Create: `src/inferaxis/core/schema_models.py`
  - Owns `Frame`, `Command`, `Action`, `ComponentSpec`, `RobotSpec`,
    `PolicyOutputSpec`, and `PolicySpec`.

- Create: `src/inferaxis/core/schema_validation.py`
  - Owns primitive schema validators and structural validation functions.
  - Uses local imports inside structural validation functions to avoid import
    cycles with `schema_models.py`.

- Create: `src/inferaxis/core/schema_compat.py`
  - Owns cross-object compatibility functions:
    `ensure_action_supported_by_robot` and
    `ensure_action_matches_policy_spec`.

- Modify: `src/inferaxis/core/schema.py`
  - Becomes the compatibility facade that re-exports the same names as today.

- Create: `src/inferaxis/runtime/inference/profiling/render_common.py`
  - Owns shared SVG formatting helpers.

- Create: `src/inferaxis/runtime/inference/profiling/render_async_trace.py`
  - Owns `_step_plot_points` and `_async_buffer_trace_svg`.

- Create: `src/inferaxis/runtime/inference/profiling/render_runtime_svg.py`
  - Owns `_runtime_action_channels`,
    `_runtime_chunk_action_channel_keys`, `_runtime_combined_step_trace`,
    `_runtime_action_trace_section`, and `_runtime_profile_svg`.

- Create: `src/inferaxis/runtime/inference/profiling/render_runtime_html.py`
  - Owns `_runtime_request_status`, `_seconds_to_ms`, and
    `_runtime_profile_html`.

- Modify: `src/inferaxis/runtime/inference/profiling/render.py`
  - Becomes the compatibility facade that re-exports existing private helper
    names used by current tests and package exports.

- Modify: `tests/helpers.py`
  - Gains shared runtime/scheduler/profiling test helpers extracted from
    `tests/test_inference_runtime.py`.

- Split: `tests/test_inference_runtime.py`
  - Move tests into:
    `tests/test_runtime_engine.py`,
    `tests/test_scheduler.py`,
    and `tests/test_profiling.py`.
  - Keep `tests/test_inference_runtime.py` absent after the split.

## Task 1: Add uv Development Tooling

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Edit `pyproject.toml` with conservative dev tooling**

Add these sections after the existing `[tool.uv]` section:

```toml
[dependency-groups]
dev = [
    "pytest>=8,<10",
    "ruff>=0.14,<1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E9", "F63", "F7", "F82"]

[tool.ruff.format]
docstring-code-format = true
```

Keep the existing Tsinghua index setting:

```toml
[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
```

- [ ] **Step 2: Refresh the lockfile**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_HTTP_TIMEOUT=120 uv lock
```

Expected: command exits `0` and `uv.lock` includes `pytest`, `ruff`, and their
transitive dependencies.

- [ ] **Step 3: Sync the development environment**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_HTTP_TIMEOUT=120 uv sync --group dev --locked
```

Expected: command exits `0` and installs the project plus dev tooling.

- [ ] **Step 4: Verify unittest still passes**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: `Ran 143 tests` and `OK`. Existing runtime warnings about RTC delay
are acceptable.

- [ ] **Step 5: Verify pytest can collect the existing suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected: `143 passed`. Existing runtime warnings about RTC delay are
acceptable.

- [ ] **Step 6: Verify ruff check is clean**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Expected: `All checks passed!`.

- [ ] **Step 7: Commit**

Run:

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add uv dev tooling"
```

Expected: commit succeeds and includes only `pyproject.toml` and `uv.lock`.

## Task 2: Add Core Internal Architecture Compatibility Tests

**Files:**
- Create: `tests/test_internal_architecture.py`

- [ ] **Step 1: Create the failing core compatibility test file**

Create `tests/test_internal_architecture.py` with this content:

```python
"""Compatibility tests for internal module boundaries."""

from __future__ import annotations

import unittest

import inferaxis as infra


class InternalArchitectureTests(unittest.TestCase):
    """Ensure new internal modules preserve legacy public imports."""

    def test_core_schema_internal_modules_reexport_compatible_objects(self) -> None:
        from inferaxis.core import command_kinds
        from inferaxis.core import schema as schema_module
        from inferaxis.core import schema_compat
        from inferaxis.core import schema_models
        from inferaxis.core import schema_validation

        self.assertIs(schema_models.Frame, infra.Frame)
        self.assertIs(schema_models.Command, infra.Command)
        self.assertIs(schema_models.Action, infra.Action)
        self.assertIs(command_kinds.BuiltinCommandKind, infra.BuiltinCommandKind)
        self.assertIs(schema_module.Frame, infra.Frame)
        self.assertIs(schema_module.Command, infra.Command)
        self.assertIs(schema_module.Action, infra.Action)
        self.assertIs(
            schema_validation.validate_action,
            schema_module.validate_action,
        )
        self.assertIs(
            schema_compat.ensure_action_supported_by_robot,
            schema_module.ensure_action_supported_by_robot,
        )

    def test_core_schema_legacy_module_exports_remain_complete(self) -> None:
        from inferaxis.core import schema as schema_module

        expected_names = {
            "Action",
            "BuiltinCommandKind",
            "COMMAND_KIND_REGISTRY",
            "CUSTOM_COMMAND_KIND_PREFIX",
            "ComponentSpec",
            "Command",
            "CommandKindSpec",
            "Frame",
            "KNOWN_COMPONENT_TYPES",
            "PolicyOutputSpec",
            "PolicySpec",
            "RobotSpec",
            "ensure_action_matches_policy_spec",
            "ensure_action_supported_by_robot",
            "get_command_kind_spec",
            "is_custom_command_kind_name",
            "is_known_command_kind",
            "register_command_kind",
            "validate_action",
            "validate_command",
            "validate_component_spec",
            "validate_frame",
            "validate_policy_output_spec",
            "validate_policy_spec",
            "validate_robot_spec",
        }

        self.assertEqual(set(schema_module.__all__), expected_names)

```

- [ ] **Step 2: Run the new test and verify it fails for the expected reason**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_internal_architecture -q
```

Expected: `FAILED` with `ImportError` or `ModuleNotFoundError` mentioning one
of the missing core modules such as `command_kinds` or `schema_models`.

- [ ] **Step 3: Keep the failing test uncommitted until Task 3 passes**

Run:

```bash
git status --short tests/test_internal_architecture.py
```

Expected: `?? tests/test_internal_architecture.py`. Task 3 commits this test
with the core schema extraction after the test passes.

## Task 3: Extract Core Schema Modules

**Files:**
- Create: `src/inferaxis/core/command_kinds.py`
- Create: `src/inferaxis/core/schema_models.py`
- Create: `src/inferaxis/core/schema_validation.py`
- Create: `src/inferaxis/core/schema_compat.py`
- Modify: `src/inferaxis/core/schema.py`

- [ ] **Step 1: Create `schema_validation.py` with primitive helpers and validators**

Create `src/inferaxis/core/schema_validation.py` by moving these definitions
from `src/inferaxis/core/schema.py`:

```python
_ensure_non_empty_string
_ensure_string_key_dict
_ensure_string_list
_ensure_ndarray
_ensure_positive_int
_coerce_numpy_mapping
_ensure_bool
_validate_command_kind_name
_kind_uses_component_dof
validate_frame
validate_command
validate_action
validate_component_spec
validate_robot_spec
validate_policy_output_spec
validate_policy_spec
```

Use this import header:

```python
"""Validation helpers for inferaxis schema objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from .arraylike import to_numpy_array
from .errors import InterfaceValidationError
```

Inside `_validate_command_kind_name`, import command-kind helpers locally:

```python
from .command_kinds import (
    CUSTOM_COMMAND_KIND_PREFIX,
    is_custom_command_kind_name,
    is_known_command_kind,
)
```

Inside validators that check concrete classes, import schema model classes
locally. For example:

```python
def validate_frame(frame: object) -> None:
    """Validate a :class:`Frame` instance."""

    from .schema_models import Frame

    if not isinstance(frame, Frame):
        raise InterfaceValidationError(
            f"frame must be a Frame instance, got {type(frame).__name__}."
        )
```

Keep the existing validation bodies and error messages unchanged after the
local imports are added.

Add this `__all__` at the bottom:

```python
__all__ = [
    "_coerce_numpy_mapping",
    "_ensure_bool",
    "_ensure_ndarray",
    "_ensure_non_empty_string",
    "_ensure_positive_int",
    "_ensure_string_key_dict",
    "_ensure_string_list",
    "_kind_uses_component_dof",
    "_validate_command_kind_name",
    "validate_action",
    "validate_command",
    "validate_component_spec",
    "validate_frame",
    "validate_policy_output_spec",
    "validate_policy_spec",
    "validate_robot_spec",
]
```

- [ ] **Step 2: Create `command_kinds.py`**

Create `src/inferaxis/core/command_kinds.py` by moving these definitions from
`schema.py`:

```python
CUSTOM_COMMAND_KIND_PREFIX
_USES_COMPONENT_DOF_META_KEY
BuiltinCommandKind
CommandKindSpec
COMMAND_KIND_REGISTRY
register_command_kind
get_command_kind_spec
is_known_command_kind
is_custom_command_kind_name
_builtin_command_kind_specs
_register_builtin_command_kinds
```

Use this import header:

```python
"""Command kind registry for inferaxis schema objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .schema_validation import (
    _ensure_non_empty_string,
    _ensure_positive_int,
    _ensure_string_key_dict,
    _ensure_string_list,
)
```

Preserve the current built-in registration behavior by keeping:

```python
_register_builtin_command_kinds()
```

at module import time.

Add this `__all__` at the bottom:

```python
__all__ = [
    "BuiltinCommandKind",
    "COMMAND_KIND_REGISTRY",
    "CUSTOM_COMMAND_KIND_PREFIX",
    "CommandKindSpec",
    "_USES_COMPONENT_DOF_META_KEY",
    "get_command_kind_spec",
    "is_custom_command_kind_name",
    "is_known_command_kind",
    "register_command_kind",
]
```

- [ ] **Step 3: Create `schema_models.py`**

Create `src/inferaxis/core/schema_models.py` by moving these definitions from
`schema.py`:

```python
KNOWN_COMPONENT_TYPES
_materialize_command
Frame
Command
Action
ComponentSpec
RobotSpec
PolicyOutputSpec
PolicySpec
```

Use this import header:

```python
"""Schema model objects for inferaxis runtime interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import time
from typing import Any

import numpy as np

from .arraylike import to_numpy_array
from .errors import InterfaceValidationError
from .schema_validation import (
    _coerce_numpy_mapping,
    _ensure_bool,
    _ensure_non_empty_string,
    _ensure_string_key_dict,
)
```

Preserve all constructor behavior and method bodies from `schema.py`.

Add this `__all__` at the bottom:

```python
__all__ = [
    "Action",
    "Command",
    "ComponentSpec",
    "Frame",
    "KNOWN_COMPONENT_TYPES",
    "PolicyOutputSpec",
    "PolicySpec",
    "RobotSpec",
]
```

- [ ] **Step 4: Create `schema_compat.py`**

Create `src/inferaxis/core/schema_compat.py` by moving these definitions from
`schema.py`:

```python
ensure_action_supported_by_robot
ensure_action_matches_policy_spec
```

Use this import header:

```python
"""Compatibility checks between schema objects."""

from __future__ import annotations

from .command_kinds import get_command_kind_spec, is_known_command_kind
from .schema_models import Action, PolicySpec, RobotSpec
from .schema_validation import (
    _kind_uses_component_dof,
    validate_action,
    validate_policy_spec,
    validate_robot_spec,
)
from .errors import InterfaceValidationError
```

Preserve both function bodies and error messages from `schema.py`.

Add this `__all__` at the bottom:

```python
__all__ = [
    "ensure_action_matches_policy_spec",
    "ensure_action_supported_by_robot",
]
```

- [ ] **Step 5: Replace `schema.py` with a compatibility facade**

Replace `src/inferaxis/core/schema.py` with:

```python
"""Compatibility exports for inferaxis schema objects.

The schema implementation is split across focused modules. This module keeps
the historical import path stable for users and internal code.
"""

from __future__ import annotations

from .command_kinds import (
    COMMAND_KIND_REGISTRY,
    CUSTOM_COMMAND_KIND_PREFIX,
    BuiltinCommandKind,
    CommandKindSpec,
    get_command_kind_spec,
    is_custom_command_kind_name,
    is_known_command_kind,
    register_command_kind,
)
from .schema_compat import (
    ensure_action_matches_policy_spec,
    ensure_action_supported_by_robot,
)
from .schema_models import (
    KNOWN_COMPONENT_TYPES,
    Action,
    Command,
    ComponentSpec,
    Frame,
    PolicyOutputSpec,
    PolicySpec,
    RobotSpec,
)
from .schema_validation import (
    validate_action,
    validate_command,
    validate_component_spec,
    validate_frame,
    validate_policy_output_spec,
    validate_policy_spec,
    validate_robot_spec,
)

__all__ = [
    "Action",
    "BuiltinCommandKind",
    "COMMAND_KIND_REGISTRY",
    "CUSTOM_COMMAND_KIND_PREFIX",
    "ComponentSpec",
    "Command",
    "CommandKindSpec",
    "Frame",
    "KNOWN_COMPONENT_TYPES",
    "PolicyOutputSpec",
    "PolicySpec",
    "RobotSpec",
    "ensure_action_matches_policy_spec",
    "ensure_action_supported_by_robot",
    "get_command_kind_spec",
    "is_custom_command_kind_name",
    "is_known_command_kind",
    "register_command_kind",
    "validate_action",
    "validate_command",
    "validate_component_spec",
    "validate_frame",
    "validate_policy_output_spec",
    "validate_policy_spec",
    "validate_robot_spec",
]
```

- [ ] **Step 6: Run the architecture tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_internal_architecture -q
```

Expected: all tests in `tests.test_internal_architecture` pass.

- [ ] **Step 7: Run the existing interface tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -p test_interfaces.py -q
```

Expected: `Ran 33 tests` and `OK`.

- [ ] **Step 8: Run full unittest suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: all tests pass with `OK`. Existing runtime warnings about RTC delay
are acceptable.

- [ ] **Step 9: Commit**

Run:

```bash
git add tests/test_internal_architecture.py src/inferaxis/core/schema.py src/inferaxis/core/command_kinds.py src/inferaxis/core/schema_models.py src/inferaxis/core/schema_validation.py src/inferaxis/core/schema_compat.py
git commit -m "refactor: split core schema modules"
```

Expected: commit succeeds and includes the core compatibility test plus schema
refactor files.

## Task 4: Extract Profiling Render Modules

**Files:**
- Create: `src/inferaxis/runtime/inference/profiling/render_common.py`
- Create: `src/inferaxis/runtime/inference/profiling/render_async_trace.py`
- Create: `src/inferaxis/runtime/inference/profiling/render_runtime_svg.py`
- Create: `src/inferaxis/runtime/inference/profiling/render_runtime_html.py`
- Modify: `src/inferaxis/runtime/inference/profiling/render.py`
- Modify: `tests/test_internal_architecture.py`

- [ ] **Step 1: Add the failing profiling render compatibility test**

Append this method to `InternalArchitectureTests` in
`tests/test_internal_architecture.py`:

```python
    def test_profiling_render_internal_modules_reexport_helpers(self) -> None:
        from inferaxis.runtime.inference.profiling import render as render_module
        from inferaxis.runtime.inference.profiling import render_async_trace
        from inferaxis.runtime.inference.profiling import render_runtime_html
        from inferaxis.runtime.inference.profiling import render_runtime_svg

        self.assertIs(
            render_async_trace._step_plot_points,
            render_module._step_plot_points,
        )
        self.assertIs(
            render_async_trace._async_buffer_trace_svg,
            render_module._async_buffer_trace_svg,
        )
        self.assertIs(
            render_runtime_svg._runtime_profile_svg,
            render_module._runtime_profile_svg,
        )
        self.assertIs(
            render_runtime_html._runtime_profile_html,
            render_module._runtime_profile_html,
        )
```

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_internal_architecture -q
```

Expected: `FAILED` with `ImportError` or `ModuleNotFoundError` mentioning one
of the missing profiling modules such as `render_async_trace`.

- [ ] **Step 2: Create `render_common.py`**

Create `src/inferaxis/runtime/inference/profiling/render_common.py` with
shared helpers moved from `render.py`:

```python
"""Shared rendering helpers for profiling reports."""

from __future__ import annotations


def _format_profile_value(value: float) -> str:
    """Format one numeric profile value compactly for SVG labels."""

    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.4g}"


__all__ = ["_format_profile_value"]
```

- [ ] **Step 3: Create `render_async_trace.py`**

Create `src/inferaxis/runtime/inference/profiling/render_async_trace.py` by
moving `_step_plot_points(...)` and `_async_buffer_trace_svg(...)` from
`render.py`.

Use this import header:

```python
"""SVG rendering for async buffer traces."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AsyncBufferTrace
```

Keep both function bodies unchanged.

Add this `__all__`:

```python
__all__ = ["_async_buffer_trace_svg", "_step_plot_points"]
```

- [ ] **Step 4: Create `render_runtime_svg.py`**

Create `src/inferaxis/runtime/inference/profiling/render_runtime_svg.py` by
moving these functions from `render.py`:

```python
_runtime_action_channels
_runtime_chunk_action_channel_keys
_runtime_combined_step_trace
_runtime_action_trace_section
_runtime_profile_svg
```

Use this import header:

```python
"""SVG rendering for live runtime inference profiles."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from .render_common import _format_profile_value

if TYPE_CHECKING:
    from .models import RuntimeInferenceProfile
```

Keep the moved function bodies unchanged except for importing
`_format_profile_value` from `render_common.py`.

Add this `__all__`:

```python
__all__ = [
    "_runtime_action_channels",
    "_runtime_action_trace_section",
    "_runtime_chunk_action_channel_keys",
    "_runtime_combined_step_trace",
    "_runtime_profile_svg",
]
```

- [ ] **Step 5: Create `render_runtime_html.py`**

Create `src/inferaxis/runtime/inference/profiling/render_runtime_html.py` by
moving these functions from `render.py`:

```python
_runtime_request_status
_seconds_to_ms
_runtime_profile_html
```

Use this import header:

```python
"""HTML rendering for live runtime inference profiles."""

from __future__ import annotations

from html import escape
from .render_runtime_svg import _runtime_profile_svg
```

Keep the moved function bodies unchanged except for importing
`_runtime_profile_svg` from `render_runtime_svg.py`.

Add this `__all__`:

```python
__all__ = [
    "_runtime_profile_html",
    "_runtime_request_status",
    "_seconds_to_ms",
]
```

- [ ] **Step 6: Replace `render.py` with a compatibility facade**

Replace `src/inferaxis/runtime/inference/profiling/render.py` with:

```python
"""Compatibility exports for profiling render helpers."""

from __future__ import annotations

from .render_async_trace import _async_buffer_trace_svg, _step_plot_points
from .render_common import _format_profile_value
from .render_runtime_html import (
    _runtime_profile_html,
    _runtime_request_status,
    _seconds_to_ms,
)
from .render_runtime_svg import (
    _runtime_action_channels,
    _runtime_action_trace_section,
    _runtime_chunk_action_channel_keys,
    _runtime_combined_step_trace,
    _runtime_profile_svg,
)

__all__ = [
    "_async_buffer_trace_svg",
    "_format_profile_value",
    "_runtime_action_channels",
    "_runtime_action_trace_section",
    "_runtime_chunk_action_channel_keys",
    "_runtime_combined_step_trace",
    "_runtime_profile_html",
    "_runtime_profile_svg",
    "_runtime_request_status",
    "_seconds_to_ms",
    "_step_plot_points",
]
```

- [ ] **Step 7: Run the architecture tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_internal_architecture -q
```

Expected: all tests in `tests.test_internal_architecture` pass.

- [ ] **Step 8: Run profiling-focused tests through the full suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: all tests pass with `OK`. Existing runtime warnings about RTC delay
are acceptable.

- [ ] **Step 9: Commit**

Run:

```bash
git add tests/test_internal_architecture.py src/inferaxis/runtime/inference/profiling/render.py src/inferaxis/runtime/inference/profiling/render_common.py src/inferaxis/runtime/inference/profiling/render_async_trace.py src/inferaxis/runtime/inference/profiling/render_runtime_svg.py src/inferaxis/runtime/inference/profiling/render_runtime_html.py
git commit -m "refactor: split profiling render modules"
```

Expected: commit succeeds and includes the profiling compatibility test plus
profiling render files.

## Task 5: Split Runtime Tests by Domain

**Files:**
- Modify: `tests/helpers.py`
- Create: `tests/test_runtime_engine.py`
- Create: `tests/test_scheduler.py`
- Create: `tests/test_profiling.py`
- Delete: `tests/test_inference_runtime.py`

- [ ] **Step 1: Move shared helper functions and classes to `tests/helpers.py`**

Move these top-level definitions from `tests/test_inference_runtime.py` into
`tests/helpers.py`:

```python
arm_value
arm_action
gripper_value
arm_and_gripper_action
make_chunk_request
RuntimeRobot
RuntimePolicy
RtcLoggingChunkPolicy
RecordingRuntimePolicy
PlainRuntimeExecutor
SingleActionChunkPolicy
PlanningSource
DeterministicClock
make_profile_clock
```

Add the imports needed by those helpers to `tests/helpers.py`:

```python
import threading

from inferaxis.core.schema import ComponentSpec
```

`ComponentSpec` is already imported in the current file; keep one import line.
Keep each moved helper body unchanged.

- [ ] **Step 2: Create `tests/test_runtime_engine.py`**

Move runtime-level tests from `InferenceRuntimeTests` into a new
`RuntimeEngineTests(unittest.TestCase)` class. Include tests whose names start
with or directly cover:

```python
test_runtime_
test_sync_runtime_
test_async_runtime_
test_inference_runtime_public_exports_remain_stable
```

Import shared helpers from `helpers.py`:

```python
from helpers import (
    PlainRuntimeExecutor,
    RecordingRuntimePolicy,
    RuntimePolicy,
    RuntimeRobot,
    SingleActionChunkPolicy,
    arm_action,
    arm_value,
    assert_array_equal,
    demo_image,
    make_profile_clock,
)
```

Keep each test method body unchanged except for imports.

- [ ] **Step 3: Create `tests/test_scheduler.py`**

Move scheduler-level tests into a new `SchedulerTests(unittest.TestCase)`
class. Include tests whose names start with or directly cover:

```python
test_chunk_scheduler_
test_async_scheduler_
test_sync_overlap_runtime_enable_rtc_discards_warmup_chunk
```

Import scheduler helpers from `helpers.py`:

```python
from helpers import (
    RtcLoggingChunkPolicy,
    arm_action,
    arm_and_gripper_action,
    arm_value,
    assert_array_equal,
    gripper_value,
    make_chunk_request,
)
```

Keep each test method body unchanged except for imports.

- [ ] **Step 4: Create `tests/test_profiling.py`**

Move profiling-level tests into a new `ProfilingTests(unittest.TestCase)`
class. Include tests whose names start with or directly cover:

```python
test_profile_
test_recommend_
test_async_buffer_trace_
test_profiling_package_reexports_existing_trace_helpers
test_async_runtime_profile_
```

Import profiling helpers from `helpers.py`:

```python
from helpers import (
    DeterministicClock,
    PlainRuntimeExecutor,
    RuntimePolicy,
    arm_action,
    arm_value,
    assert_array_equal,
    make_profile_clock,
)
```

Keep each test method body unchanged except for imports.

- [ ] **Step 5: Remove `tests/test_inference_runtime.py` after all methods move**

Delete `tests/test_inference_runtime.py` only after every test method from the
old `InferenceRuntimeTests` class exists in exactly one new test file.

Before deleting, count test method names:

```bash
rg "    def test_" tests/test_inference_runtime.py | wc -l
rg "    def test_" tests/test_runtime_engine.py tests/test_scheduler.py tests/test_profiling.py | wc -l
```

Expected: both counts match.

- [ ] **Step 6: Run unittest discovery**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: all tests pass. The total test count should be at least `146`
because `tests/test_internal_architecture.py` added three tests.

- [ ] **Step 7: Run pytest discovery**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected: all tests pass. The total test count should match the unittest
discovery count.

- [ ] **Step 8: Commit**

Run:

```bash
git add tests/helpers.py tests/test_runtime_engine.py tests/test_scheduler.py tests/test_profiling.py
git rm tests/test_inference_runtime.py
git commit -m "test: split runtime tests by domain"
```

Expected: commit succeeds and includes only test files.

## Task 6: Run Full Verification and Fix Refactor Regressions

**Files:**
- Modify only files touched by Tasks 1 through 5 when verification exposes a
  regression.

- [ ] **Step 1: Run compile verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall -q src tests
```

Expected: command exits `0`.

- [ ] **Step 2: Run unittest verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
```

Expected: all tests pass. Existing runtime warnings about RTC delay are
acceptable.

- [ ] **Step 3: Run pytest verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected: all tests pass. Existing runtime warnings about RTC delay are
acceptable.

- [ ] **Step 4: Run ruff check**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Expected: `All checks passed!`.

- [ ] **Step 5: Run ruff format check**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests
```

Expected: command exits `0`. If it reports files that need formatting, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format src tests
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests
```

Expected after formatting: command exits `0`.

- [ ] **Step 6: Confirm public import compatibility manually**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
import inferaxis as infra
import inferaxis.core.schema as schema
import inferaxis.runtime.inference.profiling.render as render

assert schema.Frame is infra.Frame
assert schema.Action is infra.Action
assert schema.BuiltinCommandKind is infra.BuiltinCommandKind
assert callable(render._async_buffer_trace_svg)
assert callable(render._runtime_profile_html)
print("compatibility imports ok")
PY
```

Expected: `compatibility imports ok`.

- [ ] **Step 7: Commit verification-only fixes**

If Step 1 through Step 6 required any fixes after the previous task commits,
run:

```bash
git add src tests pyproject.toml uv.lock
git commit -m "fix: resolve refactor verification issues"
```

Expected: commit succeeds when there are fixes. When there are no fixes, skip
this commit and keep the tree unchanged.

## Task 7: Final Review Checklist

**Files:**
- No planned file changes.

- [ ] **Step 1: Inspect module sizes**

Run:

```bash
wc -l src/inferaxis/core/schema.py src/inferaxis/core/schema_models.py src/inferaxis/core/schema_validation.py src/inferaxis/core/command_kinds.py src/inferaxis/core/schema_compat.py src/inferaxis/runtime/inference/profiling/render.py src/inferaxis/runtime/inference/profiling/render_async_trace.py src/inferaxis/runtime/inference/profiling/render_runtime_svg.py src/inferaxis/runtime/inference/profiling/render_runtime_html.py
```

Expected: `schema.py` and `profiling/render.py` are small compatibility
facades, and implementation code lives in focused modules.

- [ ] **Step 2: Inspect public diff**

Run:

```bash
git diff --stat HEAD~6..HEAD
git diff HEAD~6..HEAD -- src/inferaxis/__init__.py src/inferaxis/core/__init__.py src/inferaxis/runtime/inference/__init__.py
```

Expected: root public exports are unchanged. `src/inferaxis/core/__init__.py`
and `src/inferaxis/runtime/inference/__init__.py` either remain unchanged or
only receive compatibility-preserving imports.

- [ ] **Step 3: Run final verification commands**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
UV_CACHE_DIR=/tmp/uv-cache uv run ruff format --check src tests
```

Expected: every command exits `0`.

- [ ] **Step 4: Report final status**

Summarize:

```text
Implemented internal architecture refactor.
Public API compatibility preserved.
Verification passed:
- unittest
- pytest
- ruff check
- ruff format --check
```

Do not claim completion unless every command in Step 3 passed in the current
session.
