"""Runtime configuration helpers for :mod:`inferaxis.runtime.inference.engine`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.errors import InterfaceValidationError
from ...shared.common import validate_positive_number
from .control import RealtimeController
from .optimizers import _normalize_blend_weight

if TYPE_CHECKING:
    from .engine import InferenceMode, InferenceRuntime


def validate_runtime_config(
    runtime: "InferenceRuntime",
    *,
    mode_enum: type["InferenceMode"],
) -> None:
    """Validate one public runtime configuration in place."""

    try:
        runtime.mode = mode_enum(str(runtime.mode))
    except ValueError as exc:
        raise InterfaceValidationError(
            "InferenceRuntime.mode must be InferenceMode.SYNC or "
            f"InferenceMode.ASYNC, got {runtime.mode!r}."
        ) from exc

    _validate_nonnegative_int(
        runtime.steps_before_request,
        field_name="InferenceRuntime.steps_before_request",
    )
    _validate_optional_positive_int(
        runtime.execution_steps,
        field_name="InferenceRuntime.execution_steps",
    )
    if runtime.enable_rtc and runtime.execution_steps is None:
        raise InterfaceValidationError(
            "InferenceRuntime(enable_rtc=True) requires execution_steps=... ."
        )

    for field_name in ("warmup_requests", "profile_delay_requests"):
        _validate_nonnegative_int(
            getattr(runtime, field_name),
            field_name=f"InferenceRuntime.{field_name}",
        )

    _validate_nonnegative_int(
        runtime.interpolation_steps,
        field_name="InferenceRuntime.interpolation_steps",
    )

    if runtime.ensemble_weight is not None:
        runtime.ensemble_weight = _normalize_blend_weight(
            runtime.ensemble_weight,
            field_name="ensemble_weight",
        )

    if runtime.control_hz is not None:
        runtime.control_hz = validate_positive_number(
            runtime.control_hz,
            "InferenceRuntime.control_hz",
        )
    runtime.realtime_controller = _build_realtime_controller(runtime.control_hz)

    if not isinstance(runtime.enable_rtc, bool):
        raise InterfaceValidationError("InferenceRuntime.enable_rtc must be a bool.")
    if isinstance(runtime.latency_steps_offset, bool) or not isinstance(
        runtime.latency_steps_offset,
        int,
    ):
        raise InterfaceValidationError(
            "InferenceRuntime.latency_steps_offset must be an int."
        )
    if not isinstance(runtime.startup_validation_only, bool):
        raise InterfaceValidationError(
            "InferenceRuntime.startup_validation_only must be a bool."
        )


def scheduler_initial_latency_steps(runtime: "InferenceRuntime") -> float:
    """Return the async startup latency prior for one runtime."""

    if runtime.mode != "async":
        return 0.0
    return 1.0


def scheduler_control_period_s(runtime: "InferenceRuntime") -> float | None:
    """Return the control period forwarded into chunk scheduling."""

    if runtime.mode != "async" or runtime.realtime_controller is None:
        return None
    return runtime.realtime_controller.period_s


def build_chunk_scheduler_kwargs(
    runtime: "InferenceRuntime",
    *,
    action_source: Any,
) -> dict[str, Any]:
    """Build constructor kwargs for one internal chunk scheduler."""

    return {
        "action_source": action_source,
        "steps_before_request": runtime.steps_before_request,
        "execution_steps": runtime.execution_steps,
        "initial_latency_steps": scheduler_initial_latency_steps(runtime),
        "control_period_s": scheduler_control_period_s(runtime),
        "warmup_requests": runtime.warmup_requests,
        "profile_delay_requests": runtime.profile_delay_requests,
        "interpolation_steps": runtime.interpolation_steps,
        "use_overlap_blend": runtime.ensemble_weight is not None,
        "overlap_current_weight": (
            runtime.ensemble_weight if runtime.ensemble_weight is not None else 0.5
        ),
        "enable_rtc": runtime.enable_rtc,
        "latency_steps_offset": runtime.latency_steps_offset,
        "startup_validation_only": runtime.startup_validation_only,
    }


def sync_chunk_scheduler_config(runtime: "InferenceRuntime", scheduler: Any) -> None:
    """Update one reused scheduler from the runtime's latest settings."""

    scheduler.steps_before_request = runtime.steps_before_request
    scheduler.execution_steps = runtime.execution_steps
    scheduler.use_overlap_blend = runtime.ensemble_weight is not None
    scheduler.overlap_current_weight = (
        runtime.ensemble_weight if runtime.ensemble_weight is not None else 0.5
    )
    scheduler.initial_latency_steps = scheduler_initial_latency_steps(runtime)
    scheduler.control_period_s = scheduler_control_period_s(runtime)
    scheduler.warmup_requests = runtime.warmup_requests
    scheduler.profile_delay_requests = runtime.profile_delay_requests
    scheduler.interpolation_steps = runtime.interpolation_steps
    scheduler.enable_rtc = runtime.enable_rtc
    scheduler.latency_steps_offset = runtime.latency_steps_offset
    scheduler.startup_validation_only = runtime.startup_validation_only


def should_replace_chunk_scheduler(
    runtime: "InferenceRuntime",
    scheduler: Any,
    *,
    scheduler_key: object | None,
    source_is_single_step: bool,
) -> bool:
    """Return whether the cached scheduler must be recreated."""

    return (
        runtime._chunk_scheduler_key != scheduler_key
        or scheduler.execution_steps != runtime.execution_steps
        or (runtime.mode != "async" and source_is_single_step)
    )


def _build_realtime_controller(control_hz: float | None) -> RealtimeController | None:
    """Build the optional realtime controller from the public hz setting."""

    if control_hz is None:
        return None
    return RealtimeController(hz=control_hz)


def _validate_nonnegative_int(value: Any, *, field_name: str) -> None:
    """Require one integer setting to be non-negative."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise InterfaceValidationError(f"{field_name} must be an int >= 0.")
    if value < 0:
        raise InterfaceValidationError(f"{field_name} must be >= 0, got {value!r}.")


def _validate_optional_positive_int(value: Any, *, field_name: str) -> None:
    """Require one optional integer setting to be strictly positive."""

    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise InterfaceValidationError(
            f"{field_name} must be an int > 0 when provided."
        )
    if value <= 0:
        raise InterfaceValidationError(
            f"{field_name} must be > 0 when provided, got {value!r}."
        )


def _validate_nonnegative_real(value: Any, *, field_name: str) -> float:
    """Require one real-valued setting to be non-negative."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterfaceValidationError(
            f"{field_name} must be a real number >= 0 when provided."
        )
    validated = float(value)
    if validated < 0.0:
        raise InterfaceValidationError(
            f"{field_name} must be >= 0, got {validated!r}."
        )
    return validated
