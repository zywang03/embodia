"""Configuration validation helpers for chunk scheduling."""

from __future__ import annotations

from typing import Any

from ....core.errors import InterfaceValidationError
from ..optimizers import _normalize_blend_weight
from ..validation import resolve_validation_mode


def _validate_configuration(self) -> None:
    """Validate public scheduler settings in place."""

    _validate_nonnegative_int(
        self.steps_before_request,
        field_name="steps_before_request",
    )
    _validate_optional_positive_int(
        self.execution_steps,
        field_name="execution_steps",
    )

    self.latency_ema_beta = _validate_real(
        self.latency_ema_beta,
        field_name="latency_ema_beta",
    )
    if not 0.0 < self.latency_ema_beta <= 1.0:
        raise InterfaceValidationError(
            "latency_ema_beta must be in the range (0, 1], got "
            f"{self.latency_ema_beta!r}."
        )

    self.initial_latency_steps = _validate_nonnegative_real(
        self.initial_latency_steps,
        field_name="initial_latency_steps",
    )
    if self.fixed_latency_steps is not None:
        self.fixed_latency_steps = _validate_nonnegative_real(
            self.fixed_latency_steps,
            field_name="fixed_latency_steps",
        )
    if self.control_period_s is not None:
        self.control_period_s = _validate_real(
            self.control_period_s,
            field_name="control_period_s",
        )
        if self.control_period_s <= 0.0:
            raise InterfaceValidationError(
                "control_period_s must be > 0 when provided, got "
                f"{self.control_period_s!r}."
            )

    for field_name in ("warmup_requests", "profile_delay_requests"):
        _validate_nonnegative_int(
            getattr(self, field_name),
            field_name=field_name,
        )

    _validate_nonnegative_int(
        self.interpolation_steps,
        field_name="interpolation_steps",
    )

    if self.max_chunk_size is not None:
        _validate_optional_positive_int(
            self.max_chunk_size,
            field_name="max_chunk_size",
        )

    if not isinstance(self.enable_rtc, bool):
        raise InterfaceValidationError("enable_rtc must be a bool.")
    if self.enable_rtc and self.execution_steps is None:
        raise InterfaceValidationError(
            "execution_steps must be provided when enable_rtc=True."
        )
    if self.slow_rtc_bootstrap not in {"warn", "error", "confirm"}:
        raise InterfaceValidationError(
            "ChunkScheduler.slow_rtc_bootstrap must be 'warn', 'error', or "
            f"'confirm', got {self.slow_rtc_bootstrap!r}."
        )
    if isinstance(self.latency_steps_offset, bool) or not isinstance(
        self.latency_steps_offset,
        int,
    ):
        raise InterfaceValidationError("latency_steps_offset must be an int.")

    resolved_validation, startup_validation_only = resolve_validation_mode(
        validation=self.validation,
        startup_validation_only=self.startup_validation_only,
        field_name="ChunkScheduler",
    )
    self.validation = str(resolved_validation)
    self.startup_validation_only = startup_validation_only

    self.overlap_current_weight = _normalize_blend_weight(
        self.overlap_current_weight,
        field_name="overlap_current_weight",
    )
    if hasattr(self, "_execution_cursor"):
        self._execution_cursor.interpolation_steps = self.interpolation_steps
    if hasattr(self, "_rtc_window_builder"):
        self._rtc_window_builder.enabled = self.enable_rtc
        self._rtc_window_builder.execution_steps = self.execution_steps
        self._rtc_window_builder.steps_before_request = self.steps_before_request
    self.refresh_latency_mode()


def refresh_latency_mode(self) -> None:
    """Recompute latency-estimate mode after runtime config changes."""

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


def _validated_latency_steps_offset(self) -> int:
    """Return the signed manual latency hint offset after validation."""

    if isinstance(self.latency_steps_offset, bool) or not isinstance(
        self.latency_steps_offset,
        int,
    ):
        raise InterfaceValidationError("latency_steps_offset must be an int.")
    return int(self.latency_steps_offset)


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
        raise InterfaceValidationError(f"{field_name} must be an int when provided.")
    if value <= 0:
        raise InterfaceValidationError(
            f"{field_name} must be > 0 when provided, got {value!r}."
        )


def _validate_real(value: Any, *, field_name: str) -> float:
    """Require one real-valued setting."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterfaceValidationError(f"{field_name} must be a real number.")
    return float(value)


def _validate_nonnegative_real(value: Any, *, field_name: str) -> float:
    """Require one real-valued setting to be non-negative."""

    validated = _validate_real(value, field_name=field_name)
    if validated < 0.0:
        raise InterfaceValidationError(f"{field_name} must be >= 0, got {validated!r}.")
    return validated
