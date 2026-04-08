"""Sync profiling helpers for choosing chunk-scheduling settings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from os import PathLike
from pathlib import Path
import time
from typing import Any

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ..shared.action_source import (
    ActionSource,
    call_action_fn as _call_action_fn,
    resolve_action_source as _resolve_action_source,
)
from ..shared.dispatch import (
    POLICY_INFER_CHUNK_METHODS,
    POLICY_INFER_METHODS,
    POLICY_RESET_METHODS,
    ROBOT_ACT_METHODS,
    ROBOT_OBSERVE_METHODS,
    format_method_options,
    resolve_callable_method,
)
from ..checks import validate_action, validate_frame
from .chunk_scheduler import _RobustMeanEstimator
from .common import as_action, as_frame


@dataclass(slots=True)
class SyncInferenceProfile:
    """Serializable summary of one sync-inference profiling run."""

    steps: int
    execute_action: bool
    startup_ignored_inference_samples: int
    estimated_step_time_s: float | None
    estimated_inference_time_s: float | None
    estimated_control_hz: float | None
    estimated_latency_steps: int | None
    chunk_steps: int | None = None
    overlap_ratio: float | None = None
    suggested_overlap_steps: int | None = None
    suggested_request_trigger_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Export the profile into a plain JSON-serializable dictionary."""

        return {
            "steps": self.steps,
            "execute_action": self.execute_action,
            "startup_ignored_inference_samples": self.startup_ignored_inference_samples,
            "estimated_step_time_s": self.estimated_step_time_s,
            "estimated_inference_time_s": self.estimated_inference_time_s,
            "estimated_control_hz": self.estimated_control_hz,
            "estimated_latency_steps": self.estimated_latency_steps,
            "chunk_steps": self.chunk_steps,
            "overlap_ratio": self.overlap_ratio,
            "suggested_overlap_steps": self.suggested_overlap_steps,
            "suggested_request_trigger_steps": self.suggested_request_trigger_steps,
        }

    def write_json(self, path: str | PathLike[str]) -> None:
        """Write the profile summary to one JSON file."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def profile_sync_inference(
    robot: object,
    policy: object | None = None,
    *,
    action_fn: ActionSource | None = None,
    steps: int = 8,
    execute_action: bool = False,
    reset_policy: bool = True,
    startup_ignore_inference_samples: int = 1,
    timing_window_size: int = 64,
    timing_trim_ratio: float = 0.1,
    chunk_steps: int | None = None,
    overlap_ratio: float | None = None,
    safety_margin_steps: int = 1,
    output_path: str | PathLike[str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SyncInferenceProfile:
    """Profile sync inference and optionally write one JSON summary.

    The report intentionally focuses on software-side timing:

    - ``estimated_step_time_s``: one robust mean of sync step runtime
    - ``estimated_inference_time_s``: one robust mean of policy/policy latency

    The first inference sample is ignored by default so policy startup cost
    does not skew the estimate.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise InterfaceValidationError(f"steps must be a positive int, got {steps!r}.")
    if isinstance(execute_action, bool) is False:
        raise InterfaceValidationError("execute_action must be a bool.")
    if isinstance(reset_policy, bool) is False:
        raise InterfaceValidationError("reset_policy must be a bool.")
    if isinstance(safety_margin_steps, bool) or not isinstance(
        safety_margin_steps,
        int,
    ):
        raise InterfaceValidationError("safety_margin_steps must be an int.")
    if safety_margin_steps < 0:
        raise InterfaceValidationError(
            f"safety_margin_steps must be >= 0, got {safety_margin_steps!r}."
        )

    if chunk_steps is not None:
        if isinstance(chunk_steps, bool) or not isinstance(chunk_steps, int):
            raise InterfaceValidationError("chunk_steps must be an int when provided.")
        if chunk_steps <= 0:
            raise InterfaceValidationError(
                f"chunk_steps must be > 0 when provided, got {chunk_steps!r}."
            )
    if overlap_ratio is not None:
        if isinstance(overlap_ratio, bool) or not isinstance(overlap_ratio, (int, float)):
            raise InterfaceValidationError(
                "overlap_ratio must be a real number in (0, 1) when provided."
            )
        overlap_ratio = float(overlap_ratio)
        if not 0.0 < overlap_ratio < 1.0:
            raise InterfaceValidationError(
                f"overlap_ratio must be in (0, 1), got {overlap_ratio!r}."
            )

    action_source, can_reset = _resolve_action_source(policy, action_fn, robot=robot)
    if reset_policy and not can_reset:
        raise InterfaceValidationError(
            "reset_policy=True requires a source object exposing "
            f"{format_method_options(POLICY_RESET_METHODS)} together with "
            f"{format_method_options(POLICY_INFER_METHODS)} or "
            f"{format_method_options(POLICY_INFER_CHUNK_METHODS)}, not a bare callable."
        )
    if reset_policy and policy is not None:
        reset, reset_name = resolve_callable_method(policy, POLICY_RESET_METHODS)
        if not callable(reset) or reset_name is None:
            raise InterfaceValidationError(
                "reset_policy=True requires a policy object exposing "
                f"{format_method_options(POLICY_RESET_METHODS)}."
            )
        try:
            reset()
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(policy).__name__}.{reset_name}() raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    step_estimator = _RobustMeanEstimator(
        window_size=timing_window_size,
        startup_ignore_samples=0,
        trim_ratio=timing_trim_ratio,
    )
    inference_estimator = _RobustMeanEstimator(
        window_size=timing_window_size,
        startup_ignore_samples=startup_ignore_inference_samples,
        trim_ratio=timing_trim_ratio,
    )

    for _ in range(steps):
        step_start = float(clock())
        observe, observe_name = resolve_callable_method(robot, ROBOT_OBSERVE_METHODS)
        if not callable(observe) or observe_name is None:
            raise InterfaceValidationError(
                f"{type(robot).__name__} must expose "
                f"{format_method_options(ROBOT_OBSERVE_METHODS)}."
            )
        try:
            raw_frame = observe()
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(robot).__name__}.{observe_name}() raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        frame = as_frame(raw_frame)
        validate_frame(frame)

        inference_start = float(clock())
        raw_action = _call_action_fn(action_source, frame)
        inference_end = float(clock())

        action = as_action(raw_action)
        validate_action(action)
        if execute_action:
            act, act_name = resolve_callable_method(robot, ROBOT_ACT_METHODS)
            if not callable(act) or act_name is None:
                raise InterfaceValidationError(
                    f"{type(robot).__name__} must expose "
                    f"{format_method_options(ROBOT_ACT_METHODS)}."
                )
            try:
                act(action)
            except Exception as exc:
                raise InterfaceValidationError(
                    f"{type(robot).__name__}.{act_name}(action) raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        step_end = float(clock())

        step_estimator.add(max(step_end - step_start, 0.0))
        inference_estimator.add(max(inference_end - inference_start, 0.0))

    estimated_step_time_s = step_estimator.mean
    estimated_inference_time_s = inference_estimator.mean
    estimated_control_hz = None
    estimated_latency_steps = None
    suggested_overlap_steps = None
    suggested_request_trigger_steps = None

    if estimated_step_time_s is not None and estimated_step_time_s > 0.0:
        estimated_control_hz = 1.0 / estimated_step_time_s
        if estimated_inference_time_s is not None:
            estimated_latency_steps = (
                int((estimated_inference_time_s / estimated_step_time_s) // 1)
            )
            if estimated_inference_time_s % estimated_step_time_s > 0:
                estimated_latency_steps += 1
            estimated_latency_steps += safety_margin_steps

    if (
        chunk_steps is not None
        and overlap_ratio is not None
        and estimated_latency_steps is not None
    ):
        suggested_overlap_steps = min(
            max(int(chunk_steps * overlap_ratio + 0.999999999), 1),
            chunk_steps,
        )
        suggested_request_trigger_steps = (
            suggested_overlap_steps + estimated_latency_steps
        )

    profile = SyncInferenceProfile(
        steps=steps,
        execute_action=execute_action,
        startup_ignored_inference_samples=startup_ignore_inference_samples,
        estimated_step_time_s=estimated_step_time_s,
        estimated_inference_time_s=estimated_inference_time_s,
        estimated_control_hz=estimated_control_hz,
        estimated_latency_steps=estimated_latency_steps,
        chunk_steps=chunk_steps,
        overlap_ratio=overlap_ratio,
        suggested_overlap_steps=suggested_overlap_steps,
        suggested_request_trigger_steps=suggested_request_trigger_steps,
    )
    if output_path is not None:
        profile.write_json(output_path)
    return profile


__all__ = ["SyncInferenceProfile", "profile_sync_inference"]
