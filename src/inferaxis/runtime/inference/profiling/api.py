"""Public profiling entrypoints."""

from __future__ import annotations

from collections.abc import Callable
import math
from os import PathLike
import time

from ....core.errors import InterfaceValidationError
from ...flow import _execute_step_action, _resolve_step_frame
from ....shared.action_source import (
    ActionSink,
    ActionSource,
    FrameSource,
    first_action_and_plan_length_from_action_call,
    resolve_runtime_owner,
)
from ....shared.common import validate_positive_number
from ..engine import InferenceMode
from .logging import _emit_profile_request_trace
from .models import (
    _ProfiledRequestSample,
    _RobustMeanEstimator,
    InferenceModeRecommendation,
    SyncInferenceProfile,
)
from .simulation import _build_async_buffer_trace, _observed_latency_steps


def profile_sync_inference(
    *,
    observe_fn: FrameSource,
    target_hz: float,
    act_fn: ActionSink | None = None,
    act_src_fn: ActionSource | None = None,
    execute_action: bool = False,
    startup_ignore_inference_samples: int = 1,
    stable_inference_sample_count: int = 4,
    timing_window_size: int = 64,
    timing_trim_ratio: float = 0.1,
    steps_before_request: int = 0,
    safety_margin_steps: int = 1,
    output_path: str | PathLike[str] | None = None,
    request_log_fn: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SyncInferenceProfile:
    """Profile sync inference and optionally write one JSON summary."""

    target_hz = validate_positive_number(target_hz, "target_hz")
    target_period_s = 1.0 / target_hz

    _validate_profile_inputs(
        execute_action=execute_action,
        request_log_fn=request_log_fn,
        startup_ignore_inference_samples=startup_ignore_inference_samples,
        stable_inference_sample_count=stable_inference_sample_count,
        timing_window_size=timing_window_size,
        safety_margin_steps=safety_margin_steps,
        steps_before_request=steps_before_request,
    )

    if act_src_fn is None:
        raise InterfaceValidationError("profile_sync_inference() requires act_src_fn=....")

    total_requests = (
        startup_ignore_inference_samples + stable_inference_sample_count
    )

    step_estimator = _RobustMeanEstimator(
        window_size=timing_window_size,
        startup_ignore_samples=0,
        trim_ratio=timing_trim_ratio,
    )
    inference_estimator = _RobustMeanEstimator(
        window_size=stable_inference_sample_count,
        startup_ignore_samples=startup_ignore_inference_samples,
        trim_ratio=timing_trim_ratio,
    )
    chunk_step_estimator = _RobustMeanEstimator(
        window_size=stable_inference_sample_count,
        startup_ignore_samples=startup_ignore_inference_samples,
        trim_ratio=0.0,
    )
    request_samples: list[_ProfiledRequestSample] = []
    frame_owner = resolve_runtime_owner(observe_fn, act_src_fn)

    for request_index in range(total_requests):
        step_start = float(clock())
        frame = _resolve_step_frame(
            observe_fn,
            None,
            owner=frame_owner,
        )

        inference_start = float(clock())
        action, returned_plan_length = first_action_and_plan_length_from_action_call(
            act_src_fn,
            frame,
        )
        inference_end = float(clock())

        if execute_action:
            _execute_step_action(act_fn, action)
        step_end = float(clock())
        step_time_s = max(step_end - step_start, 0.0)
        inference_time_s = max(inference_end - inference_start, 0.0)
        ignored_inference_sample = request_index < startup_ignore_inference_samples
        observed_latency_steps = _observed_latency_steps(
            inference_time_s=inference_time_s,
            target_period_s=target_period_s,
        )

        step_estimator.add(step_time_s)
        inference_estimator.add(inference_time_s)
        chunk_step_estimator.add(float(returned_plan_length))
        request_samples.append(
            _ProfiledRequestSample(
                request_index=request_index,
                chunk_steps=returned_plan_length,
                inference_time_s=inference_time_s,
                ignored_inference_sample=ignored_inference_sample,
                observed_latency_steps=observed_latency_steps,
            )
        )
        if request_log_fn is not None:
            _emit_profile_request_trace(
                request_log_fn=request_log_fn,
                request_index=request_index,
                ignored_inference_sample=ignored_inference_sample,
                frame=frame,
                returned_plan_length=returned_plan_length,
                inference_time_s=inference_time_s,
                step_time_s=step_time_s,
            )

    estimated_step_time_s = step_estimator.mean
    estimated_inference_time_s = inference_estimator.mean
    estimated_control_hz = None
    estimated_latency_steps = None
    estimated_chunk_steps = chunk_step_estimator.mean
    estimated_max_buffered_hz = None

    if (
        estimated_inference_time_s is not None
        and estimated_inference_time_s > 0.0
        and estimated_chunk_steps is not None
        and estimated_chunk_steps > 0.0
    ):
        estimated_max_buffered_hz = estimated_chunk_steps / estimated_inference_time_s

    if estimated_inference_time_s is not None:
        estimated_latency_steps = max(
            math.ceil(estimated_inference_time_s / target_period_s)
            + safety_margin_steps,
            0,
        )

    if estimated_step_time_s is not None and estimated_step_time_s > 0.0:
        estimated_control_hz = 1.0 / estimated_step_time_s

    profile = SyncInferenceProfile(
        target_hz=target_hz,
        target_period_s=target_period_s,
        steps=total_requests,
        execute_action=execute_action,
        startup_ignored_inference_samples=startup_ignore_inference_samples,
        stable_inference_sample_count=stable_inference_sample_count,
        estimated_step_time_s=estimated_step_time_s,
        estimated_inference_time_s=estimated_inference_time_s,
        estimated_control_hz=estimated_control_hz,
        estimated_latency_steps=estimated_latency_steps,
        estimated_chunk_steps=estimated_chunk_steps,
        estimated_max_buffered_hz=estimated_max_buffered_hz,
        steps_before_request=steps_before_request,
        async_buffer_trace=_build_async_buffer_trace(
            request_samples=request_samples,
            target_hz=target_hz,
            steps_before_request=steps_before_request,
        ),
    )
    if output_path is not None:
        profile.write_json(output_path)
    return profile


def recommend_inference_mode(
    *,
    observe_fn: FrameSource,
    act_fn: ActionSink | None = None,
    act_src_fn: ActionSource | None = None,
    target_hz: float,
    execute_action: bool = False,
    startup_ignore_inference_samples: int = 1,
    stable_inference_sample_count: int = 4,
    timing_window_size: int = 64,
    timing_trim_ratio: float = 0.1,
    steps_before_request: int = 0,
    safety_margin_steps: int = 1,
    output_path: str | PathLike[str] | None = None,
    request_log_fn: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> InferenceModeRecommendation:
    """Profile sync inference and recommend one runtime mode for a target rate."""

    target_hz = validate_positive_number(target_hz, "target_hz")
    profile = profile_sync_inference(
        observe_fn=observe_fn,
        target_hz=target_hz,
        act_fn=act_fn,
        act_src_fn=act_src_fn,
        execute_action=execute_action,
        startup_ignore_inference_samples=startup_ignore_inference_samples,
        stable_inference_sample_count=stable_inference_sample_count,
        timing_window_size=timing_window_size,
        timing_trim_ratio=timing_trim_ratio,
        steps_before_request=steps_before_request,
        safety_margin_steps=safety_margin_steps,
        output_path=None,
        request_log_fn=request_log_fn,
        clock=clock,
    )
    async_supported = (
        profile.estimated_chunk_steps is not None
        and profile.estimated_chunk_steps > 1.0
    )

    estimated_step_time_s = profile.estimated_step_time_s
    target_period_s = profile.target_period_s
    sync_expected_to_meet_target = None
    if estimated_step_time_s is not None and estimated_step_time_s > 0.0:
        sync_expected_to_meet_target = estimated_step_time_s <= target_period_s

    if sync_expected_to_meet_target is True:
        recommended_mode = InferenceMode.SYNC
        reason = (
            f"Estimated sync step time {estimated_step_time_s:.6f}s fits within "
            f"the requested {target_period_s:.6f}s control period "
            f"({target_hz:.3f} Hz)."
        )
    elif sync_expected_to_meet_target is False and async_supported:
        recommended_mode = InferenceMode.ASYNC
        reason = (
            f"Estimated sync step time {estimated_step_time_s:.6f}s exceeds the "
            f"requested {target_period_s:.6f}s control period "
            f"({target_hz:.3f} Hz), and act_src_fn=... is "
            "available for async scheduling."
        )
        if profile.estimated_latency_steps is not None:
            reason += (
                f" Estimated inference latency is about "
                f"{profile.estimated_latency_steps} control steps."
            )
    elif sync_expected_to_meet_target is False:
        recommended_mode = InferenceMode.SYNC
        reason = (
            f"Estimated sync step time {estimated_step_time_s:.6f}s exceeds the "
            f"requested {target_period_s:.6f}s control period "
            f"({target_hz:.3f} Hz), but the profiled source does not return "
            "a chunk with more than one action, so async scheduling is unavailable."
        )
    elif async_supported:
        recommended_mode = InferenceMode.ASYNC
        reason = (
            "No stable sync step-time estimate was produced, but act_src_fn=... "
            "is available, so async scheduling is the safer "
            "choice when you need a target-hz recommendation."
        )
    else:
        recommended_mode = InferenceMode.SYNC
        reason = (
            "No stable sync step-time estimate was produced, and the profiled "
            "source did not demonstrate chunk output with more than one action."
        )

    recommendation = InferenceModeRecommendation(
        target_hz=target_hz,
        target_period_s=target_period_s,
        async_supported=async_supported,
        sync_expected_to_meet_target=sync_expected_to_meet_target,
        recommended_mode=recommended_mode,
        reason=reason,
        profile=profile,
    )
    if output_path is not None:
        recommendation.write_json(output_path)
    return recommendation


def _validate_profile_inputs(
    *,
    execute_action: bool,
    request_log_fn: Callable[[str], None] | None,
    startup_ignore_inference_samples: int,
    stable_inference_sample_count: int,
    timing_window_size: int,
    safety_margin_steps: int,
    steps_before_request: int,
) -> None:
    """Validate the public profiling entrypoint parameters."""

    if isinstance(execute_action, bool) is False:
        raise InterfaceValidationError("execute_action must be a bool.")
    if request_log_fn is not None and not callable(request_log_fn):
        raise InterfaceValidationError("request_log_fn must be callable when provided.")
    if isinstance(startup_ignore_inference_samples, bool) or not isinstance(
        startup_ignore_inference_samples,
        int,
    ):
        raise InterfaceValidationError(
            "startup_ignore_inference_samples must be an int."
        )
    if startup_ignore_inference_samples < 0:
        raise InterfaceValidationError(
            "startup_ignore_inference_samples must be >= 0, got "
            f"{startup_ignore_inference_samples!r}."
        )
    if isinstance(timing_window_size, bool) or not isinstance(timing_window_size, int):
        raise InterfaceValidationError("timing_window_size must be an int.")
    if timing_window_size <= 0:
        raise InterfaceValidationError(
            f"timing_window_size must be > 0, got {timing_window_size!r}."
        )
    if isinstance(safety_margin_steps, bool) or not isinstance(
        safety_margin_steps,
        int,
    ):
        raise InterfaceValidationError("safety_margin_steps must be an int.")
    if safety_margin_steps < 0:
        raise InterfaceValidationError(
            f"safety_margin_steps must be >= 0, got {safety_margin_steps!r}."
        )
    if isinstance(steps_before_request, bool) or not isinstance(
        steps_before_request,
        int,
    ):
        raise InterfaceValidationError("steps_before_request must be an int.")
    if steps_before_request < 0:
        raise InterfaceValidationError(
            "steps_before_request must be >= 0, got "
            f"{steps_before_request!r}."
        )
    if isinstance(stable_inference_sample_count, bool) or not isinstance(
        stable_inference_sample_count,
        int,
    ):
        raise InterfaceValidationError(
            "stable_inference_sample_count must be an int."
        )
    if stable_inference_sample_count <= 0:
        raise InterfaceValidationError(
            "stable_inference_sample_count must be > 0, got "
            f"{stable_inference_sample_count!r}."
        )
