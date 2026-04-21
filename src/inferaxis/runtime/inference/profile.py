"""Sync profiling helpers for choosing runtime modes and scheduling settings."""

from __future__ import annotations

from ._profile_logging import (
    _emit_profile_request_trace,
    _frame_request_summary,
    _summarize_array,
    _to_json_safe_summary,
)
from ._profile_models import (
    _ProfiledRequestSample,
    _RobustMeanEstimator,
    AsyncBufferTrace,
    AsyncBufferTraceRequest,
    AsyncBufferTraceStep,
    InferenceModeRecommendation,
    SyncInferenceProfile,
)
from ._profile_public import profile_sync_inference, recommend_inference_mode
from ._profile_render import _async_buffer_trace_svg, _step_plot_points
from ._profile_trace import (
    _PendingAsyncTraceRequest,
    _TraceActionSlot,
    _build_async_buffer_trace,
    _observed_latency_steps,
)

__all__ = [
    "InferenceModeRecommendation",
    "SyncInferenceProfile",
    "profile_sync_inference",
    "recommend_inference_mode",
]
