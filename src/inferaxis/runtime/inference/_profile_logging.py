"""Compact logging helpers for sync inference profiling."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

import numpy as np

from ...core.schema import Frame


def _summarize_array(
    value: np.ndarray,
    *,
    include_values: bool,
) -> Any:
    """Return one small JSON-safe summary for an ndarray payload."""

    if include_values and value.size <= 12 and value.ndim <= 2:
        return value.tolist()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _to_json_safe_summary(value: Any) -> Any:
    """Return one compact JSON-safe summary for request tracing."""

    if isinstance(value, np.ndarray):
        return _summarize_array(
            value,
            include_values=value.ndim <= 1,
        )
    if isinstance(value, dict):
        return {
            str(key): _to_json_safe_summary(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_safe_summary(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _frame_request_summary(frame: Frame) -> dict[str, Any]:
    """Build one compact request-trace summary for a frame."""

    summary: dict[str, Any] = {
        "images": {
            key: _summarize_array(value, include_values=False)
            for key, value in frame.images.items()
        },
        "state": {
            key: _summarize_array(value, include_values=True)
            for key, value in frame.state.items()
        },
    }
    if frame.task:
        summary["task"] = _to_json_safe_summary(frame.task)
    if frame.meta:
        summary["meta"] = _to_json_safe_summary(frame.meta)
    if frame.sequence_id is not None:
        summary["sequence_id"] = frame.sequence_id
    return summary


def _emit_profile_request_trace(
    *,
    request_log_fn: Callable[[str], None],
    request_index: int,
    ignored_inference_sample: bool,
    frame: Frame,
    returned_plan_length: int,
    inference_time_s: float,
    step_time_s: float,
) -> None:
    """Emit one human-readable profile trace line."""

    request_log_fn(
        "[profile_sync_inference] "
        f"request={request_index} "
        f"inference_sample={'ignored' if ignored_inference_sample else 'used'} "
        f"returned_chunk_steps={returned_plan_length} "
        f"inference_ms={inference_time_s * 1000.0:.3f} "
        f"step_ms={step_time_s * 1000.0:.3f} "
        f"obs={json.dumps(_frame_request_summary(frame), ensure_ascii=True, sort_keys=True)}"
    )
