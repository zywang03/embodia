"""Serializable data models for live inference runtime profiling."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from os import PathLike
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeProfileRequest:
    """One recorded async request from live runtime profiling."""

    request_index: int
    request_step: int
    launch_control_step: int
    launch_time_s: float
    reply_time_s: float | None
    accepted_time_s: float | None
    request_duration_s: float | None
    prepare_duration_s: float | None
    accept_delay_s: float | None
    usable_latency_s: float | None
    latency_hint_raw_steps: int
    waited_control_steps: int | None
    stale_raw_steps: int | None
    returned_chunk_length: int | None
    accepted_chunk_length: int | None
    accepted: bool
    dropped_as_stale: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Export the live request record into one JSON-safe dictionary."""

        return {
            "request_index": self.request_index,
            "request_step": self.request_step,
            "launch_control_step": self.launch_control_step,
            "launch_time_s": self.launch_time_s,
            "reply_time_s": self.reply_time_s,
            "accepted_time_s": self.accepted_time_s,
            "request_duration_s": self.request_duration_s,
            "prepare_duration_s": self.prepare_duration_s,
            "accept_delay_s": self.accept_delay_s,
            "usable_latency_s": self.usable_latency_s,
            "latency_hint_raw_steps": self.latency_hint_raw_steps,
            "waited_control_steps": self.waited_control_steps,
            "stale_raw_steps": self.stale_raw_steps,
            "returned_chunk_length": self.returned_chunk_length,
            "accepted_chunk_length": self.accepted_chunk_length,
            "accepted": self.accepted,
            "dropped_as_stale": self.dropped_as_stale,
            "error": self.error,
        }


@dataclass(slots=True)
class RuntimeProfileActionCommand:
    """One command value recorded for an emitted runtime action."""

    target: str
    command: str
    value: list[float]
    ref_frame: str | None

    def to_dict(self) -> dict[str, Any]:
        """Export the recorded command into a JSON-safe dictionary."""

        return {
            "target": self.target,
            "command": self.command,
            "value": list(self.value),
            "ref_frame": self.ref_frame,
        }


@dataclass(slots=True)
class RuntimeProfileActionStep:
    """One emitted action sample from live runtime profiling."""

    step_index: int
    action_time_s: float
    plan_refreshed: bool
    control_wait_s: float
    buffer_size: int | None
    execution_buffer_size: int | None
    raw_commands: list[RuntimeProfileActionCommand]
    action_commands: list[RuntimeProfileActionCommand]

    def to_dict(self) -> dict[str, Any]:
        """Export the emitted action sample into a JSON-safe dictionary."""

        return {
            "step_index": self.step_index,
            "action_time_s": self.action_time_s,
            "plan_refreshed": self.plan_refreshed,
            "control_wait_s": self.control_wait_s,
            "buffer_size": self.buffer_size,
            "execution_buffer_size": self.execution_buffer_size,
            "raw_commands": [command.to_dict() for command in self.raw_commands],
            "action_commands": [command.to_dict() for command in self.action_commands],
        }


@dataclass(slots=True)
class RuntimeProfileChunkAction:
    """One action candidate from a returned chunk, annotated after integration."""

    request_index: int
    request_step: int
    action_index: int
    step_index: int
    status: str
    commands: list[RuntimeProfileActionCommand]

    def to_dict(self) -> dict[str, Any]:
        """Export the chunk action candidate into a JSON-safe dictionary."""

        return {
            "request_index": self.request_index,
            "request_step": self.request_step,
            "action_index": self.action_index,
            "step_index": self.step_index,
            "status": self.status,
            "commands": [command.to_dict() for command in self.commands],
        }


@dataclass(slots=True)
class RuntimeInferenceProfile:
    """Serializable live request-level profile captured from async runtime use."""

    mode: str
    config: dict[str, Any]
    requests: list[RuntimeProfileRequest]
    action_steps: list[RuntimeProfileActionStep] = field(default_factory=list)
    chunk_actions: list[RuntimeProfileChunkAction] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return one compact request-level summary for the full session."""

        request_durations = [
            request.request_duration_s
            for request in self.requests
            if request.request_duration_s is not None
        ]
        usable_latencies = [
            request.usable_latency_s
            for request in self.requests
            if request.usable_latency_s is not None
        ]
        returned_chunk_lengths = [
            float(request.returned_chunk_length)
            for request in self.requests
            if request.returned_chunk_length is not None
        ]
        return {
            "total_requests": len(self.requests),
            "accepted_requests": sum(
                1 for request in self.requests if request.accepted
            ),
            "dropped_stale_requests": sum(
                1 for request in self.requests if request.dropped_as_stale
            ),
            "failed_requests": sum(
                1 for request in self.requests if request.error is not None
            ),
            "average_request_duration_s": (
                None
                if not request_durations
                else sum(request_durations) / len(request_durations)
            ),
            "average_usable_latency_s": (
                None
                if not usable_latencies
                else sum(usable_latencies) / len(usable_latencies)
            ),
            "max_usable_latency_s": (
                None if not usable_latencies else max(usable_latencies)
            ),
            "average_returned_chunk_length": (
                None
                if not returned_chunk_lengths
                else sum(returned_chunk_lengths) / len(returned_chunk_lengths)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export the live runtime profile into one JSON-safe dictionary."""

        return {
            "mode": self.mode,
            "config": dict(self.config),
            "summary": self.summary(),
            "requests": [request.to_dict() for request in self.requests],
            "action_steps": [step.to_dict() for step in self.action_steps],
            "chunk_actions": [action.to_dict() for action in self.chunk_actions],
        }

    def write_json(self, path: str | PathLike[str]) -> None:
        """Write the live runtime profile to one JSON file."""

        output_path = Path(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_html(self, path: str | PathLike[str]) -> None:
        """Write one interactive HTML chart for the live runtime profile."""

        from .render import _runtime_profile_html

        output_path = Path(path)
        output_path.write_text(
            _runtime_profile_html(self),
            encoding="utf-8",
        )
