"""Internal recorder for live async runtime profiling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
import time

from ...core.schema import Action
from .profiling.models import (
    RuntimeInferenceProfile,
    RuntimeProfileActionCommand,
    RuntimeProfileActionStep,
    RuntimeProfileChunkAction,
    RuntimeProfileRequest,
)


def resolve_live_profile_output_dir(
    output_dir: str | PathLike[str] | None,
) -> Path:
    """Return the directory used for live runtime profiling artifacts."""

    if output_dir is not None:
        return Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.cwd() / "profiles" / f"inference-runtime-{timestamp}"


@dataclass(slots=True)
class _LiveRuntimeRequestState:
    """Mutable per-request profiling state tracked during runtime execution."""

    request_index: int
    request_step: int
    launch_control_step: int
    launch_time_s: float
    latency_hint_raw_steps: int
    reply_time_s: float | None = None
    prepared_time_s: float | None = None
    accepted_time_s: float | None = None
    waited_control_steps: int | None = None
    stale_raw_steps: int | None = None
    returned_chunk_length: int | None = None
    accepted_chunk_length: int | None = None
    dropped_as_stale: bool = False
    error: str | None = None
    closed: bool = False

    def to_profile_request(self) -> RuntimeProfileRequest:
        """Convert the live mutable state into the exported model."""

        request_duration_s = None
        if self.reply_time_s is not None:
            request_duration_s = max(self.reply_time_s - self.launch_time_s, 0.0)

        prepare_duration_s = None
        if self.reply_time_s is not None and self.prepared_time_s is not None:
            prepare_duration_s = max(self.prepared_time_s - self.reply_time_s, 0.0)

        accept_delay_s = None
        if self.accepted_time_s is not None and self.prepared_time_s is not None:
            accept_delay_s = max(self.accepted_time_s - self.prepared_time_s, 0.0)

        usable_latency_s = None
        if self.accepted_time_s is not None:
            usable_latency_s = max(self.accepted_time_s - self.launch_time_s, 0.0)

        accepted = (
            self.accepted_time_s is not None
            and not self.dropped_as_stale
            and self.error is None
        )
        return RuntimeProfileRequest(
            request_index=self.request_index,
            request_step=self.request_step,
            launch_control_step=self.launch_control_step,
            launch_time_s=self.launch_time_s,
            reply_time_s=self.reply_time_s,
            accepted_time_s=self.accepted_time_s,
            request_duration_s=request_duration_s,
            prepare_duration_s=prepare_duration_s,
            accept_delay_s=accept_delay_s,
            usable_latency_s=usable_latency_s,
            latency_hint_raw_steps=self.latency_hint_raw_steps,
            waited_control_steps=self.waited_control_steps,
            stale_raw_steps=self.stale_raw_steps,
            returned_chunk_length=self.returned_chunk_length,
            accepted_chunk_length=self.accepted_chunk_length,
            accepted=accepted,
            dropped_as_stale=self.dropped_as_stale,
            error=self.error,
        )


class LiveRuntimeProfileRecorder:
    """Collect request-level async runtime events and flush them on close."""

    def __init__(self, *, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._next_request_index = 0
        self._records: list[_LiveRuntimeRequestState] = []
        self._record_by_index: dict[int, _LiveRuntimeRequestState] = {}
        self._action_steps: list[RuntimeProfileActionStep] = []
        self._chunk_actions: list[RuntimeProfileChunkAction] = []
        self._recorded_chunk_requests: set[int] = set()
        self._flushed = False

    def _commands_from_action(
        self,
        action: Action,
    ) -> list[RuntimeProfileActionCommand]:
        """Extract numeric command values from one action for profiling."""

        return [
            RuntimeProfileActionCommand(
                target=target,
                command=str(command.command),
                value=[
                    float(value)
                    for value in command.value.reshape(-1).tolist()
                ],
                ref_frame=command.ref_frame,
            )
            for target, command in action.commands.items()
        ]

    def record_launch(
        self,
        *,
        request_step: int,
        launch_control_step: int,
        launch_time_s: float,
        latency_hint_raw_steps: int,
    ) -> int:
        """Open one profiling record for a newly launched request."""

        if self._flushed:
            return self._next_request_index
        request_index = self._next_request_index
        self._next_request_index += 1
        record = _LiveRuntimeRequestState(
            request_index=request_index,
            request_step=request_step,
            launch_control_step=launch_control_step,
            launch_time_s=launch_time_s,
            latency_hint_raw_steps=latency_hint_raw_steps,
        )
        self._records.append(record)
        self._record_by_index[request_index] = record
        return request_index

    def record_reply(
        self,
        *,
        request_index: int,
        reply_time_s: float | None,
        prepared_time_s: float | None,
        returned_chunk_length: int | None,
    ) -> None:
        """Attach reply metadata once the prepared chunk is ready."""

        if self._flushed:
            return
        record = self._record_by_index.get(request_index)
        if record is None or record.closed:
            return
        record.reply_time_s = reply_time_s
        record.prepared_time_s = prepared_time_s
        record.returned_chunk_length = returned_chunk_length

    def record_accept(
        self,
        *,
        request_index: int,
        accepted_time_s: float | None,
        waited_control_steps: int,
        stale_raw_steps: int,
        accepted_chunk_length: int,
        dropped_as_stale: bool,
    ) -> None:
        """Close one request after the runtime accepts or drops its reply."""

        if self._flushed:
            return
        record = self._record_by_index.get(request_index)
        if record is None or record.closed:
            return
        record.accepted_time_s = accepted_time_s
        record.waited_control_steps = waited_control_steps
        record.stale_raw_steps = stale_raw_steps
        record.accepted_chunk_length = accepted_chunk_length
        record.dropped_as_stale = dropped_as_stale
        record.closed = True

    def record_completed_without_accept(
        self,
        *,
        request_index: int,
        request_step: int | None = None,
        actions: list[Action] | None = None,
    ) -> None:
        """Close one request that finished successfully but was not accepted."""

        if self._flushed:
            return
        record = self._record_by_index.get(request_index)
        if record is None or record.closed:
            return
        record.accepted_chunk_length = 0
        record.closed = True
        if request_step is not None and actions is not None:
            self.record_chunk_actions(
                request_index=request_index,
                request_step=request_step,
                actions=actions,
                stale_steps=0,
                accepted_length=0,
                status="unused",
            )

    def record_error(
        self,
        *,
        request_index: int,
        error: str,
        reply_time_s: float | None,
        prepared_time_s: float | None,
        returned_chunk_length: int | None,
    ) -> None:
        """Close one request after it fails inside request execution."""

        if self._flushed:
            return
        record = self._record_by_index.get(request_index)
        if record is None or record.closed:
            return
        record.reply_time_s = reply_time_s
        record.prepared_time_s = prepared_time_s
        record.returned_chunk_length = returned_chunk_length
        record.error = error
        record.closed = True

    def record_action(
        self,
        *,
        raw_action: Action,
        action: Action,
        plan_refreshed: bool,
        control_wait_s: float,
        buffer_size: int | None = None,
        execution_buffer_size: int | None = None,
    ) -> None:
        """Record one action emitted by the runtime loop."""

        if self._flushed:
            return
        self._action_steps.append(
            RuntimeProfileActionStep(
                step_index=len(self._action_steps),
                action_time_s=time.perf_counter(),
                plan_refreshed=bool(plan_refreshed),
                control_wait_s=float(control_wait_s),
                buffer_size=buffer_size,
                execution_buffer_size=execution_buffer_size,
                raw_commands=self._commands_from_action(raw_action),
                action_commands=self._commands_from_action(action),
            )
        )

    def record_chunk_actions(
        self,
        *,
        request_index: int,
        request_step: int,
        actions: list[Action],
        stale_steps: int,
        accepted_length: int,
        status: str = "integrated",
    ) -> None:
        """Record returned chunk actions with accepted/drop annotations."""

        if self._flushed or request_index in self._recorded_chunk_requests:
            return
        self._recorded_chunk_requests.add(request_index)
        accepted_start = max(stale_steps, 0)
        accepted_end = accepted_start + max(accepted_length, 0)
        for action_index, action in enumerate(actions):
            action_status = status
            if status == "integrated":
                if accepted_start <= action_index < accepted_end:
                    action_status = "accepted"
                else:
                    action_status = "dropped"
            self._chunk_actions.append(
                RuntimeProfileChunkAction(
                    request_index=request_index,
                    request_step=request_step,
                    action_index=action_index,
                    step_index=request_step + action_index,
                    status=action_status,
                    commands=self._commands_from_action(action),
                )
            )

    def flush(self, *, config_snapshot: dict[str, object]) -> None:
        """Write one JSON report plus interactive HTML visualization exactly once."""

        if self._flushed:
            return
        self._flushed = True
        for record in self._records:
            if record.closed:
                continue
            if record.error is None:
                record.error = "Request did not complete before runtime close."
            record.closed = True

        profile = RuntimeInferenceProfile(
            mode="async",
            config=dict(config_snapshot),
            requests=[record.to_profile_request() for record in self._records],
            action_steps=list(self._action_steps),
            chunk_actions=list(self._chunk_actions),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        profile.write_json(self.output_dir / "runtime_profile.json")
        profile.write_html(self.output_dir / "runtime_profile.html")


__all__ = [
    "LiveRuntimeProfileRecorder",
    "resolve_live_profile_output_dir",
]
