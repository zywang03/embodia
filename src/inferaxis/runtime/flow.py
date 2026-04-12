"""Helpers for the inference side of inferaxis's unified runtime data flow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from .checks import validate_action, validate_frame
from ..shared.action_source import (
    ActionSink,
    ActionSource,
    FrameSource,
    first_action_from_action_call,
    resolve_runtime_owner,
)
from ..shared.coerce import as_frame as _as_frame
from ..shared.coerce import maybe_as_action as _maybe_as_action
from ..shared.sequence import attach_runtime_frame_metadata


@dataclass(slots=True)
class StepResult:
    """Unified result of one inferaxis-controlled closed-loop step."""

    frame: Frame
    raw_action: Action
    action: Action
    plan_refreshed: bool = True
    control_wait_s: float = 0.0


def _resolve_step_frame(
    observe_fn: FrameSource | None,
    frame: Frame | Mapping[str, Any] | None,
    *,
    owner: object | None,
) -> Frame:
    """Normalize one input frame from either ``observe_fn`` or ``frame=...``."""

    if frame is None:
        if not callable(observe_fn):
            raise InterfaceValidationError(
                "run_step() requires either frame=... or observe_fn=...."
            )

        try:
            raw_frame = observe_fn()
        except Exception as exc:
            observe_owner = getattr(observe_fn, "__self__", None)
            observe_name = getattr(observe_fn, "__name__", None) or "observe_fn"
            caller = (
                f"{type(observe_owner).__name__}.{observe_name}()"
                if observe_owner is not None
                else f"{observe_name}()"
            )
            raise InterfaceValidationError(
                f"{caller} raised {type(exc).__name__}: {exc}"
            ) from exc
    else:
        raw_frame = frame

    normalized_frame = attach_runtime_frame_metadata(
        _as_frame(raw_frame),
        owner=owner,
    )
    validate_frame(normalized_frame)
    return normalized_frame


def _execute_step_action(
    act_fn: ActionSink | None,
    action: Action,
) -> Action:
    """Execute one action locally when a send callback is available."""

    if not callable(act_fn):
        raise InterfaceValidationError(
            "run_step() requires act_fn=... when execute_action=True. "
            "Use run_step(..., execute_action=False) for pure frame -> action "
            "inference."
        )

    try:
        raw_executed_action = act_fn(action)
    except Exception as exc:
        act_owner = getattr(act_fn, "__self__", None)
        act_name = getattr(act_fn, "__name__", None) or "act_fn"
        caller = (
            f"{type(act_owner).__name__}.{act_name}(action)"
            if act_owner is not None
            else f"{act_name}(action)"
        )
        raise InterfaceValidationError(
            f"{caller} raised {type(exc).__name__}: {exc}"
        ) from exc

    executed_action = _maybe_as_action(raw_executed_action)
    if executed_action is None:
        return action

    validate_action(executed_action)
    return executed_action


def run_step(
    *,
    observe_fn: FrameSource | None = None,
    act_fn: ActionSink | None = None,
    act_src_fn: ActionSource | None = None,
    frame: Frame | Mapping[str, Any] | None = None,
    execute_action: bool = True,
    runtime: object | None = None,
    pace_control: bool = True,
    metadata_owner: object | None = None,
) -> StepResult:
    """Run one normalized step and optionally execute the action locally.

    ``act_src_fn(frame, request)`` is the only public action-source boundary.
    Returning one action is treated the same as returning a chunk of size ``1``.
    """

    metadata_owner = (
        metadata_owner
        if metadata_owner is not None
        else resolve_runtime_owner(
            observe_fn,
            act_fn,
            act_src_fn,
        )
    )
    if runtime is not None:
        run_with_runtime = getattr(runtime, "_run_step_impl", None)
        if not callable(run_with_runtime):
            raise InterfaceValidationError(
                "runtime must expose _run_step_impl(...), for example an "
                "InferenceRuntime instance."
            )
        return run_with_runtime(
            observe_fn=observe_fn,
            act_fn=act_fn,
            act_src_fn=act_src_fn,
            frame=frame,
            execute_action=execute_action,
            pace_control=pace_control,
            metadata_owner=metadata_owner,
        )

    normalized_frame = _resolve_step_frame(
        observe_fn,
        frame,
        owner=metadata_owner,
    )
    if act_src_fn is None:
        raise InterfaceValidationError("run_step() requires act_src_fn=....")
    normalized_action = first_action_from_action_call(
        act_src_fn,
        normalized_frame,
    )

    final_action = normalized_action
    if execute_action:
        final_action = _execute_step_action(act_fn, normalized_action)

    return StepResult(
        frame=normalized_frame,
        raw_action=normalized_action,
        action=final_action,
        plan_refreshed=True,
        control_wait_s=0.0,
    )
