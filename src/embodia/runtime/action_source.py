"""Private helpers for resolving policy action sources in runtime flow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from .coerce import as_action
from .check_utils import single_step_chunk_request
from .dispatch import (
    POLICY_INFER_CHUNK_METHODS,
    POLICY_INFER_METHODS,
    POLICY_RESET_METHODS,
    ROBOT_HAS_REMOTE_POLICY_METHODS,
    ROBOT_REMOTE_ACTION_METHODS,
    format_method_options,
    resolve_callable_method,
)
from .checks import validate_action

ActionSource = Callable[[Frame], Action | Mapping[str, Any] | None]


def call_action_fn(action_fn: ActionSource, frame: Frame) -> Action | Mapping[str, Any]:
    """Run one external action source with clear runtime errors."""

    try:
        raw_action = action_fn(frame)
    except Exception as exc:
        raise InterfaceValidationError(
            f"action_fn(frame) raised {type(exc).__name__}: {exc}"
        ) from exc

    if raw_action is None:
        raise InterfaceValidationError(
            "action_fn(frame) must return an action-like value, got None."
        )
    return raw_action


def first_action_from_chunk_call(
    infer_chunk: Callable[[Frame, object], object],
    frame: Frame,
) -> Action:
    """Call one chunk-producing source and return the first action."""

    try:
        raw_plan = infer_chunk(frame, single_step_chunk_request())
    except Exception as exc:
        raise InterfaceValidationError(
            f"infer_chunk(frame, request) raised {type(exc).__name__}: {exc}"
        ) from exc

    if isinstance(raw_plan, (Action, Mapping)):
        action = as_action(raw_plan)
        validate_action(action)
        return action

    if not isinstance(raw_plan, list) or not raw_plan:
        raise InterfaceValidationError(
            "infer_chunk(frame, request) must return a non-empty action chunk."
        )

    action = as_action(raw_plan[0])
    validate_action(action)
    return action


def resolve_action_source(
    source: object | None,
    action_fn: ActionSource | None,
    *,
    robot: object | None = None,
) -> tuple[ActionSource, bool]:
    """Resolve one action source into a callable and whether it supports reset."""

    if source is not None and action_fn is not None:
        raise InterfaceValidationError(
            "run_step() accepts either a policy/callable source as the second "
            "argument or action_fn=..., not both."
        )

    if source is None:
        if action_fn is None:
            request_remote, _ = resolve_callable_method(
                robot,
                ROBOT_REMOTE_ACTION_METHODS,
            )
            has_remote, _ = resolve_callable_method(
                robot,
                ROBOT_HAS_REMOTE_POLICY_METHODS,
            )
            if callable(request_remote) and callable(has_remote):
                try:
                    enabled = bool(has_remote())
                except Exception as exc:
                    raise InterfaceValidationError(
                        f"{type(robot).__name__}.has_remote_policy() raised "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if enabled:
                    return request_remote, False
            raise InterfaceValidationError(
                "run_step() requires a policy-like source, action_fn=..., or a "
                "robot with configured remote policy."
            )
        return action_fn, False

    reset_method, _ = resolve_callable_method(source, POLICY_RESET_METHODS)
    can_reset = callable(reset_method)

    infer, _ = resolve_callable_method(source, POLICY_INFER_METHODS)
    if callable(infer):
        return infer, can_reset

    infer_chunk, _ = resolve_callable_method(source, POLICY_INFER_CHUNK_METHODS)
    if callable(infer_chunk):
        return (
            lambda frame, _infer_chunk=infer_chunk: first_action_from_chunk_call(
                _infer_chunk,
                frame,
            ),
            can_reset,
        )

    if callable(source):
        return source, False

    raise InterfaceValidationError(
        "run_step() source must expose "
        f"{format_method_options(POLICY_INFER_METHODS)}, "
        f"{format_method_options(POLICY_INFER_CHUNK_METHODS)}, or be "
        f"callable(frame), got {type(source).__name__}."
    )


__all__ = [
    "ActionSource",
    "call_action_fn",
    "first_action_from_chunk_call",
    "resolve_action_source",
]
