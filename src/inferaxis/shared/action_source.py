"""Private helpers for resolving callable action sources in runtime flow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame, validate_action
from .coerce import as_action_fast

FrameSource = Callable[[], Frame | Mapping[str, Any]]
ActionSource = Callable[[Frame, object], object]
ActionSink = Callable[[Action], object]


def callable_owner(callback: object | None) -> object | None:
    """Return one stable owner object for a callable when possible."""

    if callback is None or not callable(callback):
        return None
    bound_self = getattr(callback, "__self__", None)
    if bound_self is not None:
        return bound_self
    return callback


def callable_key(callback: object | None) -> object | None:
    """Return one stable identity key for a callable across bound-method lookups."""

    if callback is None or not callable(callback):
        return None
    bound_self = getattr(callback, "__self__", None)
    bound_func = getattr(callback, "__func__", None)
    if bound_self is not None and bound_func is not None:
        return (id(bound_self), id(bound_func))
    return id(callback)


def resolve_runtime_owner(*callbacks: object | None) -> object | None:
    """Return the best owner object to associate with runtime metadata."""

    for callback in callbacks:
        owner = callable_owner(callback)
        if owner is not None:
            return owner
    return None


def call_action_fn(
    action_source: Callable[[Frame], Action | None],
    frame: Frame,
) -> Action:
    """Run one resolved single-step source with clear runtime errors."""

    try:
        raw_action = action_source(frame)
    except Exception as exc:
        raise InterfaceValidationError(
            f"resolved action source raised {type(exc).__name__}: {exc}"
        ) from exc

    if raw_action is None:
        raise InterfaceValidationError(
            "resolved action source must return an action-like value, got None."
        )
    try:
        action = as_action_fast(raw_action)
    except TypeError as exc:
        raise InterfaceValidationError(
            "resolved action source must return an action-like value, got "
            f"{type(raw_action).__name__}."
        ) from exc
    validate_action(action)
    return action


def _first_action_and_plan_length_from_plan(
    raw_plan: object,
    *,
    caller: str,
    empty_message: str,
) -> tuple[Action, int]:
    """Normalize one action/chunk return value and report its length."""

    try:
        action = as_action_fast(raw_plan)
    except TypeError:
        action = None
    else:
        validate_action(action)
        return action, 1

    if isinstance(raw_plan, (str, bytes)) or not isinstance(raw_plan, Sequence):
        raise InterfaceValidationError(
            f"{caller} must return an action-like value or a non-empty sequence "
            f"of action-like values, got {type(raw_plan).__name__}."
        )
    if not raw_plan:
        raise InterfaceValidationError(empty_message)

    first = raw_plan[0]
    try:
        action = as_action_fast(first)
    except TypeError as exc:
        raise InterfaceValidationError(
            f"{caller} must return only action-like items, got {type(first).__name__}."
        ) from exc
    validate_action(action)
    return action, len(raw_plan)


def first_action_and_plan_length_from_action_call(
    act_src_fn: ActionSource,
    frame: Frame,
    *,
    request: object | None = None,
) -> tuple[Action, int]:
    """Call one action source and return the first action plus plan length."""

    from ..runtime.inference.contracts import ChunkRequest

    if request is None:
        request = ChunkRequest(
            request_step=0,
            request_time_s=0.0,
            active_chunk_length=0,
            remaining_steps=0,
            latency_steps=0,
        )
    try:
        raw_plan = act_src_fn(frame, request)
    except Exception as exc:
        raise InterfaceValidationError(
            f"act_src_fn(frame, request) raised {type(exc).__name__}: {exc}"
        ) from exc

    return _first_action_and_plan_length_from_plan(
        raw_plan,
        caller="act_src_fn(frame, request)",
        empty_message="act_src_fn(frame, request) must not return an empty chunk.",
    )


def first_action_from_action_call(
    act_src_fn: ActionSource,
    frame: Frame,
) -> Action:
    """Call one action source and return the first action it provides."""

    return first_action_and_plan_length_from_action_call(act_src_fn, frame)[0]


def resolve_action_source(
    *,
    act_src_fn: ActionSource | None = None,
) -> Callable[[Frame], Action]:
    """Resolve one explicit callable source into a single-step action function."""

    if act_src_fn is not None:
        return lambda frame, _act_src_fn=act_src_fn: first_action_from_action_call(
            _act_src_fn,
            frame,
        )
    raise InterfaceValidationError("run_step() requires act_src_fn=....")


__all__ = [
    "ActionSink",
    "ActionSource",
    "FrameSource",
    "call_action_fn",
    "callable_key",
    "callable_owner",
    "first_action_and_plan_length_from_action_call",
    "first_action_from_action_call",
    "resolve_action_source",
    "resolve_runtime_owner",
]
