"""Private helpers for resolving policy action sources in runtime flow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from .coerce import as_action
from .check_utils import single_step_chunk_request
from .dispatch import (
    POLICY_INFER_CHUNK_METHODS,
    POLICY_INFER_METHODS,
    POLICY_RESET_METHODS,
    format_method_options,
    resolve_callable_method,
)
from ..checks import validate_action

ActionSource = Callable[[Frame], Action | Mapping[str, Any] | None]
SOURCE_STEP_METHODS: tuple[str, ...] = (
    *POLICY_INFER_METHODS,
    "inferaxis_next_action",
    "next_action",
    "get_action",
)


def coalesce_source_argument(
    source: object | None,
    policy: object | None,
) -> object | None:
    """Resolve the preferred ``source=...`` argument with ``policy=...`` fallback."""

    if source is not None and policy is not None:
        raise InterfaceValidationError(
            "run_step() accepts either source=... or policy=..., not both."
        )
    return source if source is not None else policy


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
            "run_step() accepts either source=... or action_fn=..., not both."
        )

    if source is None:
        if action_fn is None:
            raise InterfaceValidationError(
                "run_step() requires source=... or action_fn=... "
                "inferaxis keeps robot execution local; remote deployment belongs "
                "on the policy/source side."
            )
        return action_fn, False

    if robot is not None:
        bind_robot = getattr(source, "inferaxis_bind_robot", None)
        if callable(bind_robot):
            try:
                bind_robot(robot)
            except Exception as exc:
                raise InterfaceValidationError(
                    f"{type(source).__name__}.inferaxis_bind_robot(robot) raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    reset_method, _ = resolve_callable_method(source, POLICY_RESET_METHODS)
    can_reset = callable(reset_method)

    infer, _ = resolve_callable_method(source, SOURCE_STEP_METHODS)
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
        f"{format_method_options(SOURCE_STEP_METHODS)}, "
        f"{format_method_options(POLICY_INFER_CHUNK_METHODS)}, or be "
        f"callable(frame), got {type(source).__name__}."
    )


__all__ = [
    "ActionSource",
    "SOURCE_STEP_METHODS",
    "coalesce_source_argument",
    "call_action_fn",
    "first_action_from_chunk_call",
    "resolve_action_source",
]
