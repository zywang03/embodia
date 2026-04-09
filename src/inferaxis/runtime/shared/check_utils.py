"""Private helper functions for runtime acceptance checks."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

from ...core.errors import InterfaceValidationError
from .dispatch import format_method_options, resolve_callable_method


def object_label(obj: object) -> str:
    """Return a helpful label for runtime error messages."""

    return f"{type(obj).__name__} instance"


def require_method(
    obj: object,
    method_names: tuple[str, ...],
) -> tuple[Any, str]:
    """Fetch the first required callable method from a priority list."""

    method, resolved_name = resolve_callable_method(obj, method_names)
    if callable(method) and resolved_name is not None:
        return method, resolved_name

    raise InterfaceValidationError(
        f"{object_label(obj)} is missing required method "
        f"{format_method_options(method_names)}."
    )


def call_method(method: Any, obj: object, method_name: str, *args: object) -> Any:
    """Call a checked method and wrap runtime errors consistently."""

    try:
        return method(*args)
    except Exception as exc:
        raise InterfaceValidationError(
            f"{object_label(obj)} {method_name}() raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def ensure_signature_accepts(method: Any, method_name: str, *args: object) -> None:
    """Check that a method can be called with the expected runtime arguments."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        raise InterfaceValidationError(
            f"Could not inspect signature of method {method_name!r}: {exc}."
        ) from exc

    try:
        signature.bind(*args)
    except TypeError as exc:
        raise InterfaceValidationError(
            f"Method {method_name!r} has incompatible signature {signature}; "
            f"it must accept {len(args)} runtime argument(s)."
        ) from exc


def single_step_chunk_request() -> object:
    """Build one minimal request object for chunk-based acceptance checks."""

    return SimpleNamespace(
        request_step=0,
        request_time_s=0.0,
        history_start=0,
        history_end=0,
        active_chunk_length=0,
        remaining_steps=0,
        overlap_steps=0,
        latency_steps=0,
        request_trigger_steps=0,
        plan_start_step=0,
        history_actions=[],
    )


__all__ = [
    "call_method",
    "ensure_signature_accepts",
    "object_label",
    "require_method",
    "single_step_chunk_request",
]
