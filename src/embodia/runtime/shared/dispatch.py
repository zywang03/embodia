"""Internal runtime method dispatch helpers.

This module keeps embodia's internal call sites stable while allowing wrapped
user classes to keep their native method names. embodia-prefixed methods are
always preferred when present because they are the normalized, validated
wrappers exposed by embodia mixins.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


ROBOT_GET_SPEC_METHODS: tuple[str, ...] = ("embodia_get_spec", "get_spec")
ROBOT_OBSERVE_METHODS: tuple[str, ...] = ("embodia_observe", "observe")
ROBOT_ACT_METHODS: tuple[str, ...] = ("embodia_act", "act")
ROBOT_RESET_METHODS: tuple[str, ...] = ("embodia_reset", "reset")

POLICY_GET_SPEC_METHODS: tuple[str, ...] = ("embodia_get_spec", "get_spec")
POLICY_RESET_METHODS: tuple[str, ...] = ("embodia_reset", "reset")
POLICY_INFER_METHODS: tuple[str, ...] = ("embodia_infer", "infer", "step")
POLICY_INFER_CHUNK_METHODS: tuple[str, ...] = (
    "embodia_infer_chunk",
    "infer_chunk",
    "step_chunk",
)
POLICY_PLAN_METHODS: tuple[str, ...] = ("embodia_plan", "plan")


def resolve_callable_method(
    obj: object,
    method_names: Sequence[str],
) -> tuple[Any | None, str | None]:
    """Return the first callable method exposed by ``obj`` from ``method_names``."""

    for method_name in method_names:
        candidate = getattr(obj, method_name, None)
        if callable(candidate):
            return candidate, method_name
    return None, None


def format_method_options(method_names: Sequence[str]) -> str:
    """Render one short human-readable method choice list."""

    return " or ".join(f"{method_name}(...)" for method_name in method_names)


__all__ = [
    "POLICY_GET_SPEC_METHODS",
    "POLICY_INFER_CHUNK_METHODS",
    "POLICY_INFER_METHODS",
    "POLICY_PLAN_METHODS",
    "POLICY_RESET_METHODS",
    "ROBOT_ACT_METHODS",
    "ROBOT_GET_SPEC_METHODS",
    "ROBOT_OBSERVE_METHODS",
    "ROBOT_RESET_METHODS",
    "format_method_options",
    "resolve_callable_method",
]
