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
ROBOT_REMOTE_ACTION_METHODS: tuple[str, ...] = (
    "embodia_request_remote_policy_action",
    "_request_remote_policy_action",
    "request_remote_policy_action",
)
ROBOT_HAS_REMOTE_POLICY_METHODS: tuple[str, ...] = (
    "embodia_has_remote_policy",
    "has_remote_policy",
)

MODEL_GET_SPEC_METHODS: tuple[str, ...] = ("embodia_get_spec", "get_spec")
MODEL_RESET_METHODS: tuple[str, ...] = ("embodia_reset", "reset")
MODEL_INFER_METHODS: tuple[str, ...] = ("embodia_infer", "infer", "step")
MODEL_INFER_CHUNK_METHODS: tuple[str, ...] = (
    "embodia_infer_chunk",
    "infer_chunk",
    "step_chunk",
)
MODEL_PLAN_METHODS: tuple[str, ...] = ("embodia_plan", "plan")


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
    "MODEL_GET_SPEC_METHODS",
    "MODEL_INFER_CHUNK_METHODS",
    "MODEL_INFER_METHODS",
    "MODEL_PLAN_METHODS",
    "MODEL_RESET_METHODS",
    "ROBOT_ACT_METHODS",
    "ROBOT_GET_SPEC_METHODS",
    "ROBOT_HAS_REMOTE_POLICY_METHODS",
    "ROBOT_OBSERVE_METHODS",
    "ROBOT_REMOTE_ACTION_METHODS",
    "ROBOT_RESET_METHODS",
    "format_method_options",
    "resolve_callable_method",
]
