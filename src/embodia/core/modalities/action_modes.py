"""Action-mode modality helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..errors import InterfaceValidationError
from ..schema import Action
from ._common import resolve_string_mapping

ACTION_MODE_MAP_ATTR = "ACTION_MODE_MAP"


def get_mode_map(owner: object) -> Mapping[str, str]:
    """Resolve the action-mode remapping table for a class or instance."""

    return resolve_string_mapping(owner, ACTION_MODE_MAP_ATTR)


def ensure_supported(
    action: Action,
    supported_modes: Sequence[str],
    *,
    owner_label: str,
    owner_name: str | None = None,
) -> None:
    """Ensure an action mode is supported by the target runtime."""

    if action.mode in supported_modes:
        return

    detail = "" if owner_name is None else f" {owner_name!r}"
    raise InterfaceValidationError(
        f"action mode {action.mode!r} is not supported by {owner_label}{detail}; "
        f"supported modes: {list(supported_modes)!r}."
    )


def ensure_model_output(
    action: Action,
    *,
    output_mode: str,
    model_name: str,
) -> None:
    """Ensure a model-produced action matches the declared output mode."""

    if action.mode == output_mode:
        return

    raise InterfaceValidationError(
        f"model {model_name!r} declared output action mode {output_mode!r}, "
        f"but produced {action.mode!r}."
    )


def pair_problem(
    *,
    supported_modes: Sequence[str],
    output_mode: str,
) -> str | None:
    """Return a robot/model action-mode compatibility problem, if any."""

    if output_mode in supported_modes:
        return None

    return (
        "action mode mismatch: "
        f"model outputs {output_mode!r}, "
        f"but robot supports {list(supported_modes)!r}."
    )


__all__ = ["ensure_model_output", "ensure_supported", "get_mode_map", "pair_problem"]
