"""State-key modality helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..errors import InterfaceValidationError
from ..schema import Frame
from ._common import STATE_KEYS, resolve_modality_mapping


def get_key_map(owner: object) -> Mapping[str, str]:
    """Resolve the state-key remapping table for a class or instance."""

    return resolve_modality_mapping(owner, STATE_KEYS)


def ensure_frame_keys(
    frame: Frame,
    required_keys: Sequence[str],
    *,
    owner_label: str,
    owner_name: str | None = None,
    context: str = "frame",
) -> None:
    """Ensure a frame contains the required state keys."""

    missing_keys = sorted(set(required_keys) - set(frame.state))
    if not missing_keys:
        return

    detail = "" if owner_name is None else f" for {owner_label} {owner_name!r}"
    raise InterfaceValidationError(
        f"{context} is missing {owner_label} state keys{detail}: {missing_keys!r}."
    )


def pair_problem(
    *,
    available_keys: Sequence[str],
    required_keys: Sequence[str],
) -> str | None:
    """Return a robot/policy state compatibility problem, if any."""

    missing_keys = sorted(set(required_keys) - set(available_keys))
    if not missing_keys:
        return None

    return (
        "missing state keys: "
        f"policy requires {missing_keys!r}, "
        f"but robot exposes {list(available_keys)!r}."
    )


__all__ = ["STATE_KEYS", "ensure_frame_keys", "get_key_map", "pair_problem"]
