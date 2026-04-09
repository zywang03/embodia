"""Meta-key modality helpers."""

from __future__ import annotations

from collections.abc import Mapping

from ._common import META_KEYS, resolve_modality_mapping


def get_key_map(owner: object) -> Mapping[str, str]:
    """Resolve the meta-key remapping table for a class or instance."""

    return resolve_modality_mapping(owner, META_KEYS)


__all__ = ["META_KEYS", "get_key_map"]
