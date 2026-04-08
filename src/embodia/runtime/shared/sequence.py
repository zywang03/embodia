"""Small helpers for embodia-managed frame sequence ids."""

from __future__ import annotations

from dataclasses import replace
from threading import Lock
from weakref import WeakKeyDictionary

from ...core.schema import Frame

_SEQUENCE_ATTR = "_embodia_next_frame_sequence_id"
_weak_counters: WeakKeyDictionary[object, int] = WeakKeyDictionary()
_fallback_counters: dict[int, int] = {}
_counter_lock = Lock()


def _get_owner_counter(owner: object) -> int | None:
    """Return one per-owner counter when stored as an attribute."""

    try:
        value = getattr(owner, _SEQUENCE_ATTR)
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _set_owner_counter(owner: object, value: int) -> bool:
    """Try to persist one per-owner counter as an attribute."""

    try:
        setattr(owner, _SEQUENCE_ATTR, value)
    except Exception:
        return False
    return True


def _next_sequence_id(owner: object, *, reset: bool) -> int:
    """Return the next framework-managed sequence id for one owner."""

    with _counter_lock:
        if reset:
            if _set_owner_counter(owner, 1):
                return 0
            try:
                _weak_counters[owner] = 1
                return 0
            except TypeError:
                owner_id = id(owner)
                _fallback_counters[owner_id] = 1
                return 0

        counter = _get_owner_counter(owner)
        if counter is not None:
            _set_owner_counter(owner, counter + 1)
            return counter

        try:
            counter = _weak_counters.get(owner, 0)
            _weak_counters[owner] = counter + 1
            return counter
        except TypeError:
            owner_id = id(owner)
            counter = _fallback_counters.get(owner_id, 0)
            _fallback_counters[owner_id] = counter + 1
            return counter


def ensure_frame_sequence_id(
    frame: Frame,
    *,
    owner: object | None,
    reset: bool = False,
) -> Frame:
    """Return ``frame`` with a framework-managed ``sequence_id`` when absent."""

    if frame.sequence_id is not None or owner is None:
        return frame
    return replace(
        frame,
        sequence_id=_next_sequence_id(owner, reset=reset),
    )


__all__ = ["ensure_frame_sequence_id"]
