"""Small helpers for inferaxis-managed frame sequence ids."""

from __future__ import annotations

from threading import Lock
import time
from weakref import WeakKeyDictionary

from ..core.schema import Frame

_SEQUENCE_ATTR = "_inferaxis_next_frame_sequence_id"
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


def attach_runtime_frame_metadata(
    frame: Frame,
    *,
    owner: object | None,
    reset: bool = False,
    copy_arrays: bool = True,
) -> Frame:
    """Return ``frame`` with inferaxis-managed timestamp and sequence metadata.

    ``copy_arrays`` is retained for compatibility with older callers. The
    metadata path no longer makes defensive array copies; it only chooses
    whether to run the lightweight Frame normalization path.
    """

    if copy_arrays:
        updated = Frame(
            images=frame.images,
            state=frame.state,
            task=dict(frame.task),
            meta=dict(frame.meta),
        )
    else:
        updated = object.__new__(Frame)
        updated.images = dict(frame.images)
        updated.state = dict(frame.state)
        updated.task = dict(frame.task)
        updated.meta = dict(frame.meta)
    updated.timestamp_ns = time.time_ns()
    if frame.sequence_id is not None:
        updated.sequence_id = frame.sequence_id
    else:
        updated.sequence_id = (
            _next_sequence_id(owner, reset=reset) if owner is not None else None
        )
    return updated


__all__ = ["attach_runtime_frame_metadata"]
