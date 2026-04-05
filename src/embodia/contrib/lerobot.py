"""Optional LeRobot-oriented bridge helpers.

This module intentionally stays outside embodia's main package exports.
The goal is to keep embodia centered on interface unification while still
making it easy to hand standardized episodes to a LeRobot-oriented pipeline.

These helpers do not try to replace ``LeRobotDataset``. They provide:

- a light optional dependency check
- conversion from :class:`embodia.runtime.collect.Episode` to LeRobot-oriented
  step records
- a JSONL writer for staging those records in downstream pipelines
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from ..core.errors import InterfaceValidationError
from ..runtime.collect import Episode, EpisodeStep
from ..core.transform import action_to_dict, frame_to_dict


def is_lerobot_available() -> bool:
    """Return ``True`` when the optional ``lerobot`` package is importable."""

    return importlib.util.find_spec("lerobot") is not None


def require_lerobot() -> None:
    """Raise a clear error if the optional ``lerobot`` package is missing."""

    if not is_lerobot_available():
        raise InterfaceValidationError(
            "LeRobot integration requires the optional 'lerobot' package. "
            "Install it with `pip install lerobot` or `pip install 'embodia[lerobot]'`."
        )


def step_to_lerobot_record(
    step: EpisodeStep,
    *,
    episode_index: int,
    frame_index: int,
    is_last: bool,
) -> dict[str, Any]:
    """Convert one embodia step into a LeRobot-oriented record.

    The output is intentionally conservative and keeps embodia's nested image,
    state, and action structure intact. This makes it a good staging format
    for project-specific LeRobot conversion scripts without forcing one rigid
    dataset layout on all embodia users.
    """

    if not isinstance(step, EpisodeStep):
        raise InterfaceValidationError(
            f"step must be an EpisodeStep instance, got {type(step).__name__}."
        )
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise InterfaceValidationError(
            f"episode_index must be an int, got {type(episode_index).__name__}."
        )
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise InterfaceValidationError(
            f"frame_index must be an int, got {type(frame_index).__name__}."
        )

    frame_dict = frame_to_dict(step.frame)
    action_dict = None if step.action is None else action_to_dict(step.action)
    record: dict[str, Any] = {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp": step.frame.timestamp_ns / 1_000_000_000.0,
        "timestamp_ns": step.frame.timestamp_ns,
        "next.done": bool(is_last),
        "observation.images": dict(frame_dict["images"]),
        "observation.state": dict(frame_dict["state"]),
        "task": frame_dict["task"],
        "action": action_dict,
        "embodia.meta": {
            "frame_meta": frame_dict["meta"],
            "step_meta": None if step.meta is None else dict(step.meta),
        },
    }
    return record


def episode_to_lerobot_records(
    episode: Episode,
    *,
    episode_index: int = 0,
) -> list[dict[str, Any]]:
    """Convert an embodia episode into LeRobot-oriented step records."""

    if not isinstance(episode, Episode):
        raise InterfaceValidationError(
            f"episode must be an Episode instance, got {type(episode).__name__}."
        )

    last_step_index = len(episode.steps) - 1
    return [
        step_to_lerobot_record(
            step,
            episode_index=episode_index,
            frame_index=frame_index,
            is_last=frame_index == last_step_index,
        )
        for frame_index, step in enumerate(episode.steps)
    ]


def write_lerobot_jsonl(
    episode: Episode,
    path: str | Path,
    *,
    episode_index: int = 0,
) -> Path:
    """Write LeRobot-oriented step records to a JSONL file."""

    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise InterfaceValidationError(
            f"path must be a file path, got existing directory {destination!s}."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    records = episode_to_lerobot_records(episode, episode_index=episode_index)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")
    return destination


__all__ = [
    "episode_to_lerobot_records",
    "is_lerobot_available",
    "require_lerobot",
    "step_to_lerobot_record",
    "write_lerobot_jsonl",
]
