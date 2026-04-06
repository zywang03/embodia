"""Simple HDF5 persistence helpers for embodia collection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from ..core.transform import coerce_model_spec, coerce_robot_spec
from .collect import Episode, EpisodeStep, episode_to_dict

H5_FORMAT = "embodia_episode_v1"


def is_h5_available() -> bool:
    """Return ``True`` when the optional ``h5py`` package is importable."""

    return importlib.util.find_spec("h5py") is not None


def require_h5() -> None:
    """Raise a clear error if the optional ``h5py`` package is missing."""

    if not is_h5_available():
        raise InterfaceValidationError(
            "HDF5 export requires the optional 'h5py' package. "
            "Install it with `pip install h5py` or `pip install 'embodia[h5]'`."
        )


def _h5py() -> Any:
    """Import and return ``h5py`` lazily."""

    require_h5()
    import h5py  # type: ignore[import-not-found]

    return h5py


def _decode_h5_string(value: Any) -> str:
    """Decode a scalar or row value read from an HDF5 string dataset."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise InterfaceValidationError(
        f"expected an HDF5 string value, got {type(value).__name__}."
    )


def save_episode_h5(episode: Episode, path: str | Path) -> Path:
    """Write one episode to a small HDF5 file.

    The layout intentionally stays simple:

    - root attrs for format/version metadata
    - one JSON blob for robot spec
    - one JSON blob for model spec
    - one JSON blob for episode meta
    - one JSON string dataset containing each step

    This keeps the writer easy to read while still using a real HDF5 container.
    """

    if not isinstance(episode, Episode):
        raise InterfaceValidationError(
            f"episode must be an Episode instance, got {type(episode).__name__}."
        )

    h5py = _h5py()
    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise InterfaceValidationError(
            f"path must be a file path, got existing directory {destination!s}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    exported = episode_to_dict(episode)
    step_rows = [
        json.dumps(step_dict, ensure_ascii=True)
        for step_dict in exported["steps"]
    ]

    with h5py.File(destination, "w") as handle:
        handle.attrs["embodia_format"] = H5_FORMAT
        handle.attrs["step_count"] = len(episode.steps)
        handle.create_dataset(
            "robot_spec_json",
            data=json.dumps(exported["robot_spec"], ensure_ascii=True),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "model_spec_json",
            data=json.dumps(exported["model_spec"], ensure_ascii=True),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "episode_meta_json",
            data=json.dumps(exported["meta"], ensure_ascii=True),
            dtype=string_dtype,
        )
        handle.create_dataset("steps_json", data=step_rows, dtype=string_dtype)

    return destination


def load_episode_h5(path: str | Path) -> Episode:
    """Load an episode from an HDF5 file written by :func:`save_episode_h5`."""

    h5py = _h5py()
    source = Path(path)
    if not source.exists():
        raise InterfaceValidationError(f"h5 file does not exist: {source!s}.")

    with h5py.File(source, "r") as handle:
        format_name = handle.attrs.get("embodia_format")
        if format_name != H5_FORMAT:
            raise InterfaceValidationError(
                f"unsupported h5 format {format_name!r}; expected {H5_FORMAT!r}."
            )

        robot_spec_data = json.loads(_decode_h5_string(handle["robot_spec_json"][()]))
        model_spec_data = json.loads(_decode_h5_string(handle["model_spec_json"][()]))
        episode_meta = json.loads(_decode_h5_string(handle["episode_meta_json"][()]))
        step_rows = [
            json.loads(_decode_h5_string(row))
            for row in handle["steps_json"][()]
        ]

    steps: list[EpisodeStep] = []
    for row in step_rows:
        action_data = row["action"]
        steps.append(
            EpisodeStep(
                frame=Frame(**row["frame"]),
                action=None if action_data is None else Action(**action_data),
                meta=row["meta"],
            )
        )

    return Episode(
        robot_spec=coerce_robot_spec(robot_spec_data),
        model_spec=(
            None if model_spec_data is None else coerce_model_spec(model_spec_data)
        ),
        steps=steps,
        meta=episode_meta,
    )


__all__ = ["H5_FORMAT", "is_h5_available", "load_episode_h5", "require_h5", "save_episode_h5"]
