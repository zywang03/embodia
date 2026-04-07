"""Helpers for loading embodia config from optional YAML files."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import importlib.util
from os import PathLike
from pathlib import Path
from typing import Any

from .errors import InterfaceValidationError


def is_yaml_available() -> bool:
    """Return whether optional YAML loading support is available."""

    return importlib.util.find_spec("yaml") is not None


def require_yaml() -> None:
    """Raise a clear error if optional YAML support is unavailable."""

    if not is_yaml_available():
        raise InterfaceValidationError(
            "YAML config loading requires the optional 'PyYAML' package. "
            "Install it with `pip install 'embodia[yaml]'`."
        )


def _import_yaml() -> Any:
    """Import PyYAML lazily."""

    require_yaml()
    try:
        return importlib.import_module("yaml")
    except ImportError as exc:
        raise InterfaceValidationError(
            "Failed to import the optional 'yaml' module. Install it with "
            "`pip install 'embodia[yaml]'`."
        ) from exc


def _copy_mapping(value: Mapping[object, object], *, field_name: str) -> dict[str, Any]:
    """Validate and copy one mapping with string keys."""

    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise InterfaceValidationError(
                f"{field_name} must use string keys, got {key!r}."
            )
        result[key] = item
    return result


def load_yaml_config(
    path: str | PathLike[str],
    *,
    section: str | None = None,
) -> dict[str, Any]:
    """Load one embodia config mapping from a YAML file."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InterfaceValidationError(
            f"failed to read YAML config at {config_path}: {exc}"
        ) from exc

    yaml = _import_yaml()
    try:
        loaded = yaml.safe_load(text)
    except Exception as exc:
        raise InterfaceValidationError(
            f"failed to parse YAML config at {config_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if loaded is None:
        raise InterfaceValidationError(f"YAML config at {config_path} is empty.")
    if not isinstance(loaded, Mapping):
        raise InterfaceValidationError(
            f"YAML config at {config_path} must contain a top-level mapping, got "
            f"{type(loaded).__name__}."
        )

    mapping = _copy_mapping(loaded, field_name=f"{config_path}")
    if section is None:
        return mapping

    section_value = mapping.get(section)
    if section_value is None:
        raise InterfaceValidationError(
            f"YAML config at {config_path} is missing section {section!r}."
        )
    if not isinstance(section_value, Mapping):
        raise InterfaceValidationError(
            f"YAML section {section!r} at {config_path} must be a mapping, got "
            f"{type(section_value).__name__}."
        )
    return _copy_mapping(
        section_value,
        field_name=f"{config_path}:{section}",
    )


def load_component_yaml_config(
    path: str | PathLike[str],
    *,
    component: str,
) -> dict[str, Any]:
    """Load config for one embodia component from a YAML file.

    This supports two shapes:

    1. a multi-component file with top-level sections such as ``robot`` and
       ``model``
    2. a direct single-component mapping containing fields such as
       ``robot_spec`` or ``model_spec``
    """

    loaded = load_yaml_config(path)
    component_value = loaded.get(component)
    if component_value is not None:
        if not isinstance(component_value, Mapping):
            raise InterfaceValidationError(
                f"YAML section {component!r} at {Path(path)} must be a mapping, "
                f"got {type(component_value).__name__}."
            )
        return _copy_mapping(component_value, field_name=f"{Path(path)}:{component}")

    known_components = {"robot", "model"}
    present_components = sorted(
        key
        for key, value in loaded.items()
        if key in known_components and isinstance(value, Mapping)
    )
    if present_components:
        raise InterfaceValidationError(
            f"YAML config at {Path(path)} contains top-level sections "
            f"{present_components!r}, but is missing section {component!r}."
        )

    return loaded


__all__ = [
    "is_yaml_available",
    "load_component_yaml_config",
    "load_yaml_config",
    "require_yaml",
]
