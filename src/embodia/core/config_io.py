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


def _copy_string_mapping(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = True,
) -> dict[str, str]:
    """Validate and copy one ``mapping[str, str]`` block."""

    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"{field_name} must be a mapping[str, str], got "
            f"{type(value).__name__}."
        )

    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise InterfaceValidationError(
                f"{field_name} must use non-empty string keys and values, got "
                f"{key!r} -> {item!r}."
            )
        if not key.strip() or not item.strip():
            raise InterfaceValidationError(
                f"{field_name} must use non-empty string keys and values."
            )
        result[key] = item

    if not allow_empty and not result:
        raise InterfaceValidationError(f"{field_name} must not be empty.")

    return result


def _ensure_non_empty_string(value: object, *, field_name: str) -> str:
    """Validate one non-empty string field."""

    if not isinstance(value, str) or not value.strip():
        raise InterfaceValidationError(f"{field_name} must be a non-empty string.")
    return value


def _invert_unique_mapping(
    mapping: Mapping[str, str],
    *,
    field_name: str,
) -> dict[str, str]:
    """Invert one ``standard -> native`` mapping into ``native -> standard``."""

    result: dict[str, str] = {}
    for standard_name, native_name in mapping.items():
        if native_name in result:
            raise InterfaceValidationError(
                f"{field_name} maps multiple embodia names to the same native "
                f"name {native_name!r}, which is ambiguous."
            )
        result[native_name] = standard_name
    return result


def _merge_string_mapping(
    destination: dict[str, str],
    source: Mapping[str, str],
    *,
    field_name: str,
) -> None:
    """Merge one ``mapping[str, str]`` while detecting ambiguous collisions."""

    for key, value in source.items():
        existing = destination.get(key)
        if existing is not None and existing != value:
            raise InterfaceValidationError(
                f"{field_name} maps {key!r} to both {existing!r} and {value!r}."
            )
        destination[key] = value


def _copy_named_mapping(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = True,
) -> dict[str, dict[str, Any]]:
    """Validate and copy one ``mapping[str, mapping]`` block."""

    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"{field_name} must be a mapping, got {type(value).__name__}."
        )

    result: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise InterfaceValidationError(
                f"{field_name} must use non-empty string keys, got {key!r}."
            )
        if not isinstance(item, Mapping):
            raise InterfaceValidationError(
                f"{field_name}[{key!r}] must be a mapping, got "
                f"{type(item).__name__}."
            )
        result[key] = _copy_mapping(item, field_name=f"{field_name}[{key!r}]")

    if not allow_empty and not result:
        raise InterfaceValidationError(f"{field_name} must not be empty.")

    return result


def _validate_interface_keys(
    interface: Mapping[str, Any],
    *,
    field_name: str,
    allowed_keys: set[str],
) -> None:
    """Ensure one YAML ``interface`` block only uses supported keys."""

    unknown = sorted(key for key in interface if key not in allowed_keys)
    if not unknown:
        return

    expected = ", ".join(repr(key) for key in sorted(allowed_keys))
    found = ", ".join(repr(key) for key in unknown)
    raise InterfaceValidationError(
        f"{field_name} contains unsupported field(s) {found}. "
        f"Expected only: {expected}."
    )


def _expand_robot_interface_config(
    interface: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    """Expand one compact robot-side YAML interface block."""

    _validate_interface_keys(
        interface,
        field_name=field_name,
        allowed_keys={"name", "images", "task", "meta", "groups"},
    )

    name = _ensure_non_empty_string(interface.get("name"), field_name=f"{field_name}.name")
    images = _copy_string_mapping(
        interface.get("images", {}),
        field_name=f"{field_name}.images",
    )
    task = _copy_string_mapping(
        interface.get("task", {}),
        field_name=f"{field_name}.task",
    )
    meta = _copy_string_mapping(
        interface.get("meta", {}),
        field_name=f"{field_name}.meta",
    )

    groups = _copy_named_mapping(
        interface.get("groups"),
        field_name=f"{field_name}.groups",
        allow_empty=False,
    )
    robot_groups: list[dict[str, Any]] = []
    state_maps: dict[str, str] = {}
    action_mode_maps: dict[str, str] = {}
    control_target_maps: dict[str, str] = {}

    for standard_name, group_config in groups.items():
        _validate_interface_keys(
            group_config,
            field_name=f"{field_name}.groups[{standard_name!r}]",
            allowed_keys={
                "kind",
                "dof",
                "state",
                "action_modes",
                "meta",
                "native_name",
            },
        )
        kind = _ensure_non_empty_string(
            group_config.get("kind"),
            field_name=f"{field_name}.groups[{standard_name!r}].kind",
        )
        dof = group_config.get("dof")
        if isinstance(dof, bool) or not isinstance(dof, int) or dof <= 0:
            raise InterfaceValidationError(
                f"{field_name}.groups[{standard_name!r}].dof must be a positive int."
            )
        group_state = _copy_string_mapping(
            group_config.get("state", {}),
            field_name=f"{field_name}.groups[{standard_name!r}].state",
        )
        group_action_modes = _copy_string_mapping(
            group_config.get("action_modes"),
            field_name=f"{field_name}.groups[{standard_name!r}].action_modes",
            allow_empty=False,
        )
        group_meta = _copy_mapping(
            group_config.get("meta", {}),
            field_name=f"{field_name}.groups[{standard_name!r}].meta",
        )
        native_name = _ensure_non_empty_string(
            group_config.get("native_name", standard_name),
            field_name=f"{field_name}.groups[{standard_name!r}].native_name",
        )
        _merge_string_mapping(
            state_maps,
            _invert_unique_mapping(
                group_state,
                field_name=f"{field_name}.groups[{standard_name!r}].state",
            ),
            field_name=f"{field_name}.groups[{standard_name!r}].state",
        )
        _merge_string_mapping(
            action_mode_maps,
            _invert_unique_mapping(
                group_action_modes,
                field_name=f"{field_name}.groups[{standard_name!r}].action_modes",
            ),
            field_name=f"{field_name}.groups[{standard_name!r}].action_modes",
        )
        _merge_string_mapping(
            control_target_maps,
            {native_name: standard_name},
            field_name=f"{field_name}.groups[{standard_name!r}].native_name",
        )
        robot_groups.append(
            {
                "name": standard_name,
                "kind": kind,
                "dof": dof,
                "action_modes": list(group_action_modes.values()),
                "state_keys": list(group_state.values()),
                "meta": group_meta,
            }
        )

    return {
        "robot_spec": {
            "name": name,
            "image_keys": list(images.values()),
            "groups": robot_groups,
            "task_keys": list(task.values()),
            "meta": {},
        },
        "modality_maps": {
            "images": _invert_unique_mapping(images, field_name=f"{field_name}.images"),
            "control_targets": control_target_maps,
            "state": state_maps,
            "task": _invert_unique_mapping(task, field_name=f"{field_name}.task"),
            "meta": _invert_unique_mapping(meta, field_name=f"{field_name}.meta"),
            "action_modes": action_mode_maps,
        },
    }


def _expand_model_interface_config(
    interface: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    """Expand one compact model-side YAML interface block."""

    _validate_interface_keys(
        interface,
        field_name=field_name,
        allowed_keys={
            "name",
            "images",
            "state",
            "task",
            "meta",
            "outputs",
        },
    )

    name = _ensure_non_empty_string(interface.get("name"), field_name=f"{field_name}.name")
    images = _copy_string_mapping(
        interface.get("images", {}),
        field_name=f"{field_name}.images",
    )
    state = _copy_string_mapping(
        interface.get("state", {}),
        field_name=f"{field_name}.state",
    )
    task = _copy_string_mapping(
        interface.get("task", {}),
        field_name=f"{field_name}.task",
    )
    meta = _copy_string_mapping(
        interface.get("meta", {}),
        field_name=f"{field_name}.meta",
    )

    outputs = _copy_named_mapping(
        interface.get("outputs"),
        field_name=f"{field_name}.outputs",
        allow_empty=False,
    )
    model_outputs: list[dict[str, Any]] = []
    action_mode_maps: dict[str, str] = {}
    control_target_maps: dict[str, str] = {}

    for standard_target, output_config in outputs.items():
        _validate_interface_keys(
            output_config,
            field_name=f"{field_name}.outputs[{standard_target!r}]",
            allowed_keys={"mode", "dim", "meta", "native_name", "native_mode"},
        )
        mode = _ensure_non_empty_string(
            output_config.get("mode"),
            field_name=f"{field_name}.outputs[{standard_target!r}].mode",
        )
        dim = output_config.get("dim")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise InterfaceValidationError(
                f"{field_name}.outputs[{standard_target!r}].dim must be a positive int."
            )
        output_meta = _copy_mapping(
            output_config.get("meta", {}),
            field_name=f"{field_name}.outputs[{standard_target!r}].meta",
        )
        native_name = _ensure_non_empty_string(
            output_config.get("native_name", standard_target),
            field_name=f"{field_name}.outputs[{standard_target!r}].native_name",
        )
        native_mode = _ensure_non_empty_string(
            output_config.get("native_mode", mode),
            field_name=f"{field_name}.outputs[{standard_target!r}].native_mode",
        )
        _merge_string_mapping(
            control_target_maps,
            {native_name: standard_target},
            field_name=f"{field_name}.outputs[{standard_target!r}].native_name",
        )
        _merge_string_mapping(
            action_mode_maps,
            {native_mode: mode},
            field_name=f"{field_name}.outputs[{standard_target!r}].native_mode",
        )
        model_outputs.append(
            {
                "target": standard_target,
                "mode": mode,
                "dim": dim,
                "meta": output_meta,
            }
        )

    return {
        "model_spec": {
            "name": name,
            "required_image_keys": list(images.values()),
            "required_state_keys": list(state.values()),
            "required_task_keys": list(task.values()),
            "outputs": model_outputs,
            "meta": {},
        },
        "modality_maps": {
            "images": _invert_unique_mapping(images, field_name=f"{field_name}.images"),
            "state": _invert_unique_mapping(state, field_name=f"{field_name}.state"),
            "task": _invert_unique_mapping(task, field_name=f"{field_name}.task"),
            "meta": _invert_unique_mapping(meta, field_name=f"{field_name}.meta"),
            "control_targets": control_target_maps,
            "action_modes": action_mode_maps,
        },
    }


def expand_component_yaml_interface_config(
    loaded: Mapping[str, Any],
    *,
    component: str,
    path: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Expand one compact YAML ``interface`` block into runtime config fields."""

    copied = _copy_mapping(loaded, field_name=f"{path or '<config>'}:{component}")
    raw_interface = copied.pop("interface", None)
    field_name = (
        f"{Path(path)}:{component}.interface"
        if path is not None
        else f"{component}.interface"
    )
    if raw_interface is None:
        raise InterfaceValidationError(
            f"{field_name} is required. YAML-based embodia config must declare "
            "one compact 'interface' block."
        )
    if not isinstance(raw_interface, Mapping):
        raise InterfaceValidationError(
            f"{field_name} must be a mapping, got {type(raw_interface).__name__}."
        )

    if component == "robot":
        expanded = _expand_robot_interface_config(raw_interface, field_name=field_name)
    elif component == "model":
        expanded = _expand_model_interface_config(raw_interface, field_name=field_name)
    else:
        raise InterfaceValidationError(
            f"Unsupported component {component!r} when expanding YAML interface."
        )

    copied.update(expanded)
    return copied


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
       ``interface`` and ``method_aliases``
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
    "expand_component_yaml_interface_config",
    "is_yaml_available",
    "load_component_yaml_config",
    "load_yaml_config",
    "require_yaml",
]
