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


def _ensure_non_empty_string(value: object, *, field_name: str) -> str:
    """Validate one non-empty string field."""

    if not isinstance(value, str) or not value.strip():
        raise InterfaceValidationError(f"{field_name} must be a non-empty string.")
    return value


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


def _copy_string_list(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = True,
) -> list[str]:
    """Validate and copy one ``list[str]`` block."""

    if not isinstance(value, list):
        raise InterfaceValidationError(
            f"{field_name} must be a list[str], got {type(value).__name__}."
        )

    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _ensure_non_empty_string(item, field_name=f"{field_name}[{index}]")
        if text in seen:
            raise InterfaceValidationError(
                f"{field_name} contains duplicate entry {text!r}."
            )
        seen.add(text)
        result.append(text)

    if not allow_empty and not result:
        raise InterfaceValidationError(f"{field_name} must not be empty.")
    return result


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


def _validate_keys(
    value: Mapping[str, Any],
    *,
    field_name: str,
    allowed_keys: set[str],
) -> None:
    """Ensure one mapping only uses supported keys."""

    unknown = sorted(key for key in value if key not in allowed_keys)
    if not unknown:
        return

    expected = ", ".join(repr(key) for key in sorted(allowed_keys))
    found = ", ".join(repr(key) for key in unknown)
    raise InterfaceValidationError(
        f"{field_name} contains unsupported field(s) {found}. "
        f"Expected only: {expected}."
    )


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
    """Merge one ``mapping[str, str]`` while detecting collisions."""

    for key, value in source.items():
        existing = destination.get(key)
        if existing is not None and existing != value:
            raise InterfaceValidationError(
                f"{field_name} maps {key!r} to both {existing!r} and {value!r}."
            )
        destination[key] = value


def _expand_shared_schema(
    schema: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    """Expand one shared runtime schema block."""

    _validate_keys(
        schema,
        field_name=field_name,
        allowed_keys={"images", "task", "meta", "groups"},
    )

    image_keys = _copy_string_list(
        schema.get("images", []),
        field_name=f"{field_name}.images",
        allow_empty=True,
    )
    task_keys = _copy_string_list(
        schema.get("task", []),
        field_name=f"{field_name}.task",
        allow_empty=True,
    )
    meta = _copy_mapping(
        schema.get("meta", {}),
        field_name=f"{field_name}.meta",
    )

    groups = _copy_named_mapping(
        schema.get("groups"),
        field_name=f"{field_name}.groups",
        allow_empty=False,
    )

    expanded_groups: list[dict[str, Any]] = []
    group_index: dict[str, dict[str, Any]] = {}

    for group_name, raw_group in groups.items():
        _validate_keys(
            raw_group,
            field_name=f"{field_name}.groups[{group_name!r}]",
            allowed_keys={"kind", "dof", "state", "command_kinds", "meta"},
        )
        kind = _ensure_non_empty_string(
            raw_group.get("kind"),
            field_name=f"{field_name}.groups[{group_name!r}].kind",
        )
        dof = raw_group.get("dof")
        if isinstance(dof, bool) or not isinstance(dof, int) or dof <= 0:
            raise InterfaceValidationError(
                f"{field_name}.groups[{group_name!r}].dof must be a positive int."
            )
        state_keys = _copy_string_list(
            raw_group.get("state", []),
            field_name=f"{field_name}.groups[{group_name!r}].state",
            allow_empty=True,
        )
        command_kinds = _copy_string_list(
            raw_group.get("command_kinds"),
            field_name=f"{field_name}.groups[{group_name!r}].command_kinds",
            allow_empty=False,
        )
        group_meta = _copy_mapping(
            raw_group.get("meta", {}),
            field_name=f"{field_name}.groups[{group_name!r}].meta",
        )

        expanded = {
            "name": group_name,
            "kind": kind,
            "dof": dof,
            "supported_command_kinds": command_kinds,
            "state_keys": state_keys,
            "meta": group_meta,
        }
        expanded_groups.append(expanded)
        group_index[group_name] = expanded

    return {
        "image_keys": image_keys,
        "task_keys": task_keys,
        "meta": meta,
        "groups": expanded_groups,
        "group_index": group_index,
    }


def _expand_robot_config(
    loaded: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    """Expand one robot YAML section into runtime config."""

    _validate_keys(
        loaded,
        field_name=field_name,
        allowed_keys={"name", "method_aliases", "remote_policy"},
    )

    shared = _expand_shared_schema(schema, field_name=f"{field_name}.schema")
    name = _ensure_non_empty_string(
        loaded.get("name", "robot"),
        field_name=f"{field_name}.name",
    )
    return {
        "robot_spec": {
            "name": name,
            "image_keys": list(shared["image_keys"]),
            "groups": list(shared["groups"]),
            "task_keys": list(shared["task_keys"]),
            "meta": dict(shared["meta"]),
        },
        "modality_maps": {},
    }


def _expand_model_config(
    loaded: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    """Expand one model YAML section into runtime config."""

    _validate_keys(
        loaded,
        field_name=field_name,
        allowed_keys={"name", "requires", "outputs", "method_aliases"},
    )

    shared = _expand_shared_schema(schema, field_name=f"{field_name}.schema")
    name = _ensure_non_empty_string(
        loaded.get("name", "model"),
        field_name=f"{field_name}.name",
    )

    requires = loaded.get("requires", {})
    if not isinstance(requires, Mapping):
        raise InterfaceValidationError(
            f"{field_name}.requires must be a mapping, got {type(requires).__name__}."
        )
    _validate_keys(
        requires,
        field_name=f"{field_name}.requires",
        allowed_keys={"images", "state", "task"},
    )

    all_state_keys: list[str] = []
    for group in shared["groups"]:
        for state_key in group["state_keys"]:
            if state_key not in all_state_keys:
                all_state_keys.append(state_key)

    required_image_keys = _copy_string_list(
        requires.get("images", list(shared["image_keys"])),
        field_name=f"{field_name}.requires.images",
        allow_empty=True,
    )
    required_state_keys = _copy_string_list(
        requires.get("state", all_state_keys),
        field_name=f"{field_name}.requires.state",
        allow_empty=True,
    )
    required_task_keys = _copy_string_list(
        requires.get("task", list(shared["task_keys"])),
        field_name=f"{field_name}.requires.task",
        allow_empty=True,
    )

    outputs = _copy_string_mapping(
        loaded.get("outputs"),
        field_name=f"{field_name}.outputs",
        allow_empty=False,
    )
    expanded_outputs: list[dict[str, Any]] = []
    for target, command_kind in outputs.items():
        group = shared["group_index"].get(target)
        if group is None:
            expected = ", ".join(repr(name) for name in sorted(shared["group_index"]))
            raise InterfaceValidationError(
                f"{field_name}.outputs contains unknown target {target!r}. "
                f"Expected one of: {expected}."
            )
        if command_kind not in group["supported_command_kinds"]:
            raise InterfaceValidationError(
                f"{field_name}.outputs[{target!r}] uses command kind "
                f"{command_kind!r}, but shared schema group {target!r} only "
                f"supports {group['supported_command_kinds']!r}."
            )
        expanded_outputs.append(
            {
                "target": target,
                "command_kind": command_kind,
                "dim": group["dof"],
                "meta": {},
            }
        )

    return {
        "model_spec": {
            "name": name,
            "required_image_keys": required_image_keys,
            "required_state_keys": required_state_keys,
            "required_task_keys": required_task_keys,
            "outputs": expanded_outputs,
            "meta": dict(shared["meta"]),
        },
        "modality_maps": {},
    }


def expand_component_yaml_config(
    loaded: Mapping[str, Any],
    *,
    component: str,
    path: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Expand one component YAML mapping into runtime config fields."""

    copied = _copy_mapping(loaded, field_name=f"{path or '<config>'}:{component}")
    raw_schema = copied.pop("schema", None)
    if raw_schema is None:
        raise InterfaceValidationError(
            f"{Path(path) if path is not None else '<config>'}:{component} is "
            "missing the required shared 'schema' block."
        )
    if not isinstance(raw_schema, Mapping):
        raise InterfaceValidationError(
            f"{Path(path) if path is not None else '<config>'}:schema must be a "
            f"mapping, got {type(raw_schema).__name__}."
        )

    field_name = f"{Path(path)}:{component}" if path is not None else component
    if component == "robot":
        expanded = _expand_robot_config(copied, raw_schema, field_name=field_name)
    elif component == "model":
        expanded = _expand_model_config(copied, raw_schema, field_name=field_name)
    else:
        raise InterfaceValidationError(
            f"Unsupported component {component!r} when expanding YAML config."
        )

    copied.pop("name", None)
    copied.pop("requires", None)
    copied.pop("outputs", None)
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
    """Load config for one embodia component from a YAML file."""

    loaded = load_yaml_config(path)
    component_value = loaded.get(component)
    if component_value is not None:
        if not isinstance(component_value, Mapping):
            raise InterfaceValidationError(
                f"YAML section {component!r} at {Path(path)} must be a mapping, "
                f"got {type(component_value).__name__}."
            )
        result = _copy_mapping(component_value, field_name=f"{Path(path)}:{component}")
        shared_schema = loaded.get("schema")
        if shared_schema is not None:
            if not isinstance(shared_schema, Mapping):
                raise InterfaceValidationError(
                    f"YAML section 'schema' at {Path(path)} must be a mapping, got "
                    f"{type(shared_schema).__name__}."
                )
            result["schema"] = _copy_mapping(
                shared_schema,
                field_name=f"{Path(path)}:schema",
            )
        return result

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
    "expand_component_yaml_config",
    "is_yaml_available",
    "load_component_yaml_config",
    "load_yaml_config",
    "require_yaml",
]
