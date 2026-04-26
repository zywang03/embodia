"""Compatibility exports for inferaxis schema objects.

The schema implementation is split across focused modules. This module keeps
the historical import path stable for users and internal code.
"""

from __future__ import annotations

from .command_kinds import (
    COMMAND_KIND_REGISTRY,
    CUSTOM_COMMAND_KIND_PREFIX,
    BuiltinCommandKind,
    CommandKindSpec,
    get_command_kind_spec,
    is_custom_command_kind_name,
    is_known_command_kind,
    register_command_kind,
)
from .schema_compat import (
    ensure_action_matches_policy_spec,
    ensure_action_supported_by_robot,
)
from .schema_models import (
    KNOWN_COMPONENT_TYPES,
    Action,
    Command,
    ComponentSpec,
    Frame,
    PolicyOutputSpec,
    PolicySpec,
    RobotSpec,
)
from .schema_validation import (
    validate_action,
    validate_command,
    validate_component_spec,
    validate_frame,
    validate_policy_output_spec,
    validate_policy_spec,
    validate_robot_spec,
)

__all__ = [
    "Action",
    "BuiltinCommandKind",
    "COMMAND_KIND_REGISTRY",
    "CUSTOM_COMMAND_KIND_PREFIX",
    "ComponentSpec",
    "Command",
    "CommandKindSpec",
    "Frame",
    "KNOWN_COMPONENT_TYPES",
    "PolicyOutputSpec",
    "PolicySpec",
    "RobotSpec",
    "ensure_action_matches_policy_spec",
    "ensure_action_supported_by_robot",
    "get_command_kind_spec",
    "is_custom_command_kind_name",
    "is_known_command_kind",
    "register_command_kind",
    "validate_action",
    "validate_command",
    "validate_component_spec",
    "validate_frame",
    "validate_policy_output_spec",
    "validate_policy_spec",
    "validate_robot_spec",
]
