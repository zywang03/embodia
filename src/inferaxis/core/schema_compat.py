"""Compatibility checks between schema objects."""

from __future__ import annotations

from .command_kinds import get_command_kind_spec, is_known_command_kind
from .schema_models import Action, PolicySpec, RobotSpec
from .schema_validation import (
    _kind_uses_component_dof,
    validate_action,
    validate_policy_spec,
    validate_robot_spec,
)
from .errors import InterfaceValidationError


def ensure_action_supported_by_robot(action: Action, spec: RobotSpec) -> None:
    """Ensure an action is compatible with one robot spec."""

    validate_action(action)
    validate_robot_spec(spec)

    for target, command in action.commands.items():
        component = spec.get_component(target)
        if component is None:
            raise InterfaceValidationError(
                f"robot {spec.name!r} does not define component {target!r}."
            )
        if command.command not in component.command:
            raise InterfaceValidationError(
                f"robot {spec.name!r} component {component.name!r} does not support "
                f"command {command.command!r}; supported commands: "
                f"{component.command!r}."
            )
        if not is_known_command_kind(command.command):
            continue
        command_spec = get_command_kind_spec(command.command)
        if (
            command_spec.allowed_component_types
            and component.type not in command_spec.allowed_component_types
        ):
            raise InterfaceValidationError(
                f"command {command.command!r} is not allowed for robot component "
                f"{component.name!r} of type {component.type!r}; allowed component "
                f"types: {command_spec.allowed_component_types!r}."
            )
        if (
            _kind_uses_component_dof(command_spec)
            and len(command.value) != component.dof
        ):
            raise InterfaceValidationError(
                f"command {command.command!r} for robot component "
                f"{component.name!r} must match component dof={component.dof}, got "
                f"dim={len(command.value)}."
            )


def ensure_action_matches_policy_spec(action: Action, spec: PolicySpec) -> None:
    """Ensure an action matches one policy's declared outputs."""

    validate_action(action)
    validate_policy_spec(spec)

    command_targets = set(action.commands)
    output_targets = {output.target for output in spec.outputs}
    if command_targets != output_targets:
        missing = sorted(output_targets - command_targets)
        extra = sorted(command_targets - output_targets)
        details: list[str] = []
        if missing:
            details.append(f"missing targets: {missing!r}")
        if extra:
            details.append(f"unexpected targets: {extra!r}")
        raise InterfaceValidationError(
            f"policy {spec.name!r} produced commands that do not match its outputs; "
            + ", ".join(details)
            + "."
        )

    for output in spec.outputs:
        command = action.commands[output.target]
        if command.command != output.command:
            raise InterfaceValidationError(
                f"policy {spec.name!r} output {output.target!r} declared command "
                f"{output.command!r}, but produced {command.command!r}."
            )
        if len(command.value) != output.dim:
            raise InterfaceValidationError(
                f"policy {spec.name!r} output {output.target!r} declared dim "
                f"{output.dim}, but produced dim {len(command.value)}."
            )


__all__ = [
    "ensure_action_matches_policy_spec",
    "ensure_action_supported_by_robot",
]
