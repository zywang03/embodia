"""Action-mode modality helpers."""

from __future__ import annotations

from collections.abc import Mapping

from ..errors import InterfaceValidationError
from ..schema import Action, ModelSpec, RobotSpec
from ._common import ACTION_MODES, resolve_modality_mapping


def get_mode_map(owner: object) -> Mapping[str, str]:
    """Resolve the action-mode remapping table for a class or instance."""

    return resolve_modality_mapping(owner, ACTION_MODES)


def ensure_supported(
    action: Action,
    spec: RobotSpec,
    *,
    owner_label: str,
    owner_name: str | None = None,
) -> None:
    """Ensure an action is supported by one robot spec."""

    for command in action.commands:
        group = spec.get_group(command.target)
        if group is None:
            detail = "" if owner_name is None else f" {owner_name!r}"
            raise InterfaceValidationError(
                f"{owner_label}{detail} is missing control group {command.target!r}."
            )
        if command.mode not in group.action_modes:
            detail = "" if owner_name is None else f" {owner_name!r}"
            raise InterfaceValidationError(
                f"control group {command.target!r} on {owner_label}{detail} does "
                f"not support mode {command.mode!r}; supported modes: "
                f"{group.action_modes!r}."
            )


def ensure_model_output(
    action: Action,
    *,
    spec: ModelSpec,
) -> None:
    """Ensure a model-produced action matches the declared outputs."""

    produced_targets = {command.target for command in action.commands}
    expected_targets = {output.target for output in spec.outputs}
    if produced_targets != expected_targets:
        raise InterfaceValidationError(
            f"model {spec.name!r} produced targets {sorted(produced_targets)!r}, "
            f"expected {sorted(expected_targets)!r}."
        )

    for output in spec.outputs:
        command = next(item for item in action.commands if item.target == output.target)
        if command.mode != output.mode:
            raise InterfaceValidationError(
                f"model {spec.name!r} output {output.target!r} declared mode "
                f"{output.mode!r}, but produced {command.mode!r}."
            )


def pair_problem(
    *,
    robot_spec: RobotSpec,
    model_spec: ModelSpec,
) -> str | None:
    """Return the first robot/model action-structure mismatch, if any."""

    for output in model_spec.outputs:
        group = robot_spec.get_group(output.target)
        if group is None:
            return f"robot is missing required control group {output.target!r}."
        if output.mode not in group.action_modes:
            return (
                f"control group {output.target!r} does not support model output "
                f"mode {output.mode!r}; supported modes: {group.action_modes!r}."
            )
        if output.dim != group.dof:
            return (
                f"control group {output.target!r} has dof={group.dof}, but model "
                f"declares dim={output.dim}."
            )
    return None


__all__ = [
    "ACTION_MODES",
    "ensure_model_output",
    "ensure_supported",
    "get_mode_map",
    "pair_problem",
]
