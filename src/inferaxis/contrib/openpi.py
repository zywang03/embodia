"""Optional OpenPI compatibility helpers built on inferaxis's remote transport.

This module keeps OpenPI-specific request/response adaptation out of inferaxis's
core runtime. It lets a local inferaxis-wrapped robot talk directly to a remote
OpenPI policy server without requiring the remote policy to inherit
``PolicyMixin``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.arraylike import to_numpy_array
from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Command, Frame, PolicyOutputSpec, PolicySpec, RobotSpec
from ..core.transform import (
    coerce_frame,
    coerce_policy_spec,
    coerce_robot_spec,
)
from ..runtime.checks import (
    validate_action,
    validate_frame,
    validate_robot_spec,
)
from ..runtime.shared.dispatch import ROBOT_GET_SPEC_METHODS, resolve_callable_method
from .remote_transform import _coerce_action_rows


ActionSelector = int | slice | tuple[int, int] | Sequence[int]


@dataclass(slots=True)
class OpenPIActionGroup:
    """Describe how one OpenPI action vector maps into one inferaxis command."""

    target: str
    command: str
    selector: ActionSelector
    ref_frame: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _normalize_selector(
    selector: ActionSelector,
    *,
    vector_length: int,
    field_name: str,
) -> list[int]:
    """Normalize one vector selector into explicit indices."""

    if isinstance(selector, bool):
        raise InterfaceValidationError(
            f"{field_name} must be an int, slice, (start, stop), or sequence of ints."
        )

    if isinstance(selector, int):
        index = selector
        if index < 0:
            index += vector_length
        if index < 0 or index >= vector_length:
            raise InterfaceValidationError(
                f"{field_name} index {selector!r} is out of range for vector length "
                f"{vector_length}."
            )
        return [index]

    if isinstance(selector, slice):
        indices = list(range(*selector.indices(vector_length)))
        if not indices:
            raise InterfaceValidationError(
                f"{field_name} resolved to an empty slice for vector length "
                f"{vector_length}."
            )
        return indices

    if (
        isinstance(selector, tuple)
        and len(selector) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in selector)
    ):
        start, stop = selector
        indices = list(range(*slice(start, stop).indices(vector_length)))
        if not indices:
            raise InterfaceValidationError(
                f"{field_name} resolved to an empty range for vector length "
                f"{vector_length}."
            )
        return indices

    if isinstance(selector, (str, bytes)) or not isinstance(selector, Sequence):
        raise InterfaceValidationError(
            f"{field_name} must be an int, slice, (start, stop), or sequence of ints."
        )

    indices: list[int] = []
    for item_index, item in enumerate(selector):
        if isinstance(item, bool) or not isinstance(item, int):
            raise InterfaceValidationError(
                f"{field_name}[{item_index}] must be an int, got "
                f"{type(item).__name__}."
            )
        normalized = item
        if normalized < 0:
            normalized += vector_length
        if normalized < 0 or normalized >= vector_length:
            raise InterfaceValidationError(
                f"{field_name}[{item_index}]={item!r} is out of range for vector "
                f"length {vector_length}."
            )
        indices.append(normalized)

    if not indices:
        raise InterfaceValidationError(f"{field_name} must not be empty.")
    if len(set(indices)) != len(indices):
        raise InterfaceValidationError(f"{field_name} contains duplicate indices.")
    return indices


def _validate_action_groups(
    groups: Sequence[OpenPIActionGroup],
) -> list[OpenPIActionGroup]:
    """Validate a sequence of OpenPI action-group mappings."""

    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise InterfaceValidationError(
            "action_groups must be a sequence of OpenPIActionGroup objects."
        )
    if not groups:
        raise InterfaceValidationError("action_groups must not be empty.")

    normalized: list[OpenPIActionGroup] = []
    seen_targets: set[str] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, OpenPIActionGroup):
            raise InterfaceValidationError(
                f"action_groups[{index}] must be OpenPIActionGroup, got "
                f"{type(group).__name__}."
            )
        if not isinstance(group.target, str) or not group.target.strip():
            raise InterfaceValidationError(
                f"action_groups[{index}].target must be a non-empty string."
            )
        if not isinstance(group.command, str) or not group.command.strip():
            raise InterfaceValidationError(
                f"action_groups[{index}].command must be a non-empty string."
            )
        if group.target in seen_targets:
            raise InterfaceValidationError(
                f"action_groups contains duplicate target {group.target!r}."
            )
        seen_targets.add(group.target)
        normalized.append(
            OpenPIActionGroup(
                target=group.target,
                command=group.command,
                selector=group.selector,
                ref_frame=group.ref_frame,
                meta=dict(group.meta),
            )
        )
    return normalized


def frame_to_openpi_obs(
    frame: Frame | Mapping[str, Any],
    *,
    image_map: Mapping[str, str] | None = None,
    state_map: Mapping[str, str] | None = None,
    task_map: Mapping[str, str] | None = None,
    meta_map: Mapping[str, str] | None = None,
    include_timestamp_ns: bool = False,
    include_sequence_id: bool = False,
) -> dict[str, Any]:
    """Build one flat OpenPI observation dictionary from an inferaxis frame.

    This helper is intentionally small and only covers straightforward top-level
    key remapping. For nested OpenPI observation payloads, pass your own
    ``obs_builder(frame)`` callable to :class:`OpenPITransform`.
    """

    normalized_frame = coerce_frame(frame)
    validate_frame(normalized_frame)

    obs: dict[str, Any] = {}

    for mapping, source_name, source in (
        (image_map, "frame.images", normalized_frame.images),
        (state_map, "frame.state", normalized_frame.state),
        (task_map, "frame.task", normalized_frame.task),
        (meta_map, "frame.meta", normalized_frame.meta),
    ):
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise InterfaceValidationError(
                f"{source_name} mapping must be a mapping, got {type(mapping).__name__}."
            )
        for openpi_key, frame_key in mapping.items():
            if not isinstance(openpi_key, str) or not openpi_key.strip():
                raise InterfaceValidationError(
                    f"{source_name} mapping keys must be non-empty strings."
                )
            if not isinstance(frame_key, str) or not frame_key.strip():
                raise InterfaceValidationError(
                    f"{source_name} mapping values must be non-empty strings."
                )
            if frame_key not in source:
                raise InterfaceValidationError(
                    f"{source_name} is missing required key {frame_key!r} for "
                    f"OpenPI field {openpi_key!r}."
                )
            obs[openpi_key] = source[frame_key]

    if include_timestamp_ns:
        obs["timestamp_ns"] = normalized_frame.timestamp_ns
    if include_sequence_id and normalized_frame.sequence_id is not None:
        obs["sequence_id"] = normalized_frame.sequence_id
    return obs


def _flatten_numeric_value(value: object, *, field_name: str) -> list[float]:
    """Convert one scalar or tensor-like numeric value into ``list[float]``."""

    array = to_numpy_array(
        value,
        field_name=field_name,
        wrap_scalar=True,
        numeric_only=True,
        allow_bool=False,
        copy=False,
        dtype=np.float64,
    )
    return array.reshape(-1).tolist()


def _extract_prompt(task: Mapping[str, Any]) -> str | None:
    """Extract one common language-conditioning field when present."""

    for key in ("prompt", "language_instruction", "instruction", "text"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _selector_width(selector: ActionSelector, *, field_name: str) -> int:
    """Return the number of dimensions described by one selector."""

    if isinstance(selector, bool):
        raise InterfaceValidationError(
            f"{field_name} must be an int, slice, (start, stop), or sequence of ints."
        )
    if isinstance(selector, int):
        return 1
    if isinstance(selector, slice):
        if selector.start is None or selector.stop is None:
            raise InterfaceValidationError(
                f"{field_name} must use explicit start and stop values."
            )
        return max(0, selector.stop - selector.start)
    if (
        isinstance(selector, tuple)
        and len(selector) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in selector)
    ):
        start, stop = selector
        return max(0, stop - start)
    if isinstance(selector, (str, bytes)) or not isinstance(selector, Sequence):
        raise InterfaceValidationError(
            f"{field_name} must be an int, slice, (start, stop), or sequence of ints."
        )
    if not selector:
        raise InterfaceValidationError(f"{field_name} must not be empty.")
    return len(selector)


def build_default_openpi_obs(
    frame: Frame | Mapping[str, Any],
    *,
    robot_spec: RobotSpec | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one default nested OpenPI-style observation payload.

    This default path is intentionally simple:

    - ``observation.images`` keeps inferaxis image keys unchanged
    - ``observation.state`` becomes one flat numeric state vector
    - ``prompt`` is read from common task keys when present
    """

    normalized_frame = coerce_frame(frame)
    validate_frame(normalized_frame)

    spec: RobotSpec | None = None
    if robot_spec is not None:
        spec = coerce_robot_spec(robot_spec)
        validate_robot_spec(spec)

    if spec is not None:
        images = {
            key: normalized_frame.images[key]
            for key in spec.image_keys
            if key in normalized_frame.images
        }
        state_values: list[float] = []
        for component in spec.components:
            state_key = component.name
            if state_key not in normalized_frame.state:
                raise InterfaceValidationError(
                    f"frame.state is missing required key {state_key!r} for "
                    "default OpenPI adaptation."
                )
            state_values.extend(
                _flatten_numeric_value(
                    normalized_frame.state[state_key],
                    field_name=f"frame.state[{state_key!r}]",
                )
            )
    else:
        images = dict(normalized_frame.images)
        state_values = []
        for state_key, value in normalized_frame.state.items():
            state_values.extend(
                _flatten_numeric_value(
                    value,
                    field_name=f"frame.state[{state_key!r}]",
                )
            )

    obs: dict[str, Any] = {
        "observation": {
            "images": images,
            "state": state_values,
        }
    }
    prompt = _extract_prompt(normalized_frame.task)
    if prompt is not None:
        obs["prompt"] = prompt
    return obs


def build_default_openpi_action_groups(
    robot_spec: RobotSpec | Mapping[str, Any],
) -> list[OpenPIActionGroup]:
    """Infer one default action split from a robot spec.

    Each component must expose exactly one supported command so inferaxis can
    decode one returned OpenPI action vector without extra user-provided schema.
    """

    spec = coerce_robot_spec(robot_spec)
    validate_robot_spec(spec)

    groups: list[OpenPIActionGroup] = []
    offset = 0
    for component in spec.components:
        if len(component.command) != 1:
            raise InterfaceValidationError(
                "default OpenPI adaptation requires each robot component to "
                "declare exactly one supported command; component "
                f"{component.name!r} exposes {len(component.command)} commands."
            )
        groups.append(
            OpenPIActionGroup(
                target=component.name,
                command=component.command[0],
                selector=(offset, offset + component.dof),
            )
        )
        offset += component.dof
    return groups


def build_default_openpi_policy_spec(
    robot_spec: RobotSpec | Mapping[str, Any],
    *,
    name: str = "openpi_remote_policy",
) -> PolicySpec:
    """Synthesize one inferaxis policy spec from a robot spec."""

    spec = coerce_robot_spec(robot_spec)
    validate_robot_spec(spec)
    action_groups = build_default_openpi_action_groups(spec)
    return PolicySpec(
        name=name,
        required_image_keys=list(spec.image_keys),
        required_state_keys=spec.all_state_keys(),
        required_task_keys=[],
        outputs=[
            PolicyOutputSpec(
                target=group.target,
                command=group.command,
                dim=_selector_width(
                    group.selector,
                    field_name=f"{group.target}.selector",
                ),
            )
            for group in action_groups
        ],
    )


def openpi_action_plan_from_response(
    response_or_actions: object,
    *,
    action_groups: Sequence[OpenPIActionGroup],
) -> list[Action]:
    """Convert one OpenPI-style action chunk into inferaxis actions."""

    groups = _validate_action_groups(action_groups)
    rows = _coerce_action_rows(response_or_actions)

    plan: list[Action] = []
    for row_index, row in enumerate(rows):
        commands: dict[str, Command] = {}
        for group_index, group in enumerate(groups):
            indices = _normalize_selector(
                group.selector,
                vector_length=len(row),
                field_name=(
                    f"action_groups[{group_index}].selector for row {row_index}"
                ),
            )
            commands[group.target] = Command(
                command=group.command,
                value=row[indices].copy(),
                ref_frame=group.ref_frame,
                meta=dict(group.meta),
            )
        action = Action(commands=commands)
        validate_action(action)
        plan.append(action)
    return plan


def openpi_first_action_from_response(
    response_or_actions: object,
    *,
    action_groups: Sequence[OpenPIActionGroup],
) -> Action:
    """Convert one OpenPI response and return the first inferaxis action."""

    return openpi_action_plan_from_response(
        response_or_actions,
        action_groups=action_groups,
    )[0]


class OpenPITransform:
    """OpenPI request/response adapter for ``RemotePolicy(openpi=True)``.

    The default behavior is intentionally lightweight:

    - infer OpenPI observation packing from the wrapped inferaxis robot spec
    - infer action-vector splits from robot components
    - synthesize one local policy spec for `check_policy` / `check_pair`

    Advanced users can still override pieces by passing their own
    ``robot_spec``, ``obs_builder``, ``action_groups``, or ``policy_spec``.
    """

    def __init__(
        self,
        *,
        robot_spec: RobotSpec | Mapping[str, Any] | None = None,
        obs_builder: Callable[[Frame], Mapping[str, Any]] | None = None,
        action_groups: Sequence[OpenPIActionGroup] | None = None,
        policy_spec: PolicySpec | Mapping[str, Any] | None = None,
    ) -> None:
        self._robot_spec = (
            None if robot_spec is None else coerce_robot_spec(robot_spec)
        )
        if self._robot_spec is not None:
            validate_robot_spec(self._robot_spec)
        self._obs_builder = obs_builder
        self._policy_spec = (
            None if policy_spec is None else coerce_policy_spec(policy_spec)
        )
        self._action_groups = (
            None if action_groups is None else _validate_action_groups(action_groups)
        )
        if self._action_groups is None and self._robot_spec is not None:
            self._action_groups = build_default_openpi_action_groups(self._robot_spec)
        if self._policy_spec is None and self._robot_spec is not None:
            self._policy_spec = build_default_openpi_policy_spec(self._robot_spec)

    def bind_robot(self, robot: object) -> None:
        """Infer OpenPI adaptation rules from one inferaxis robot object."""

        if self._robot_spec is not None:
            if self._action_groups is None:
                self._action_groups = build_default_openpi_action_groups(
                    self._robot_spec
                )
            if self._policy_spec is None:
                self._policy_spec = build_default_openpi_policy_spec(self._robot_spec)
            return

        get_spec, get_spec_name = resolve_callable_method(robot, ROBOT_GET_SPEC_METHODS)
        if not callable(get_spec) or get_spec_name is None:
            raise InterfaceValidationError(
                "RemotePolicy(openpi=True) requires the robot to expose "
                "inferaxis_get_spec() or get_spec()."
            )

        try:
            raw_spec = get_spec()
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(robot).__name__}.{get_spec_name}() raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        spec = coerce_robot_spec(raw_spec)
        validate_robot_spec(spec)
        self._robot_spec = spec
        self._action_groups = build_default_openpi_action_groups(spec)
        self._policy_spec = build_default_openpi_policy_spec(spec)

    def build_obs(self, frame: Frame | Mapping[str, Any]) -> dict[str, Any]:
        """Convert one inferaxis frame into the default OpenPI-style payload."""

        if self._obs_builder is not None:
            normalized_frame = coerce_frame(frame)
            validate_frame(normalized_frame)
            obs = self._obs_builder(normalized_frame)
            if not isinstance(obs, Mapping):
                raise InterfaceValidationError(
                    "OpenPITransform.obs_builder(frame) must return a mapping, "
                    f"got {type(obs).__name__}."
                )
            return dict(obs)

        return build_default_openpi_obs(frame, robot_spec=self._robot_spec)

    def action_plan_from_response(
        self,
        response_or_actions: object,
    ) -> list[Action]:
        """Convert one OpenPI action chunk into inferaxis actions."""

        if self._action_groups is None:
            raise InterfaceValidationError(
                "RemotePolicy(openpi=True) must be bound to an inferaxis robot "
                "before the first inference step."
            )
        return openpi_action_plan_from_response(
            response_or_actions,
            action_groups=self._action_groups,
        )

    def first_action_from_response(
        self,
        response_or_actions: object,
    ) -> Action:
        """Convert one OpenPI response into the first inferaxis action."""

        return self.action_plan_from_response(response_or_actions)[0]

    def get_policy_spec(self) -> PolicySpec:
        """Return the synthesized local policy spec when available."""

        if self._policy_spec is None:
            raise InterfaceValidationError(
                "RemotePolicy(openpi=True) does not have a synthesized policy "
                "spec yet. Bind it to an inferaxis robot first."
            )
        return self._policy_spec


__all__ = [
    "OpenPITransform",
    "OpenPIActionGroup",
    "frame_to_openpi_obs",
    "openpi_action_plan_from_response",
    "openpi_first_action_from_response",
]
