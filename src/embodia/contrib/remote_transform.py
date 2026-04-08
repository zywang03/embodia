"""Payload transforms for embodia remote inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ..core.arraylike import optional_array_to_list
from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame
from ..core.transform import coerce_action, coerce_frame, frame_to_dict
from ..runtime.checks import validate_action, validate_frame


def _coerce_optional_python_list(value: object, *, field_name: str) -> object:
    """Convert optional ndarray/tensor inputs into plain Python lists."""

    converted = optional_array_to_list(value, field_name=field_name)
    if converted is not None:
        return converted

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()

    return value


def _ensure_float_list(value: object, *, field_name: str) -> list[float]:
    """Validate and normalize one numeric action vector."""

    value = _coerce_optional_python_list(value, field_name=field_name)
    if isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list):
        raise InterfaceValidationError(
            f"{field_name} must be a list-like numeric vector, got "
            f"{type(value).__name__}."
        )

    numbers: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise InterfaceValidationError(
                f"{field_name}[{index}] must be a real number, got "
                f"{type(item).__name__}."
            )
        numbers.append(float(item))
    return numbers


def _extract_actions_value(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
) -> object:
    """Return the raw remote ``actions`` payload."""

    if isinstance(response_or_actions, Mapping):
        if "actions" not in response_or_actions:
            raise InterfaceValidationError(
                "remote response must contain an 'actions' field."
            )
        return response_or_actions["actions"]
    return response_or_actions


def _coerce_action_rows(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
) -> list[list[float]]:
    """Normalize remote action payloads into ``list[list[float]]``."""

    actions = _coerce_optional_python_list(
        _extract_actions_value(response_or_actions),
        field_name="actions",
    )
    if isinstance(actions, tuple):
        actions = list(actions)

    if not isinstance(actions, list):
        raise InterfaceValidationError(
            "remote actions must be a 1D or 2D list-like numeric payload, got "
            f"{type(actions).__name__}."
        )
    if not actions:
        raise InterfaceValidationError("remote inference returned an empty action chunk.")

    first = actions[0]
    if isinstance(first, bool):
        raise InterfaceValidationError(
            "remote actions must contain numeric values, not booleans."
        )
    if isinstance(first, Real):
        return [_ensure_float_list(actions, field_name="actions")]

    rows: list[list[float]] = []
    for row_index, row in enumerate(actions):
        rows.append(
            _ensure_float_list(row, field_name=f"actions[{row_index}]")
        )
    return rows


def _read_action_shape_from_embodia_metadata(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
) -> tuple[str | None, str | None, str | None]:
    """Read action target/kind/ref-frame hints from one remote response."""

    if not isinstance(response_or_actions, Mapping):
        return None, None, None

    embodia_meta = response_or_actions.get("embodia")
    if not isinstance(embodia_meta, Mapping):
        return None, None, None

    target = embodia_meta.get("action_target")
    if not isinstance(target, str) or not target.strip():
        target = None

    kind = embodia_meta.get("action_kind")
    if not isinstance(kind, str) or not kind.strip():
        kind = None

    ref_frame = embodia_meta.get("action_ref_frame")
    if ref_frame is not None:
        if not isinstance(ref_frame, str) or not ref_frame.strip():
            raise InterfaceValidationError(
                "remote response embodia.action_ref_frame must be a non-empty "
                "string when provided."
            )

    return target, kind, ref_frame


def _coerce_embodia_action_plan(
    plan: Action | Mapping[str, Any] | Sequence[Action | Mapping[str, Any]],
) -> list[Action]:
    """Normalize an embodia action plan into a validated list of actions."""

    if isinstance(plan, Action) or isinstance(plan, Mapping):
        action = coerce_action(plan)
        validate_action(action)
        return [action]

    if isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
        raise InterfaceValidationError(
            "action plan must be one action-like object or a sequence of "
            f"action-like objects, got {type(plan).__name__}."
        )

    actions: list[Action] = []
    for index, item in enumerate(plan):
        action = coerce_action(item)
        try:
            validate_action(action)
        except InterfaceValidationError as exc:
            raise InterfaceValidationError(
                f"invalid action at plan index {index}: {exc}"
            ) from exc
        actions.append(action)

    if not actions:
        raise InterfaceValidationError("action plan must not be empty.")
    return actions


def actions_to_action_plan(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    *,
    target: str | None = None,
    kind: str | None = None,
    ref_frame: str | None = None,
) -> list[Action]:
    """Convert one remote action chunk into embodia-standard actions."""

    metadata_target, metadata_kind, metadata_ref_frame = (
        _read_action_shape_from_embodia_metadata(response_or_actions)
    )
    resolved_target = target if target is not None else metadata_target
    resolved_kind = kind if kind is not None else metadata_kind
    resolved_ref_frame = (
        ref_frame if ref_frame is not None else metadata_ref_frame
    )

    if resolved_target is None:
        raise InterfaceValidationError(
            "remote action payload does not define an action target. Provide "
            "target=..., include embodia.action_target in the response, or "
            "use a custom response_to_action callback."
        )
    if resolved_kind is None:
        raise InterfaceValidationError(
            "remote action payload does not define a command kind. Provide "
            "kind=..., include embodia.action_kind in the response, or use "
            "a custom response_to_action callback."
        )

    plan = [
        Action.single(
            target=resolved_target,
            kind=resolved_kind,
            value=row,
            ref_frame=resolved_ref_frame,
        )
        for row in _coerce_action_rows(response_or_actions)
    ]
    for action in plan:
        validate_action(action)
    return plan


def first_action_from_response(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    *,
    target: str | None = None,
    kind: str | None = None,
    ref_frame: str | None = None,
) -> Action:
    """Convert one remote action chunk and return its first action."""

    return actions_to_action_plan(
        response_or_actions,
        target=target,
        kind=kind,
        ref_frame=ref_frame,
    )[0]


def response_from_action_plan(
    plan: Action | Mapping[str, Any] | Sequence[Action | Mapping[str, Any]],
    *,
    include_embodia_metadata: bool = True,
) -> dict[str, Any]:
    """Convert embodia actions into one remote response dictionary."""

    actions = _coerce_embodia_action_plan(plan)
    commands_per_action = [len(action.commands) for action in actions]
    if any(count != 1 for count in commands_per_action):
        raise InterfaceValidationError(
            "response_from_action_plan() currently expects one command per "
            "action because the current remote wire format carries one action "
            "vector per step."
        )

    first_target, first_command = next(iter(actions[0].commands.items()))
    response: dict[str, Any] = {
        "actions": [list(next(iter(action.commands.values())).value) for action in actions],
    }

    if include_embodia_metadata:
        response["embodia"] = {
            "action_target": first_target,
            "action_kind": first_command.kind,
            "action_ref_frame": first_command.ref_frame,
            "chunk_size": len(actions),
        }

    return response


@dataclass(slots=True)
class RemoteTransform:
    """Bundle remote payload conversion in one small object.

    This keeps robot/policy code focused on one transform object instead of
    scattering transport-specific conversion details across multiple callbacks.
    """

    command_kind: str
    action_target: str = "arm"
    ref_frame: str | None = None
    frame_to_obs_fn: Callable[[Frame], Mapping[str, Any]] | None = None
    obs_to_frame_fn: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]] | None = None
    response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None
    include_embodia_metadata: bool = True

    def __post_init__(self) -> None:
        """Validate configuration early for clearer runtime errors."""

        if not isinstance(self.action_target, str) or not self.action_target.strip():
            raise InterfaceValidationError(
                "RemoteTransform.action_target must be a non-empty string."
            )
        if not isinstance(self.command_kind, str) or not self.command_kind.strip():
            raise InterfaceValidationError(
                "RemoteTransform.command_kind must be a non-empty string."
            )
        if self.ref_frame is not None and (
            not isinstance(self.ref_frame, str) or not self.ref_frame.strip()
        ):
            raise InterfaceValidationError(
                "RemoteTransform.ref_frame must be a non-empty string when provided."
            )
        if self.frame_to_obs_fn is not None and not callable(self.frame_to_obs_fn):
            raise InterfaceValidationError(
                "RemoteTransform.frame_to_obs_fn must be callable when provided."
            )
        if self.obs_to_frame_fn is not None and not callable(self.obs_to_frame_fn):
            raise InterfaceValidationError(
                "RemoteTransform.obs_to_frame_fn must be callable when provided."
            )
        if self.response_builder is not None and not callable(self.response_builder):
            raise InterfaceValidationError(
                "RemoteTransform.response_builder must be callable when provided."
            )
        if not isinstance(self.include_embodia_metadata, bool):
            raise InterfaceValidationError(
                "RemoteTransform.include_embodia_metadata must be a bool."
            )

    def build_obs(self, frame: Frame | Mapping[str, Any]) -> dict[str, Any]:
        """Convert one embodia frame into one remote observation payload."""

        normalized_frame = coerce_frame(frame)
        validate_frame(normalized_frame)
        if self.frame_to_obs_fn is None:
            return frame_to_dict(normalized_frame)

        obs = self.frame_to_obs_fn(normalized_frame)
        if not isinstance(obs, Mapping):
            raise InterfaceValidationError(
                "RemoteTransform.frame_to_obs_fn must return a mapping, got "
                f"{type(obs).__name__}."
            )
        return dict(obs)

    def build_frame(self, obs: Mapping[str, Any]) -> Frame:
        """Convert one remote observation mapping into an embodia frame."""

        if not isinstance(obs, Mapping):
            raise InterfaceValidationError(
                f"remote observation must be a mapping, got {type(obs).__name__}."
            )

        if self.obs_to_frame_fn is None:
            frame = coerce_frame(obs)
        else:
            frame = coerce_frame(self.obs_to_frame_fn(obs))
        validate_frame(frame)
        return frame

    def action_plan_from_response(
        self,
        response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    ) -> list[Action]:
        """Convert one remote response into a validated embodia action plan."""

        return actions_to_action_plan(
            response_or_actions,
            target=self.action_target,
            kind=self.command_kind,
            ref_frame=self.ref_frame,
        )

    def first_action_from_response(
        self,
        response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    ) -> Action:
        """Convert one remote response into the first embodia action."""

        return self.action_plan_from_response(response_or_actions)[0]

    def response_from_action_plan(
        self,
        plan: Action | Mapping[str, Any] | Sequence[Action | Mapping[str, Any]],
        *,
        frame: Frame | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert embodia actions into one remote response payload."""

        if self.response_builder is None:
            return response_from_action_plan(
                plan,
                include_embodia_metadata=self.include_embodia_metadata,
            )

        if frame is None:
            raise InterfaceValidationError(
                "RemoteTransform.response_from_action_plan() requires frame=... "
                "when using a custom response_builder."
            )

        normalized_frame = coerce_frame(frame)
        validate_frame(normalized_frame)
        response = self.response_builder(
            _coerce_embodia_action_plan(plan),
            normalized_frame,
        )
        if not isinstance(response, Mapping):
            raise InterfaceValidationError(
                "RemoteTransform.response_builder must return a mapping, got "
                f"{type(response).__name__}."
            )
        return dict(response)


__all__ = [
    "RemoteTransform",
    "actions_to_action_plan",
    "first_action_from_response",
    "response_from_action_plan",
]
