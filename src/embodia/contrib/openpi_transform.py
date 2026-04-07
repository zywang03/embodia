"""OpenPI-specific payload transforms for embodia remote inference."""

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
    """Return the raw OpenPI ``actions`` payload."""

    if isinstance(response_or_actions, Mapping):
        if "actions" not in response_or_actions:
            raise InterfaceValidationError(
                "OpenPI response must contain an 'actions' field."
            )
        return response_or_actions["actions"]
    return response_or_actions


def _coerce_action_rows(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
) -> list[list[float]]:
    """Normalize OpenPI action payloads into ``list[list[float]]``."""

    actions = _coerce_optional_python_list(
        _extract_actions_value(response_or_actions),
        field_name="actions",
    )
    if isinstance(actions, tuple):
        actions = list(actions)

    if not isinstance(actions, list):
        raise InterfaceValidationError(
            "OpenPI actions must be a 1D or 2D list-like numeric payload, got "
            f"{type(actions).__name__}."
        )
    if not actions:
        raise InterfaceValidationError("OpenPI returned an empty action chunk.")

    first = actions[0]
    if isinstance(first, bool):
        raise InterfaceValidationError(
            "OpenPI actions must contain numeric values, not booleans."
        )
    if isinstance(first, Real):
        return [_ensure_float_list(actions, field_name="actions")]

    rows: list[list[float]] = []
    for row_index, row in enumerate(actions):
        rows.append(
            _ensure_float_list(row, field_name=f"actions[{row_index}]")
        )
    return rows


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


def openpi_actions_to_action_plan(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    *,
    target: str = "arm",
    mode: str,
    dt: float = 0.1,
    ref_frame: str | None = None,
) -> list[Action]:
    """Convert an OpenPI action chunk into embodia-standard actions."""

    plan = [
        Action.single(
            target=target,
            mode=mode,
            value=row,
            ref_frame=ref_frame,
            dt=dt,
        )
        for row in _coerce_action_rows(response_or_actions)
    ]
    for action in plan:
        validate_action(action)
    return plan


def openpi_first_action(
    response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    *,
    target: str = "arm",
    mode: str,
    dt: float = 0.1,
    ref_frame: str | None = None,
) -> Action:
    """Convert an OpenPI action chunk and return its first action."""

    return openpi_actions_to_action_plan(
        response_or_actions,
        target=target,
        mode=mode,
        dt=dt,
        ref_frame=ref_frame,
    )[0]


def openpi_response_from_action_plan(
    plan: Action | Mapping[str, Any] | Sequence[Action | Mapping[str, Any]],
    *,
    include_embodia_metadata: bool = True,
) -> dict[str, Any]:
    """Convert embodia actions into an OpenPI-compatible response dict."""

    actions = _coerce_embodia_action_plan(plan)
    commands_per_action = [len(action.commands) for action in actions]
    if any(count != 1 for count in commands_per_action):
        raise InterfaceValidationError(
            "openpi_response_from_action_plan() currently expects one command per "
            "action because OpenPI responses carry one action vector per step."
        )

    first_command = actions[0].commands[0]
    response: dict[str, Any] = {
        "actions": [list(action.commands[0].value) for action in actions],
    }

    if include_embodia_metadata:
        response["embodia"] = {
            "action_target": first_command.target,
            "action_mode": first_command.mode,
            "action_dt": actions[0].dt,
            "action_ref_frame": first_command.ref_frame,
            "chunk_size": len(actions),
        }

    return response


@dataclass(slots=True)
class OpenPITransform:
    """Bundle OpenPI payload conversion in one small object.

    This keeps robot/model code focused on one transform object instead of
    scattering OpenPI-specific conversion details across multiple callbacks.
    """

    action_mode: str
    action_target: str = "arm"
    dt: float = 0.1
    ref_frame: str | None = None
    frame_to_obs_fn: Callable[[Frame], Mapping[str, Any]] | None = None
    obs_to_frame_fn: Callable[[Mapping[str, Any]], Frame | Mapping[str, Any]] | None = None
    response_builder: Callable[[list[Action], Frame], Mapping[str, Any]] | None = None
    include_embodia_metadata: bool = True

    def __post_init__(self) -> None:
        """Validate configuration early for clearer runtime errors."""

        if not isinstance(self.action_target, str) or not self.action_target.strip():
            raise InterfaceValidationError(
                "OpenPITransform.action_target must be a non-empty string."
            )
        if not isinstance(self.action_mode, str) or not self.action_mode.strip():
            raise InterfaceValidationError(
                "OpenPITransform.action_mode must be a non-empty string."
            )
        if isinstance(self.dt, bool) or not isinstance(self.dt, Real):
            raise InterfaceValidationError(
                "OpenPITransform.dt must be a real number."
            )
        self.dt = float(self.dt)
        if self.dt <= 0.0:
            raise InterfaceValidationError("OpenPITransform.dt must be > 0.")
        if self.ref_frame is not None and (
            not isinstance(self.ref_frame, str) or not self.ref_frame.strip()
        ):
            raise InterfaceValidationError(
                "OpenPITransform.ref_frame must be a non-empty string when provided."
            )
        if self.frame_to_obs_fn is not None and not callable(self.frame_to_obs_fn):
            raise InterfaceValidationError(
                "OpenPITransform.frame_to_obs_fn must be callable when provided."
            )
        if self.obs_to_frame_fn is not None and not callable(self.obs_to_frame_fn):
            raise InterfaceValidationError(
                "OpenPITransform.obs_to_frame_fn must be callable when provided."
            )
        if self.response_builder is not None and not callable(self.response_builder):
            raise InterfaceValidationError(
                "OpenPITransform.response_builder must be callable when provided."
            )
        if not isinstance(self.include_embodia_metadata, bool):
            raise InterfaceValidationError(
                "OpenPITransform.include_embodia_metadata must be a bool."
            )

    def build_obs(self, frame: Frame | Mapping[str, Any]) -> dict[str, Any]:
        """Convert one embodia frame into an OpenPI observation payload."""

        normalized_frame = coerce_frame(frame)
        validate_frame(normalized_frame)
        if self.frame_to_obs_fn is None:
            return frame_to_dict(normalized_frame)

        obs = self.frame_to_obs_fn(normalized_frame)
        if not isinstance(obs, Mapping):
            raise InterfaceValidationError(
                "OpenPITransform.frame_to_obs_fn must return a mapping, got "
                f"{type(obs).__name__}."
            )
        return dict(obs)

    def build_frame(self, obs: Mapping[str, Any]) -> Frame:
        """Convert one OpenPI observation mapping into an embodia frame."""

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
        """Convert one OpenPI response into a validated embodia action plan."""

        return openpi_actions_to_action_plan(
            response_or_actions,
            target=self.action_target,
            mode=self.action_mode,
            dt=self.dt,
            ref_frame=self.ref_frame,
        )

    def first_action_from_response(
        self,
        response_or_actions: Mapping[str, Any] | Sequence[Any] | object,
    ) -> Action:
        """Convert one OpenPI response into the first embodia action."""

        return self.action_plan_from_response(response_or_actions)[0]

    def response_from_action_plan(
        self,
        plan: Action | Mapping[str, Any] | Sequence[Action | Mapping[str, Any]],
        *,
        frame: Frame | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert embodia actions into an OpenPI response payload."""

        if self.response_builder is None:
            return openpi_response_from_action_plan(
                plan,
                include_embodia_metadata=self.include_embodia_metadata,
            )

        if frame is None:
            raise InterfaceValidationError(
                "OpenPITransform.response_from_action_plan() requires frame=... "
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
                "OpenPITransform.response_builder must return a mapping, got "
                f"{type(response).__name__}."
            )
        return dict(response)


__all__ = [
    "OpenPITransform",
    "openpi_actions_to_action_plan",
    "openpi_first_action",
    "openpi_response_from_action_plan",
]
