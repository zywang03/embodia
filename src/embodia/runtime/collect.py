"""Helpers for standardized runtime data collection on top of embodia.

This module is intentionally small. The goal is to keep the public flow easy to
read:

- normalize and validate frames
- normalize and validate actions
- collect those into a lightweight episode structure
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.modalities import action_modes, images, state
from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Frame, ModelSpec, RobotSpec
from ..core.transform import (
    action_to_dict,
    coerce_action,
    coerce_frame,
    coerce_model_spec,
    coerce_robot_spec,
    frame_to_dict,
    model_spec_to_dict,
    robot_spec_to_dict,
)
from .checks import (
    _pair_problems,
    validate_action,
    validate_frame,
    validate_model_spec,
    validate_robot_spec,
)

ActionSource = Callable[[Frame], Action | Mapping[str, Any] | None]


@dataclass(slots=True)
class EpisodeStep:
    """One standardized collection step."""

    frame: Frame
    action: Action | None = None
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class Episode:
    """A small, normalized episode container for rollout or data collection."""

    robot_spec: RobotSpec
    steps: list[EpisodeStep] = field(default_factory=list)
    model_spec: ModelSpec | None = None
    meta: dict[str, Any] | None = None


def _copy_optional_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> dict[str, Any] | None:
    """Copy optional metadata while enforcing ``dict[str, Any]`` shape."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"{field_name} must be a mapping with string keys, "
            f"got {type(value).__name__}."
        )

    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise InterfaceValidationError(
                f"{field_name} keys must be strings, got {key!r}."
            )
        result[key] = item
    return result


def _call(obj: object, method_name: str, *args: object) -> Any:
    """Call a required method with clear runtime errors."""

    method = getattr(obj, method_name, None)
    if not callable(method):
        raise InterfaceValidationError(
            f"{type(obj).__name__} instance is missing callable method "
            f"{method_name!r}."
        )

    try:
        return method(*args)
    except Exception as exc:
        raise InterfaceValidationError(
            f"{type(obj).__name__} instance {method_name}() raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _load_robot_spec(robot: object) -> RobotSpec:
    """Load and validate a robot spec."""

    spec = coerce_robot_spec(_call(robot, "get_spec"))
    validate_robot_spec(spec)
    return spec


def _load_model_spec(model: object) -> ModelSpec:
    """Load and validate a model spec."""

    spec = coerce_model_spec(_call(model, "get_spec"))
    validate_model_spec(spec)
    return spec


def _normalize_frame(
    value: Frame | Mapping[str, Any],
    *,
    robot_spec: RobotSpec,
    context: str,
) -> Frame:
    """Normalize a frame and ensure it satisfies the robot declaration."""

    frame = coerce_frame(value)
    validate_frame(frame)

    images.ensure_frame_keys(
        frame,
        robot_spec.image_keys,
        owner_label="robot",
        owner_name=robot_spec.name,
        context=context,
    )
    state.ensure_frame_keys(
        frame,
        robot_spec.state_keys,
        owner_label="robot",
        owner_name=robot_spec.name,
        context=context,
    )

    return frame


def _normalize_action(
    value: Action | Mapping[str, Any],
    *,
    robot_spec: RobotSpec,
    model_spec: ModelSpec | None = None,
) -> Action:
    """Normalize an action and ensure it matches declared specs."""

    action = coerce_action(value)
    validate_action(action)

    action_modes.ensure_supported(
        action,
        robot_spec.action_modes,
        owner_label="robot",
        owner_name=robot_spec.name,
    )

    if model_spec is not None:
        action_modes.ensure_model_output(
            action,
            output_mode=model_spec.output_action_mode,
            model_name=model_spec.name,
        )

    return action


def episode_step_to_dict(step: EpisodeStep) -> dict[str, Any]:
    """Export an :class:`EpisodeStep` into a plain dictionary."""

    if not isinstance(step, EpisodeStep):
        raise InterfaceValidationError(
            f"step must be an EpisodeStep instance, got {type(step).__name__}."
        )

    return {
        "frame": frame_to_dict(step.frame),
        "action": None if step.action is None else action_to_dict(step.action),
        "meta": None if step.meta is None else dict(step.meta),
    }


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    """Export an :class:`Episode` into a plain dictionary."""

    if not isinstance(episode, Episode):
        raise InterfaceValidationError(
            f"episode must be an Episode instance, got {type(episode).__name__}."
        )

    return {
        "robot_spec": robot_spec_to_dict(episode.robot_spec),
        "model_spec": (
            None
            if episode.model_spec is None
            else model_spec_to_dict(episode.model_spec)
        ),
        "steps": [episode_step_to_dict(step) for step in episode.steps],
        "meta": None if episode.meta is None else dict(episode.meta),
    }


def record_step(
    robot: object,
    *,
    frame: Frame | Mapping[str, Any] | None = None,
    action: Action | Mapping[str, Any] | None = None,
    execute_action: bool = False,
    step_meta: Mapping[str, Any] | None = None,
) -> EpisodeStep:
    """Collect one normalized robot step.

    If ``frame`` is omitted, embodia reads ``robot.observe()``.
    If ``action`` is provided, embodia validates it and can optionally forward
    it to ``robot.act(...)``.
    """

    robot_spec = _load_robot_spec(robot)
    raw_frame = _call(robot, "observe") if frame is None else frame
    normalized_frame = _normalize_frame(
        raw_frame,
        robot_spec=robot_spec,
        context="collected frame",
    )

    normalized_action: Action | None = None
    if action is not None:
        normalized_action = _normalize_action(action, robot_spec=robot_spec)
        if execute_action:
            _call(robot, "act", normalized_action)

    return EpisodeStep(
        frame=normalized_frame,
        action=normalized_action,
        meta=_copy_optional_mapping(step_meta, "step_meta"),
    )


def collect_episode(
    robot: object,
    *,
    steps: int,
    model: object | None = None,
    action_fn: ActionSource | None = None,
    execute_actions: bool = False,
    reset_robot: bool = False,
    reset_model: bool = False,
    include_reset_frame: bool = False,
    episode_meta: Mapping[str, Any] | None = None,
) -> Episode:
    """Collect a small standardized episode.

    Supported modes:

    - robot-only passive collection
    - robot + external action source
    - robot + model rollout collection
    """

    if isinstance(steps, bool) or not isinstance(steps, int):
        raise InterfaceValidationError(
            f"steps must be an int >= 0, got {type(steps).__name__}."
        )
    if steps < 0:
        raise InterfaceValidationError("steps must be >= 0.")
    if model is not None and action_fn is not None:
        raise InterfaceValidationError(
            "collect_episode() accepts either model=... or action_fn=..., not both."
        )
    if reset_model and model is None:
        raise InterfaceValidationError(
            "reset_model=True requires model=... to be provided."
        )

    robot_spec = _load_robot_spec(robot)
    model_spec: ModelSpec | None = None
    if model is not None:
        model_spec = _load_model_spec(model)
        problems = _pair_problems(robot_spec, model_spec)
        if problems:
            raise InterfaceValidationError(
                "Robot/model pair is incompatible:\n- " + "\n- ".join(problems)
            )
        if reset_model:
            reset_result = _call(model, "reset")
            if reset_result is not None:
                raise InterfaceValidationError(
                    f"model reset() must return None, got "
                    f"{type(reset_result).__name__}."
                )

    collected_steps: list[EpisodeStep] = []

    if reset_robot:
        reset_frame = _normalize_frame(
            _call(robot, "reset"),
            robot_spec=robot_spec,
            context="reset frame",
        )
        if include_reset_frame:
            collected_steps.append(
                EpisodeStep(
                    frame=reset_frame,
                    meta={"step_index": -1, "source": "reset"},
                )
            )
    elif include_reset_frame:
        initial_frame = _normalize_frame(
            _call(robot, "observe"),
            robot_spec=robot_spec,
            context="initial frame",
        )
        collected_steps.append(
            EpisodeStep(
                frame=initial_frame,
                meta={"step_index": -1, "source": "initial_observe"},
            )
        )

    for step_index in range(steps):
        frame = _normalize_frame(
            _call(robot, "observe"),
            robot_spec=robot_spec,
            context="collected frame",
        )

        action: Action | None = None
        if model is not None:
            action = _normalize_action(
                _call(model, "step", frame),
                robot_spec=robot_spec,
                model_spec=model_spec,
            )
        elif action_fn is not None:
            try:
                raw_action = action_fn(frame)
            except Exception as exc:
                raise InterfaceValidationError(
                    f"action_fn(frame) raised {type(exc).__name__}: {exc}"
                ) from exc
            if raw_action is not None:
                action = _normalize_action(raw_action, robot_spec=robot_spec)

        if execute_actions and action is not None:
            _call(robot, "act", action)

        collected_steps.append(
            EpisodeStep(
                frame=frame,
                action=action,
                meta={"step_index": step_index},
            )
        )

    return Episode(
        robot_spec=robot_spec,
        model_spec=model_spec,
        steps=collected_steps,
        meta=_copy_optional_mapping(episode_meta, "episode_meta"),
    )


__all__ = [
    "ActionSource",
    "Episode",
    "EpisodeStep",
    "collect_episode",
    "episode_step_to_dict",
    "episode_to_dict",
    "record_step",
]
