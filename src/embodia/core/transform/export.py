"""Export helpers for embodia schema objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schema import Action, Frame, ModelSpec, RobotSpec
from .coerce import coerce_action, coerce_frame, coerce_model_spec, coerce_robot_spec


def frame_to_dict(frame: Frame | Mapping[str, Any]) -> dict[str, Any]:
    """Export a frame-like object into a shallow plain dictionary."""

    normalized = coerce_frame(frame)
    return {
        "timestamp_ns": normalized.timestamp_ns,
        "images": dict(normalized.images),
        "state": dict(normalized.state),
        "task": None if normalized.task is None else dict(normalized.task),
        "meta": None if normalized.meta is None else dict(normalized.meta),
    }


def action_to_dict(action: Action | Mapping[str, Any]) -> dict[str, Any]:
    """Export an action-like object into a shallow plain dictionary."""

    normalized = coerce_action(action)
    return {
        "mode": normalized.mode,
        "value": list(normalized.value),
        "gripper": normalized.gripper,
        "ref_frame": normalized.ref_frame,
        "dt": normalized.dt,
    }


def robot_spec_to_dict(spec: RobotSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a robot spec-like object into a shallow plain dictionary."""

    normalized = coerce_robot_spec(spec)
    return {
        "name": normalized.name,
        "action_modes": list(normalized.action_modes),
        "image_keys": list(normalized.image_keys),
        "state_keys": list(normalized.state_keys),
    }


def model_spec_to_dict(spec: ModelSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Export a model spec-like object into a shallow plain dictionary."""

    normalized = coerce_model_spec(spec)
    return {
        "name": normalized.name,
        "required_image_keys": list(normalized.required_image_keys),
        "required_state_keys": list(normalized.required_state_keys),
        "output_action_mode": normalized.output_action_mode,
    }


__all__ = [
    "action_to_dict",
    "frame_to_dict",
    "model_spec_to_dict",
    "robot_spec_to_dict",
]
