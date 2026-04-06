"""Public package exports for embodia."""

from .core.errors import InterfaceValidationError
from .core.mixins import ModelMixin, RobotMixin
from .core.modalities import ACTION_MODES, IMAGE_KEYS, STATE_KEYS, ModalityToken
from .core.protocols import ModelProtocol, RobotProtocol
from .core.schema import Action, ActionMode, Frame, ModelSpec, RobotSpec
from .core.transform import (
    action_to_dict,
    coerce_action,
    coerce_frame,
    coerce_model_spec,
    coerce_robot_spec,
    frame_to_dict,
    invert_mapping,
    model_spec_to_dict,
    remap_action,
    remap_frame,
    remap_mapping_keys,
    remap_model_spec,
    remap_robot_spec,
    robot_spec_to_dict,
)
from .runtime.checks import (
    check_model,
    check_pair,
    check_robot,
    validate_action,
    validate_frame,
    validate_model_spec,
    validate_robot_spec,
)
from .runtime.collect import (
    Episode,
    EpisodeStep,
    collect_episode,
    episode_step_to_dict,
    episode_to_dict,
    record_step,
)
from .runtime.flow import StepResult, run_step
from .runtime.h5 import (
    H5_FORMAT,
    is_h5_available,
    load_episode_h5,
    require_h5,
    save_episode_h5,
)

__all__ = [
    "Action",
    "ActionMode",
    "ACTION_MODES",
    "collect_episode",
    "Frame",
    "Episode",
    "EpisodeStep",
    "H5_FORMAT",
    "IMAGE_KEYS",
    "InterfaceValidationError",
    "ModelMixin",
    "ModelProtocol",
    "ModelSpec",
    "RobotMixin",
    "RobotProtocol",
    "RobotSpec",
    "ModalityToken",
    "STATE_KEYS",
    "action_to_dict",
    "StepResult",
    "check_model",
    "check_pair",
    "check_robot",
    "episode_step_to_dict",
    "episode_to_dict",
    "coerce_action",
    "coerce_frame",
    "coerce_model_spec",
    "coerce_robot_spec",
    "frame_to_dict",
    "invert_mapping",
    "is_h5_available",
    "load_episode_h5",
    "model_spec_to_dict",
    "remap_action",
    "remap_frame",
    "remap_mapping_keys",
    "remap_model_spec",
    "remap_robot_spec",
    "record_step",
    "require_h5",
    "robot_spec_to_dict",
    "run_step",
    "save_episode_h5",
    "validate_action",
    "validate_frame",
    "validate_model_spec",
    "validate_robot_spec",
]
