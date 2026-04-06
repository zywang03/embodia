"""Runtime validation and compatibility checks."""

from .checks import (
    InterfaceValidationError,
    check_model,
    check_pair,
    check_robot,
    validate_action,
    validate_frame,
    validate_model_spec,
    validate_robot_spec,
)
from .collect import (
    Episode,
    EpisodeStep,
    collect_episode,
    episode_step_to_dict,
    episode_to_dict,
    record_step,
)
from .flow import StepResult, run_step
from .h5 import H5_FORMAT, is_h5_available, load_episode_h5, require_h5, save_episode_h5

__all__ = [
    "Episode",
    "EpisodeStep",
    "H5_FORMAT",
    "InterfaceValidationError",
    "StepResult",
    "check_model",
    "check_pair",
    "check_robot",
    "collect_episode",
    "episode_step_to_dict",
    "episode_to_dict",
    "is_h5_available",
    "load_episode_h5",
    "record_step",
    "require_h5",
    "run_step",
    "save_episode_h5",
    "validate_action",
    "validate_frame",
    "validate_model_spec",
    "validate_robot_spec",
]
