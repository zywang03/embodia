"""Public package exports for inferaxis.

The root package intentionally exposes only the small day-to-day API.
Advanced schema and validation helpers stay in submodules.
"""

from .core.errors import InterfaceValidationError
from .core.schema import Action, BuiltinCommandKind, Command, Frame, PolicySpec, RobotSpec
from .core.transform import action_to_dict, coerce_action, coerce_frame, frame_to_dict
from .runtime.checks import check_pair
from .runtime.flow import run_step
from .runtime.inference import (
    ActionEnsembler,
    ActionInterpolator,
    ChunkRequest,
    InferenceMode,
    InferenceRuntime,
    RealtimeController,
    RtcArgs,
    profile_sync_inference,
    recommend_inference_mode,
)

__all__ = [
    "Action",
    "BuiltinCommandKind",
    "ActionEnsembler",
    "ActionInterpolator",
    "ChunkRequest",
    "Command",
    "Frame",
    "InferenceMode",
    "InferenceRuntime",
    "InterfaceValidationError",
    "PolicySpec",
    "RealtimeController",
    "RtcArgs",
    "RobotSpec",
    "action_to_dict",
    "check_pair",
    "coerce_action",
    "coerce_frame",
    "frame_to_dict",
    "profile_sync_inference",
    "recommend_inference_mode",
    "run_step",
]
