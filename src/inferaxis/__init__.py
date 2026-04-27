"""Public package exports for inferaxis.

The root package intentionally exposes only the small day-to-day API.
Advanced schema and validation helpers stay in submodules.
"""

from .core.errors import InterfaceValidationError
from .core.schema import (
    Action,
    BuiltinCommandKind,
    Command,
    Frame,
    PolicySpec,
    RobotSpec,
)
from .core.transform import action_to_dict, coerce_action, coerce_frame, frame_to_dict
from .runtime.checks import check_pair
from .runtime.flow import run_step
from .runtime.inference import (
    ChunkRequest,
    InferenceMode,
    InferenceRuntime,
    RealtimeController,
    RtcArgs,
)

__all__ = [
    "Action",
    "BuiltinCommandKind",
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
    "run_step",
]
