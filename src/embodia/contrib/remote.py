"""Generic remote-policy helpers.

This module is the preferred user-facing entrypoint for embodia's remote policy
support. The current implementation is backed by one websocket transport, but
user code can stay generic and only depend on `remote`.
"""

from __future__ import annotations

from .remote_transport import (
    RemoteMsgpackCodec,
    RemotePolicy,
    RemotePolicyRunner,
    RemoteWebsocketClientPolicy as RemotePolicyClient,
    RemoteWebsocketPolicyServer as RemotePolicyServer,
    build_policy_adapter as build_remote_policy_adapter,
    build_policy_server as build_remote_policy_server,
    is_remote_available,
    require_remote,
    serve_policy as serve_remote_policy,
)
from .remote_transform import (
    RemoteTransform,
    actions_to_action_plan,
    first_action_from_response,
    response_from_action_plan,
)

__all__ = [
    "RemoteMsgpackCodec",
    "RemotePolicy",
    "RemotePolicyClient",
    "RemotePolicyRunner",
    "RemotePolicyServer",
    "RemoteTransform",
    "actions_to_action_plan",
    "build_remote_policy_adapter",
    "build_remote_policy_server",
    "first_action_from_response",
    "is_remote_available",
    "require_remote",
    "response_from_action_plan",
    "serve_remote_policy",
]
