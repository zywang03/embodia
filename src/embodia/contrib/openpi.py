"""Optional OpenPI compatibility helpers built on embodia's remote transport.

This module keeps OpenPI-specific request/response adaptation out of embodia's
core runtime. It lets a local embodia-wrapped robot talk directly to a remote
OpenPI policy server without requiring the remote policy to inherit
``PolicyMixin``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import InterfaceValidationError
from ..core.schema import Action, Command, Frame
from ..core.transform import coerce_action, coerce_frame, coerce_policy_spec
from ..runtime.checks import validate_action, validate_frame
from ..runtime.inference.protocols import ChunkRequest
from .remote_transform import _coerce_action_rows, _coerce_embodia_action_plan
from .remote_transport import RemotePolicyRunner


ActionSelector = int | slice | tuple[int, int] | Sequence[int]


@dataclass(slots=True)
class OpenPIActionGroup:
    """Describe how one OpenPI action vector maps into one embodia command."""

    target: str
    kind: str
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
        if not isinstance(group.kind, str) or not group.kind.strip():
            raise InterfaceValidationError(
                f"action_groups[{index}].kind must be a non-empty string."
            )
        if group.target in seen_targets:
            raise InterfaceValidationError(
                f"action_groups contains duplicate target {group.target!r}."
            )
        seen_targets.add(group.target)
        normalized.append(
            OpenPIActionGroup(
                target=group.target,
                kind=group.kind,
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
    """Build one flat OpenPI observation dictionary from an embodia frame.

    This helper is intentionally small and only covers straightforward top-level
    key remapping. For nested OpenPI observation payloads, pass your own
    ``obs_builder(frame)`` callable to :class:`OpenPIPolicySource`.
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


def openpi_action_plan_from_response(
    response_or_actions: object,
    *,
    action_groups: Sequence[OpenPIActionGroup],
) -> list[Action]:
    """Convert one OpenPI-style action chunk into embodia actions."""

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
                kind=group.kind,
                value=[row[item_index] for item_index in indices],
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
    """Convert one OpenPI response and return the first embodia action."""

    return openpi_action_plan_from_response(
        response_or_actions,
        action_groups=action_groups,
    )[0]


class OpenPIPolicySource:
    """Embodia action source for a remote OpenPI policy server.

    The remote side only needs to speak OpenPI's websocket inference protocol.
    It does not need to depend on embodia or inherit ``PolicyMixin``.
    """

    def __init__(
        self,
        *,
        runner: object | None = None,
        host: str = "localhost",
        port: int | None = None,
        api_key: str | None = None,
        retry_interval_s: float = 5.0,
        connect_timeout_s: float | None = None,
        additional_headers: Mapping[str, str] | None = None,
        connect_immediately: bool = False,
        wait_for_server: bool = True,
        obs_builder: Callable[[Frame], Mapping[str, Any]],
        response_to_action: Callable[[object], Action | Mapping[str, Any]] | None = None,
        response_to_action_plan: Callable[[object], object] | None = None,
        action_groups: Sequence[OpenPIActionGroup] | None = None,
        policy_spec: object | None = None,
        enabled: bool = True,
    ) -> None:
        if not callable(obs_builder):
            raise InterfaceValidationError(
                "OpenPIPolicySource requires obs_builder(frame) to be callable."
            )
        if action_groups is not None and (
            response_to_action is not None or response_to_action_plan is not None
        ):
            raise InterfaceValidationError(
                "OpenPIPolicySource accepts action_groups=... or "
                "response_to_action=/response_to_action_plan=..., not both."
            )
        if response_to_action is not None and response_to_action_plan is not None:
            raise InterfaceValidationError(
                "OpenPIPolicySource accepts response_to_action=... or "
                "response_to_action_plan=..., not both."
            )

        self._runner = (
            runner
            if runner is not None
            else RemotePolicyRunner(
                enabled=enabled,
                host=host,
                port=port,
                api_key=api_key,
                retry_interval_s=retry_interval_s,
                connect_timeout_s=connect_timeout_s,
                additional_headers=additional_headers,
                connect_immediately=connect_immediately,
                wait_for_server=wait_for_server,
            )
        )
        infer = getattr(self._runner, "infer", None)
        if not callable(infer):
            raise InterfaceValidationError(
                "OpenPIPolicySource runner must expose infer(obs)."
            )

        self._obs_builder = obs_builder
        self._policy_spec = (
            None if policy_spec is None else coerce_policy_spec(policy_spec)
        )
        self._action_groups = (
            None if action_groups is None else _validate_action_groups(action_groups)
        )
        self._response_to_action = response_to_action
        self._response_to_action_plan = response_to_action_plan

        if (
            self._response_to_action is None
            and self._response_to_action_plan is None
            and self._action_groups is None
        ):
            raise InterfaceValidationError(
                "OpenPIPolicySource requires action_groups=..., "
                "response_to_action=..., or response_to_action_plan=...."
            )

    def _build_obs(self, frame: Frame | Mapping[str, Any]) -> dict[str, Any]:
        """Build one OpenPI observation payload."""

        normalized_frame = coerce_frame(frame)
        validate_frame(normalized_frame)
        obs = self._obs_builder(normalized_frame)
        if not isinstance(obs, Mapping):
            raise InterfaceValidationError(
                "obs_builder(frame) must return a mapping, got "
                f"{type(obs).__name__}."
            )
        return dict(obs)

    def _decode_plan(self, response: object) -> list[Action]:
        """Decode one OpenPI response into a validated action plan."""

        if self._response_to_action_plan is not None:
            plan = _coerce_embodia_action_plan(self._response_to_action_plan(response))
        elif self._response_to_action is not None:
            plan = [coerce_action(self._response_to_action(response))]
        else:
            assert self._action_groups is not None
            plan = openpi_action_plan_from_response(
                response,
                action_groups=self._action_groups,
            )

        for action in plan:
            validate_action(action)
        return plan

    def infer(self, frame: Frame | Mapping[str, Any]) -> Action:
        """Run one remote OpenPI inference step and return the first action."""

        response = getattr(self._runner, "infer")(self._build_obs(frame))
        return self._decode_plan(response)[0]

    def infer_chunk(
        self,
        frame: Frame | Mapping[str, Any],
        request: ChunkRequest | object,
    ) -> list[Action]:
        """Run one remote OpenPI inference step and return the full action chunk."""

        del request
        response = getattr(self._runner, "infer")(self._build_obs(frame))
        return self._decode_plan(response)

    def reset(self) -> None:
        """Forward reset to the underlying runner when supported."""

        reset = getattr(self._runner, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        """Close the underlying runner when supported."""

        close = getattr(self._runner, "close", None)
        if callable(close):
            close()

    def get_server_metadata(self) -> dict[str, Any]:
        """Return remote server metadata when available."""

        get_server_metadata = getattr(self._runner, "get_server_metadata", None)
        if not callable(get_server_metadata):
            return {}

        metadata = get_server_metadata()
        if not isinstance(metadata, Mapping):
            raise InterfaceValidationError(
                "OpenPIPolicySource runner get_server_metadata() must return a "
                f"mapping, got {type(metadata).__name__}."
            )
        return dict(metadata)

    def get_spec(self) -> Any:
        """Return an optional local policy spec when one was configured.

        OpenPI servers are not expected to publish embodia-native policy specs,
        so this is optional and mainly useful when callers want to run
        ``check_policy`` / ``check_pair`` locally.
        """

        if self._policy_spec is not None:
            return self._policy_spec

        metadata = self.get_server_metadata()
        embodia_metadata = metadata.get("embodia")
        if isinstance(embodia_metadata, Mapping):
            raw_spec = embodia_metadata.get("policy_spec")
            if raw_spec is not None:
                self._policy_spec = coerce_policy_spec(raw_spec)
                return self._policy_spec

        raise InterfaceValidationError(
            "OpenPIPolicySource has no local policy_spec and the remote server "
            "does not publish embodia.policy_spec metadata."
        )


def build_openpi_policy_source(**kwargs: Any) -> OpenPIPolicySource:
    """Convenience wrapper around :class:`OpenPIPolicySource`."""

    return OpenPIPolicySource(**kwargs)


__all__ = [
    "OpenPIActionGroup",
    "OpenPIPolicySource",
    "build_openpi_policy_source",
    "frame_to_openpi_obs",
    "openpi_action_plan_from_response",
    "openpi_first_action_from_response",
]
