"""Runtime validation helpers and acceptance checks.

This module intentionally keeps runtime acceptance logic thin. Structural
schema validation lives in :mod:`inferaxis.core.schema`; the functions here focus
on calling user objects safely and verifying that runtime behavior matches the
declared specs.
"""

from __future__ import annotations

import inspect

from ..core.errors import InterfaceValidationError
from ..core.modalities import images, state, task
from ..core.schema import (
    Action,
    Frame,
    PolicySpec,
    RobotSpec,
    ensure_action_matches_policy_spec,
    ensure_action_supported_by_robot,
    validate_action,
    validate_component_spec,
    validate_command,
    validate_frame,
    validate_policy_output_spec,
    validate_policy_spec,
    validate_robot_spec,
)
from ..core.transform import coerce_policy_spec, coerce_robot_spec
from ..shared.coerce import as_frame as _as_frame
from ..shared.sequence import attach_runtime_frame_metadata


def _object_label(obj: object) -> str:
    """Return a helpful label for runtime error messages."""

    return f"{type(obj).__name__} instance"


def _require_method(obj: object, method_name: str) -> object:
    """Return one required callable method by exact name."""

    method = getattr(obj, method_name, None)
    if callable(method):
        return method
    raise InterfaceValidationError(
        f"{_object_label(obj)} is missing required method {method_name}(...)."
    )


def _optional_method(obj: object, method_name: str) -> object | None:
    """Return one optional callable method by exact name."""

    method = getattr(obj, method_name, None)
    return method if callable(method) else None


def _call_method(
    method: object, obj: object, method_name: str, *args: object
) -> object:
    """Call a checked method and wrap runtime errors consistently."""

    assert callable(method)
    try:
        return method(*args)
    except Exception as exc:
        raise InterfaceValidationError(
            f"{_object_label(obj)} {method_name}() raised {type(exc).__name__}: {exc}"
        ) from exc


def _ensure_signature_accepts(method: object, method_name: str, *args: object) -> None:
    """Check that a method can be called with the expected runtime arguments."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        raise InterfaceValidationError(
            f"Could not inspect signature of method {method_name!r}: {exc}."
        ) from exc

    try:
        signature.bind(*args)
    except TypeError as exc:
        raise InterfaceValidationError(
            f"Method {method_name!r} has incompatible signature {signature}; "
            f"it must accept {len(args)} runtime argument(s)."
        ) from exc


def _pair_problems(robot_spec: RobotSpec, policy_spec: PolicySpec) -> list[str]:
    """Return compatibility problems between a robot spec and a policy spec."""

    problems: list[str] = []

    image_problem = images.pair_problem(
        available_keys=robot_spec.image_keys,
        required_keys=policy_spec.required_image_keys,
    )
    if image_problem is not None:
        problems.append(image_problem)

    state_problem = state.pair_problem(
        available_keys=robot_spec.all_state_keys(),
        required_keys=policy_spec.required_state_keys,
    )
    if state_problem is not None:
        problems.append(state_problem)

    for output in policy_spec.outputs:
        component = robot_spec.get_component(output.target)
        if component is None:
            problems.append(f"robot is missing required component {output.target!r}.")
            continue
        if output.command not in component.command:
            problems.append(
                f"component {output.target!r} does not support policy output "
                f"command {output.command!r}; supported commands: "
                f"{component.command!r}."
            )
        if output.dim != component.dof:
            problems.append(
                f"component {output.target!r} has dof={component.dof}, but policy "
                f"declares dim={output.dim}."
            )

    return problems


def _validate_robot_frame(spec: RobotSpec, frame: Frame) -> None:
    """Validate one robot-produced frame against the declared robot spec."""

    validate_frame(frame)
    images.ensure_frame_keys(frame, spec.image_keys, owner_label="robot")
    state.ensure_frame_keys(frame, spec.all_state_keys(), owner_label="robot")


def check_robot(
    robot: object,
    *,
    sample_frame: Frame | None = None,
) -> None:
    """Runtime-check whether an object is a compatible robot implementation.

    This is a dry-run validation helper: it checks declared methods and, when
    needed, requests at most one observation frame. It never calls
    ``send_action(...)``.
    """

    get_spec_name = "get_spec"
    observe_name = "get_obs"
    act_name = "send_action"
    reset_name = "reset"
    get_spec = _require_method(robot, get_spec_name)
    observe = _require_method(robot, observe_name)
    act = _require_method(robot, act_name)
    reset = _optional_method(robot, reset_name)

    _ensure_signature_accepts(get_spec, get_spec_name)
    _ensure_signature_accepts(observe, observe_name)
    _ensure_signature_accepts(act, act_name, object())
    if callable(reset):
        _ensure_signature_accepts(reset, reset_name)

    raw_spec = _call_method(get_spec, robot, get_spec_name)
    try:
        spec = coerce_robot_spec(raw_spec)
    except InterfaceValidationError as exc:
        raise InterfaceValidationError(
            f"{_object_label(robot)} {get_spec_name}() must return RobotSpec or "
            f"robot-spec-like mapping: {exc}"
        ) from exc
    validate_robot_spec(spec)

    if sample_frame is not None:
        _validate_robot_frame(spec, _as_frame(sample_frame))
        return

    raw_frame = _call_method(observe, robot, observe_name)
    try:
        frame = _as_frame(raw_frame)
    except TypeError as exc:
        raise InterfaceValidationError(
            f"{_object_label(robot)} {observe_name}() must return Frame, got "
            f"{type(raw_frame).__name__}."
        ) from exc
    _validate_robot_frame(spec, frame)


def check_policy(
    policy: object,
    *,
    sample_frame: Frame | None = None,
) -> None:
    """Runtime-check whether an object is a compatible policy implementation.

    This is a dry-run validation helper: it checks declared methods and, when
    ``sample_frame`` is provided, issues exactly one ``infer(frame, request)``
    call. It never calls any execution path and does not call ``reset()``.
    """

    get_spec_name = "get_spec"
    reset_name = "reset"
    infer_name = "infer"
    get_spec = _require_method(policy, get_spec_name)
    reset = _optional_method(policy, reset_name)
    infer = _require_method(policy, infer_name)

    _ensure_signature_accepts(get_spec, get_spec_name)
    if callable(reset):
        _ensure_signature_accepts(reset, reset_name)
    _ensure_signature_accepts(infer, infer_name, object(), object())

    raw_spec = _call_method(get_spec, policy, get_spec_name)
    try:
        spec = coerce_policy_spec(raw_spec)
    except InterfaceValidationError as exc:
        raise InterfaceValidationError(
            f"{_object_label(policy)} {get_spec_name}() must return PolicySpec or "
            f"policy-spec-like mapping: {exc}"
        ) from exc
    validate_policy_spec(spec)

    if sample_frame is None:
        return

    normalized_frame = _as_frame(sample_frame)
    validate_frame(normalized_frame)
    images.ensure_frame_keys(
        normalized_frame,
        spec.required_image_keys,
        owner_label="policy",
    )
    state.ensure_frame_keys(
        normalized_frame,
        spec.required_state_keys,
        owner_label="policy",
    )
    task.ensure_frame_keys(
        normalized_frame,
        spec.required_task_keys,
        owner_label="policy",
    )
    from ..runtime.inference.contracts import ChunkRequest

    request = ChunkRequest(
        request_step=0,
        request_time_s=0.0,
        active_chunk_length=0,
        remaining_steps=0,
        latency_steps=0,
    )
    plan = _call_method(infer, policy, infer_name, normalized_frame, request)
    if isinstance(plan, Action):
        actions = [plan]
    elif isinstance(plan, list):
        actions = plan
    else:
        raise InterfaceValidationError(
            f"{_object_label(policy)} {infer_name}(frame, request) must "
            f"return Action or list[Action], got {type(plan).__name__}."
        )
    if not actions:
        raise InterfaceValidationError(
            f"{_object_label(policy)} {infer_name}(frame, request) must not "
            "return an empty chunk."
        )
    for index, raw_action in enumerate(actions):
        if not isinstance(raw_action, Action):
            raise InterfaceValidationError(
                f"{_object_label(policy)} {infer_name}(frame, request) returned "
                f"non-action item at index {index}: {type(raw_action).__name__}."
            )
        action = raw_action
        validate_action(action)
        ensure_action_matches_policy_spec(action, spec)


def check_pair(
    executor: object,
    policy: object,
    *,
    sample_frame: Frame | None = None,
) -> None:
    """Validate that a local executor/frame and a policy are compatible.

    This is a dry-run pair check. It requests at most one observation frame and
    one policy inference result, and never executes actions on the local side.
    """

    normalized_frame = sample_frame
    if normalized_frame is None:
        observe_name = "get_obs"
        observe = _optional_method(executor, observe_name)
        if callable(observe):
            try:
                raw_frame = observe()
            except Exception as exc:
                raise InterfaceValidationError(
                    f"{type(executor).__name__}.{observe_name}() raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            try:
                normalized_frame = attach_runtime_frame_metadata(
                    _as_frame(raw_frame),
                    owner=executor,
                )
            except TypeError as exc:
                raise InterfaceValidationError(
                    f"{type(executor).__name__}.{observe_name}() must return Frame, got "
                    f"{type(raw_frame).__name__}."
                ) from exc
        else:
            raise InterfaceValidationError(
                "check_pair(...) requires sample_frame=... when the local "
                "executor does not expose get_obs(...)."
            )

    normalized_frame = _as_frame(normalized_frame)
    validate_frame(normalized_frame)
    check_policy(policy, sample_frame=normalized_frame)

    robot_get_spec_name = "get_spec"
    robot_get_spec = _optional_method(executor, robot_get_spec_name)
    if not callable(robot_get_spec):
        return

    check_robot(executor, sample_frame=normalized_frame)
    policy_get_spec_name = "get_spec"
    policy_get_spec = _require_method(policy, policy_get_spec_name)

    robot_spec = coerce_robot_spec(
        _call_method(robot_get_spec, executor, robot_get_spec_name)
    )
    policy_spec = coerce_policy_spec(
        _call_method(policy_get_spec, policy, policy_get_spec_name)
    )

    validate_robot_spec(robot_spec)
    validate_policy_spec(policy_spec)

    problems = _pair_problems(robot_spec, policy_spec)
    if problems:
        raise InterfaceValidationError(
            "executor/policy pair is incompatible:\n- " + "\n- ".join(problems)
        )


__all__ = [
    "InterfaceValidationError",
    "check_policy",
    "check_pair",
    "check_robot",
    "ensure_action_matches_policy_spec",
    "ensure_action_supported_by_robot",
    "validate_action",
    "validate_component_spec",
    "validate_command",
    "validate_frame",
    "validate_policy_output_spec",
    "validate_policy_spec",
    "validate_robot_spec",
]
