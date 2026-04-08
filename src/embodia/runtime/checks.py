"""Runtime validation helpers and acceptance checks.

This module intentionally keeps runtime acceptance logic thin. Structural
schema validation lives in :mod:`embodia.core.schema`; the functions here focus
on calling user objects safely and verifying that runtime behavior matches the
declared specs.
"""

from __future__ import annotations

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
from .shared.dispatch import (
    POLICY_GET_SPEC_METHODS,
    POLICY_INFER_CHUNK_METHODS,
    POLICY_INFER_METHODS,
    POLICY_RESET_METHODS,
    ROBOT_ACT_METHODS,
    ROBOT_GET_SPEC_METHODS,
    ROBOT_OBSERVE_METHODS,
    ROBOT_RESET_METHODS,
    format_method_options,
    resolve_callable_method,
)
from .shared.check_utils import (
    call_method as _call_method,
    ensure_signature_accepts as _ensure_signature_accepts,
    object_label as _object_label,
    require_method as _require_method,
    single_step_chunk_request as _single_step_chunk_request,
)


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
            problems.append(
                f"robot is missing required component {output.target!r}."
            )
            continue
        if output.command_kind not in component.supported_command_kinds:
            problems.append(
                f"component {output.target!r} does not support policy output "
                f"command kind {output.command_kind!r}; supported command kinds: "
                f"{component.supported_command_kinds!r}."
            )
        if output.dim != component.dof:
            problems.append(
                f"component {output.target!r} has dof={component.dof}, but policy "
                f"declares dim={output.dim}."
            )

    return problems


def check_robot(robot: object, *, call_observe: bool = True) -> None:
    """Runtime-check whether an object is a compatible robot implementation."""

    get_spec, get_spec_name = _require_method(robot, ROBOT_GET_SPEC_METHODS)
    observe, observe_name = _require_method(robot, ROBOT_OBSERVE_METHODS)
    act, act_name = _require_method(robot, ROBOT_ACT_METHODS)
    reset, reset_name = _require_method(robot, ROBOT_RESET_METHODS)

    _ensure_signature_accepts(get_spec, get_spec_name)
    _ensure_signature_accepts(observe, observe_name)
    _ensure_signature_accepts(act, act_name, object())
    _ensure_signature_accepts(reset, reset_name)

    spec = _call_method(get_spec, robot, get_spec_name)
    if not isinstance(spec, RobotSpec):
        raise InterfaceValidationError(
            f"{_object_label(robot)} {get_spec_name}() must return RobotSpec, "
            f"got {type(spec).__name__}."
        )
    validate_robot_spec(spec)

    if not call_observe:
        return

    frame = _call_method(observe, robot, observe_name)
    if not isinstance(frame, Frame):
        raise InterfaceValidationError(
            f"{_object_label(robot)} {observe_name}() must return Frame, "
            f"got {type(frame).__name__}."
        )
    validate_frame(frame)
    images.ensure_frame_keys(frame, spec.image_keys, owner_label="robot")
    state.ensure_frame_keys(frame, spec.all_state_keys(), owner_label="robot")


def check_policy(policy: object, *, sample_frame: Frame | None = None) -> None:
    """Runtime-check whether an object is a compatible policy implementation."""

    get_spec, get_spec_name = _require_method(policy, POLICY_GET_SPEC_METHODS)
    reset, reset_name = _require_method(policy, POLICY_RESET_METHODS)
    infer, infer_name = resolve_callable_method(policy, POLICY_INFER_METHODS)
    infer_chunk, infer_chunk_name = resolve_callable_method(
        policy,
        POLICY_INFER_CHUNK_METHODS,
    )
    if not callable(infer) and not callable(infer_chunk):
        raise InterfaceValidationError(
            f"{_object_label(policy)} must expose "
            f"{format_method_options(POLICY_INFER_METHODS)} or "
            f"{format_method_options(POLICY_INFER_CHUNK_METHODS)}."
        )

    _ensure_signature_accepts(get_spec, get_spec_name)
    _ensure_signature_accepts(reset, reset_name)
    if callable(infer) and infer_name is not None:
        _ensure_signature_accepts(infer, infer_name, object())
    if callable(infer_chunk) and infer_chunk_name is not None:
        _ensure_signature_accepts(infer_chunk, infer_chunk_name, object(), object())

    spec = _call_method(get_spec, policy, get_spec_name)
    if not isinstance(spec, PolicySpec):
        raise InterfaceValidationError(
            f"{_object_label(policy)} {get_spec_name}() must return PolicySpec, "
            f"got {type(spec).__name__}."
        )
    validate_policy_spec(spec)

    reset_result = _call_method(reset, policy, reset_name)
    if reset_result is not None:
        raise InterfaceValidationError(
            f"{_object_label(policy)} {reset_name}() must return None, "
            f"got {type(reset_result).__name__}."
        )

    if sample_frame is None:
        return

    validate_frame(sample_frame)
    images.ensure_frame_keys(
        sample_frame,
        spec.required_image_keys,
        owner_label="policy",
    )
    state.ensure_frame_keys(
        sample_frame,
        spec.required_state_keys,
        owner_label="policy",
    )
    task.ensure_frame_keys(
        sample_frame,
        spec.required_task_keys,
        owner_label="policy",
    )

    if callable(infer) and infer_name is not None:
        action = _call_method(infer, policy, infer_name, sample_frame)
        if not isinstance(action, Action):
            raise InterfaceValidationError(
                f"{_object_label(policy)} {infer_name}(frame) must return Action, "
                f"got {type(action).__name__}."
            )
        validate_action(action)
        ensure_action_matches_policy_spec(action, spec)
        return

    assert callable(infer_chunk) and infer_chunk_name is not None
    plan = _call_method(
        infer_chunk,
        policy,
        infer_chunk_name,
        sample_frame,
        _single_step_chunk_request(),
    )
    if isinstance(plan, Action):
        actions = [plan]
    elif isinstance(plan, list):
        actions = plan
    else:
        raise InterfaceValidationError(
            f"{_object_label(policy)} {infer_chunk_name}(frame, request) must return "
            f"list[Action], got {type(plan).__name__}."
        )
    if not actions:
        raise InterfaceValidationError(
            f"{_object_label(policy)} {infer_chunk_name}(frame, request) must not "
            "return an empty chunk."
        )
    for index, action in enumerate(actions):
        if not isinstance(action, Action):
            raise InterfaceValidationError(
                f"{_object_label(policy)} {infer_chunk_name}(frame, request) returned "
                f"non-Action item at index {index}: {type(action).__name__}."
            )
        validate_action(action)
        ensure_action_matches_policy_spec(action, spec)


def check_pair(
    robot: object,
    policy: object,
    *,
    sample_frame: Frame | None = None,
) -> None:
    """Validate that a robot and a policy are individually valid and compatible."""

    bind_robot = getattr(policy, "embodia_bind_robot", None)
    if callable(bind_robot):
        try:
            bind_robot(robot)
        except Exception as exc:
            raise InterfaceValidationError(
                f"{type(policy).__name__}.embodia_bind_robot(robot) raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    check_robot(robot, call_observe=sample_frame is None)
    check_policy(policy, sample_frame=sample_frame)

    robot_get_spec, robot_get_spec_name = _require_method(robot, ROBOT_GET_SPEC_METHODS)
    policy_get_spec, policy_get_spec_name = _require_method(
        policy,
        POLICY_GET_SPEC_METHODS,
    )

    robot_spec = _call_method(robot_get_spec, robot, robot_get_spec_name)
    policy_spec = _call_method(policy_get_spec, policy, policy_get_spec_name)

    validate_robot_spec(robot_spec)
    validate_policy_spec(policy_spec)

    problems = _pair_problems(robot_spec, policy_spec)
    if problems:
        raise InterfaceValidationError(
            "Robot/policy pair is incompatible:\n- " + "\n- ".join(problems)
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
