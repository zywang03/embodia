"""Scheduler lifecycle helpers for :mod:`inferaxis.runtime.inference.engine`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.errors import InterfaceValidationError
from ...core.schema import Action, Frame
from ...shared.action_source import (
    ActionSource,
    callable_key,
    first_action_and_plan_length_from_action_call,
)
from .contracts import ChunkRequest
from .engine_config import (
    build_chunk_scheduler_kwargs,
    sync_chunk_scheduler_config,
    should_replace_chunk_scheduler,
)
from .scheduler import ChunkScheduler

if TYPE_CHECKING:
    from .engine import InferenceRuntime


def ensure_chunk_scheduler(
    runtime: "InferenceRuntime",
    *,
    act_src_fn: ActionSource | None,
) -> ChunkScheduler | None:
    """Create or reuse the hidden chunk scheduler when needed."""

    scheduler_key = callable_key(act_src_fn)
    source_is_single_step = scheduler_key in runtime._single_step_source_keys
    if runtime.mode != "async" and source_is_single_step:
        return None
    if runtime.mode == "async" and source_is_single_step:
        raise InterfaceValidationError(
            "InferenceRuntime(mode=ASYNC) requires act_src_fn=... to return "
            "a chunk with more than one action."
        )

    scheduler = runtime._chunk_scheduler
    if scheduler is not None:
        if should_replace_chunk_scheduler(
            runtime,
            scheduler,
            scheduler_key=scheduler_key,
            source_is_single_step=source_is_single_step,
        ):
            scheduler.close()
            runtime._chunk_scheduler = None
            runtime._chunk_scheduler_key = None
        else:
            sync_chunk_scheduler_config(runtime, scheduler)
            return scheduler

    if act_src_fn is None:
        raise InterfaceValidationError(
            "InferenceRuntime async chunk scheduling requires act_src_fn=...."
        )

    scheduler = ChunkScheduler(
        **build_chunk_scheduler_kwargs(
            runtime,
            action_source=act_src_fn,
        )
    )
    runtime._chunk_scheduler = scheduler
    runtime._chunk_scheduler_key = scheduler_key
    return scheduler


def default_request() -> ChunkRequest:
    """Build the direct-call request used by non-scheduled sources."""

    return ChunkRequest(
        request_step=0,
        request_time_s=0.0,
        active_chunk_length=0,
        remaining_steps=0,
        latency_steps=0,
    )


def bootstrap_chunk_scheduler(
    runtime: "InferenceRuntime",
    *,
    frame: Frame,
    chunk_scheduler: ChunkScheduler,
) -> bool:
    """Run async bootstrap outside of ``ChunkScheduler.next_action()``."""

    if runtime.mode != "async":
        return False
    return chunk_scheduler.bootstrap(frame, validate_frame_input=False)


def resolve_raw_action(
    runtime: "InferenceRuntime",
    *,
    frame: Frame,
    act_src_fn: ActionSource | None,
    source_key: object | None,
) -> tuple[Action, bool, int]:
    """Resolve one raw action together with plan metadata."""

    known_single_step_source = source_key in runtime._single_step_source_keys
    chunk_scheduler = ensure_chunk_scheduler(runtime, act_src_fn=act_src_fn)
    if chunk_scheduler is None:
        if act_src_fn is None:
            raise InterfaceValidationError("run_step() requires act_src_fn=....")
        raw_action, plan_length = first_action_and_plan_length_from_action_call(
            act_src_fn,
            frame,
            request=default_request(),
        )
        if known_single_step_source and plan_length != 1:
            raise InterfaceValidationError(
                "act_src_fn was previously classified as single-step for "
                f"this runtime, but returned chunk length {plan_length}."
            )
        if source_key is not None and plan_length == 1:
            runtime._single_step_source_keys.add(source_key)
        return raw_action, True, plan_length

    plan_refreshed = bootstrap_chunk_scheduler(
        runtime,
        frame=frame,
        chunk_scheduler=chunk_scheduler,
    )
    raw_action, next_plan_refreshed = chunk_scheduler.next_action(
        frame,
        prefetch_async=runtime.mode == "async",
        validate_frame_input=False,
    )
    plan_refreshed = plan_refreshed or next_plan_refreshed
    plan_length = chunk_scheduler.active_source_plan_length
    if plan_length == 1 and source_key is not None:
        runtime._single_step_source_keys.add(source_key)
        chunk_scheduler.close()
        runtime._chunk_scheduler = None
        runtime._chunk_scheduler_key = None
        if runtime.mode == "async":
            raise InterfaceValidationError(
                "InferenceRuntime(mode=ASYNC) requires act_src_fn=... "
                "to return a chunk with more than one action."
            )
    return raw_action, plan_refreshed, plan_length
