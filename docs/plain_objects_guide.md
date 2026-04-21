# inferaxis plain-objects guide

`inferaxis` no longer needs mixins. The intended integration style is now:
plain local objects, explicit method references, one shared runtime schema, and
`run_step(...)` as the outer loop.

## Smallest integration surface

The local side only needs `get_obs()`. Add `send_action()` when you want
closed-loop execution and `reset()` when you want a reset entrypoint. The action
side can be any callable that matches:

- `act_src_fn(frame, request) -> Action | list[Action]`

That means the most direct closed-loop form is:

```python
result = infra.run_step(
    observe_fn=executor.get_obs,
    act_fn=executor.send_action,
    act_src_fn=policy.infer,
)
```

`run_step(...)` accepts any explicit callable you pass as `act_src_fn=...`. If
you use object-level checks such as `check_policy(...)`, the checked method name
is the fixed `infer(...)`. Those checks are dry-run helpers: they validate one
observed frame and one `infer(...)` call, and never call `send_action(...)`.

## Observation and action shape

The shared runtime shape is intentionally small:

```python
frame = infra.Frame(
    images={"front_rgb": np.ndarray(...)},
    state={
        "left_arm": np.ndarray(...),
        "left_gripper": np.ndarray(...),
    },
)

action = infra.Action(
    commands={
        "left_arm": infra.Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=np.ndarray(...),
        ),
        "left_gripper": infra.Command(
            command=infra.BuiltinCommandKind.GRIPPER_POSITION,
            value=np.ndarray(...),
        ),
    },
)
```

`get_obs()` and `reset()` must return `Frame`.
`infer(frame, request)` must return `Action` or `list[Action]`.
Returning one action is treated as chunk size `1`. `send_action(action)`
receives `Action`. Runtime-managed `timestamp_ns` and `sequence_id` are filled
internally by inferaxis.

The dictionary key such as `left_arm` or `left_gripper` is the component name.
The nested `command` field chooses how that component should be controlled.

## Runtime-side features

`InferenceRuntime(...)` stays separate from your own classes. This keeps
overlap blending, async scheduling, overlap management, and loop pacing out of your
policy implementation:

```python
runtime = infra.InferenceRuntime(
    mode=infra.InferenceMode.ASYNC,
    steps_before_request=0,
    warmup_requests=1,
    profile_delay_requests=3,
    realtime_controller=infra.RealtimeController(hz=50.0),
)
```

For `mode=ASYNC`, no manual latency seed is needed anymore. When a
`RealtimeController(...)` is attached, inferaxis first issues request-only
warmup calls for `warmup_requests`, then profiles delay for
`profile_delay_requests`, converts the measured request time into control-step
latency, and only then starts sending actions to the robot. This bootstrap
happens automatically on the first `run_step(...)` call once `observe_fn` and
`act_src_fn` are available.
Because of that startup warmup, `infer(frame, request)` should derive its chunk
from `frame` and `request` instead of relying on mutable "call count" state.
If you want startup warmup/profile to happen outside the first `run_step(...)`
call, invoke `runtime.bootstrap_async(...)` once before entering the loop.

If you use `ASYNC` or `steps_before_request`, keep the same
`infer(frame, request)` boundary. Return one action for chunk size `1`, or
return `list[Action]` when the source can emit a future chunk. With
`enable_rtc=True`, read `request.prev_action_chunk`,
`request.inference_delay`, and `request.execute_horizon` directly; the same
values are also available under `request.rtc_args`. `prev_action_chunk` is the
fixed-length raw chunk built from the current live buffer head and padded on
the left to the locked source chunk length, while `inference_delay` and
`execute_horizon` are both measured relative to request launch, giving an
effective RTC interval of `[inference_delay, execute_horizon)`. During cold
start, the first RTC bootstrap request still has no RTC args so inferaxis can
lock that chunk length and seed the first RTC context, and the later
warmup/profile requests already send that RTC context before the first
executable chunk is accepted. If the last RTC warmup request takes more than
`500ms`, inferaxis warns and asks whether startup should continue. A complete
RTC-aware async example lives in
`examples/06_async_inference_with_rtc.py`.

For chunked async scheduling, the runtime waits until `steps_before_request` raw steps
from the currently active chunk have been executed, then launches the next
request. `steps_before_request=0` means request immediately when a chunk is accepted.
`H_hat` still starts from the startup delay profiled over
`profile_delay_requests` requests and is then kept up to date as an EMA of
observed request latency in control steps. When handoff blending is enabled via
`ensemble_weight=...`, the weight can be one scalar or a `(low, high)` pair for
a linear earliest-to-latest ramp across the aligned handoff prefix. Built-in
gripper commands switch to the new chunk directly instead of being averaged
across the handoff boundary. `ensemble_weight` defaults to `None`. If it is
omitted, aligned handoff steps are not blended and the new chunk replaces the
old one directly.
