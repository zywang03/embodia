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
optimizers, async scheduling, overlap management, and loop pacing out of your
policy implementation:

```python
runtime = infra.InferenceRuntime(
    mode=infra.InferenceMode.ASYNC,
    overlap_ratio=0.5,
    action_optimizers=[
        infra.ActionEnsembler(current_weight=0.5),
        infra.ActionInterpolator(steps=1),
    ],
    realtime_controller=infra.RealtimeController(hz=50.0),
)
```

If you use `ASYNC` or `overlap_ratio`, keep the same
`infer(frame, request)` boundary. Return one action for chunk size `1`, or
return `list[Action]` when the source can emit a future chunk.

For chunked async scheduling, the runtime uses
`overlap_steps = floor(overlap_ratio * chunk_size)` and
`trigger_steps = ceil(H_hat) + overlap_steps`, where `H_hat` is an EMA of
observed request latency in control steps.
