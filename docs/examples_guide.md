# inferaxis examples guide

The public examples are intentionally fixed to six local paths:

1. [`examples/01_sync_inference.py`](../examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](../examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](../examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](../examples/04_replay_collected_data.py)
5. [`examples/05_profile_inference_latency.py`](../examples/05_profile_inference_latency.py)
6. [`examples/06_async_inference_with_rtc.py`](../examples/06_async_inference_with_rtc.py)

The examples now center on the same function-first path throughout: plain local
objects, explicit method references such as `observe_fn=...`, `act_fn=...`,
`act_src_fn=policy.infer`, plus `run_step(...)` and
`InferenceRuntime(...)`. Numeric payloads are numpy arrays, not Python lists,
and any name prefixed with `YOUR_OWN_` is just a placeholder to replace in your
own project.

`examples/01_sync_inference.py` shows the smallest closed-loop single-step sync
runtime with loop-rate control. `examples/02_async_inference.py` keeps
the same outer `run_step(...)` call and only changes the passed policy behavior:
return one `Action` for chunk size `1`, or return multiple future
actions from the same `act_src_fn(frame, request)` contract. `examples/03` and `examples/04`
show that data collection and replay still use the same outer loop; the only
thing that changes is where the action comes from. `examples/05` is a small
utility example for profiling sync inference latency. It can estimate
sustainable control hz directly from measured inference latency and returned
chunk length, but it always profiles against one required `target_hz` because
the control rate should already be fixed by the real system or dataset setup.
`examples/06` keeps the async runtime shape but enables top-level RTC request
fields so a policy can read the full remaining chunk, estimated delay, and
execute horizon for RTC-aware planning.
The async runtime uses `floor(overlap_ratio * chunk_size)` overlap steps and a
step-based latency EMA to decide when to request the next chunk.
For `pi06star`-style setups, a realistic starting point is `chunk_steps=50`
with `overlap_ratio=0.2`; `32` is also common for Pi0FAST, while smaller
training-oriented configs often use `16`, `15`, or `10`.

If you want the plain-object method contract, read
[`docs/plain_objects_guide.md`](./plain_objects_guide.md).
