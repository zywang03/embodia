# inferaxis examples guide

The public examples are intentionally fixed to four core paths:

1. [`examples/01_sync_inference.py`](../examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](../examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](../examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](../examples/04_replay_collected_data.py)

There is also one optional deployment-oriented folder:

5. [`examples/remote/serve_inferaxis_policy.py`](../examples/remote/serve_inferaxis_policy.py)
6. [`examples/remote/robot_with_inferaxis_remote_policy.py`](../examples/remote/robot_with_inferaxis_remote_policy.py)

They all share [`examples/basic_runtime.yml`](../examples/basic_runtime.yml), so
the Python files stay focused on the main user-facing flow: mixin inheritance,
`from_yaml(...)`, `run_step(...)`, and `InferenceRuntime(...)`.
For inference examples, `run_step(robot, source=policy)` is the preferred style.
Numeric runtime payloads in those examples are numpy arrays, not Python lists.
Any example field prefixed with `YOUR_OWN_` is a placeholder you are expected to
rename inside your own project.

The examples follow one grouped-command, numpy-based schema throughout:

- robot image/state payloads use the same component keys on both observation
  and action paths, such as `YOUR_OWN_arm` and `YOUR_OWN_gripper`
- robot embodiment is declared through components such as `YOUR_OWN_arm` and
  `YOUR_OWN_gripper`
- every action step is an `Action` with one or more `commands`
- each command lives under one component key and stores `command` plus a
  numpy-backed `value`

The action source itself can vary without changing the outer loop:

- sync/async inference examples use `source=policy`
- data collection uses `source=teleop_object`
- replay keeps using a plain callable source
- remote deployment uses `source=RemotePolicy(...)`

For inferaxis's own remote example, `RemotePolicy(...)` only needs connection
settings. Command target/command are inferred from the remote response or the
server's published policy metadata. The same client example can also switch to
an OpenPI-compatible remote source with `openpi=True`.

If you need more configuration detail, read
[`docs/yaml_config_example.yml`](./yaml_config_example.yml) and
[`docs/mixin_guide.md`](./mixin_guide.md).
