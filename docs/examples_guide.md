# embodia examples guide

The public examples are intentionally fixed to four core paths:

1. [`examples/01_sync_inference.py`](../examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](../examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](../examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](../examples/04_replay_collected_data.py)

They all share [`examples/basic_runtime.yml`](../examples/basic_runtime.yml), so
the Python files stay focused on the main user-facing flow: mixin inheritance,
`from_yaml(...)`, `run_step(...)`, and `InferenceRuntime(...)`.

The examples follow one grouped-command schema throughout:

- robot state is standardized into semantic keys such as `joint_positions` and `position`
- robot embodiment is declared through control groups such as `arm` and `gripper`
- every action step is an `Action` with one or more `commands`
- each command addresses one control group through `target + mode + value`

If you need more configuration detail, read
[`docs/yaml_config_example.yml`](./yaml_config_example.yml) and
[`docs/mixin_guide.md`](./mixin_guide.md).
