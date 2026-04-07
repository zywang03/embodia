# embodia examples guide

Most users do not need to read every example. The smallest useful path is:

1. [`examples/00_mixin_quickstart.py`](../examples/00_mixin_quickstart.py) to
   see the basic "edit your outer class in place, then load config" pattern.
2. [`examples/01_robot_data_collection.py`](../examples/01_robot_data_collection.py)
   if your first goal is robot-only collection with your own save logic.
3. [`examples/03_inference_runtime.py`](../examples/03_inference_runtime.py)
   if your first goal is inference-time action management such as ensemble,
   async overlap-conditioned chunk scheduling, and Hz pacing.

Several local examples intentionally share the same config file:
[`examples/basic_runtime.yml`](../examples/basic_runtime.yml). That keeps the
Python examples focused on data flow instead of repeating the same mapping
tables. For a more commented config reference, read
[`docs/yaml_config_example.yml`](./yaml_config_example.yml).

The remaining examples each cover a distinct sub-application:

- [`examples/02_rollout_loop.py`](../examples/02_rollout_loop.py) shows
  a model rollout loop where you keep your own records.
- [`examples/04_lerobot_bridge.py`](../examples/04_lerobot_bridge.py) shows the
  same embodia step data exported into one LeRobot-style JSONL shape.
