[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# embodia

`embodia` is a small Python library for unified runtime interfaces between
robots and models. It focuses on one thing: making different robot classes and
model classes speak the same runtime data flow. It is not a training framework,
not a server stack, not ROS, and not a plugin system.

For most users, the public path should stay small:

1. inherit `RobotMixin` / `ModelMixin`
2. load interface alignment with `from_yaml(...)`
3. call `run_step(...)`
4. add `InferenceRuntime(...)` only when you need inference-time features

Everything else exists to support that path, not to replace it.

## Install

```bash
git clone https://github.com/zywang03/embodia.git
cd embodia
pip install .
```

If you want YAML-based config loading with `from_yaml(...)`:

```bash
pip install ".[yaml]"
```

If you want the optional OpenPI remote-policy helpers:

```bash
pip install ".[openpi-remote]"
```

embodia does not require `numpy` or `torch`. If a user project already has
them installed, embodia can accept their arrays at the runtime boundary and
normalize them into embodia's core data structures.

## Quickstart

Keep embodia on the outermost layer of your existing classes and keep your
native methods as they are:

```python
import embodia as em


class YourRobot(em.RobotMixin):
    def capture(self): ...
    def send_command(self, action): ...
    def home(self): ...


class YourModel(em.ModelMixin):
    def clear_state(self): ...
    def infer(self, frame): ...
```

Then load the runtime alignment from YAML:

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
model = YourModel.from_yaml("docs/yaml_config_example.yml")
```

That YAML only describes interface alignment. Constructor arguments stay in
Python code. If a model needs extra conditioning such as a prompt, put it in
`Frame.task`.

The normalized action shape is:

```python
{
    "mode": "ee_delta",
    "value": [...],              # primary action vector
    "channels": {"gripper": 0.5},  # optional named extra actuators
    "ref_frame": "tool",
    "dt": 0.1,
}
```

`channels` is intentionally generic. `gripper` is just one common key, not a
special built-in field.

The smallest local inference path is:

```python
result = em.run_step(robot, model)
```

If you want runtime-side features such as async scheduling or pacing, keep the
same entrypoint and add a runtime object:

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.ASYNC,
    overlap_ratio=0.2,
)

result = em.run_step(robot, model, runtime=runtime)
```

For normal usage, that is the whole story. `check_*`, `from_config(...)`, and
other lower-level helpers are still available, but they are optional integration
tools rather than the main user path.

## Examples

`examples/` is intentionally fixed to four core scripts:

1. [`examples/01_sync_inference.py`](./examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](./examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](./examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](./examples/04_replay_collected_data.py)

They all share [`examples/basic_runtime.yml`](./examples/basic_runtime.yml).
That shared config already includes both `joint_positions` and
`gripper_position` on the state side, while the Python examples emit extra
end-effector channels through `Action.channels`.

## Design

embodia is centered on unified runtime data flow. The core pieces are
`Frame`, `Action`, `RobotProtocol`, `ModelProtocol`, `RobotMixin`,
`ModelMixin`, `run_step()`, and `InferenceRuntime`. The preferred split is that
your robot and model keep doing their native work, while embodia handles
alignment, remapping, validation, and runtime flow around them.

If you need more detail, the extra docs are:

- [`docs/mixin_guide.md`](./docs/mixin_guide.md) for config styles and mapping rules
- [`docs/yaml_config_example.yml`](./docs/yaml_config_example.yml) for a commented YAML example
- [`docs/examples_guide.md`](./docs/examples_guide.md) for how the example scripts are organized
