[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# embodia

`embodia` is a small Python library for unified runtime interfaces between
robots and models. It focuses on one thing: making different robot classes and
policy classes speak the same runtime data flow. It is not a training framework,
not a server stack, not ROS, and not a plugin system.

For most users, the public path should stay small:

1. inherit `RobotMixin` / `PolicyMixin`
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


class YourPolicy(em.PolicyMixin):
    def clear_state(self): ...
    def infer(self, frame): ...
```

Then load the runtime alignment from YAML:

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
policy = YourPolicy.from_yaml("docs/yaml_config_example.yml")
```

That YAML only describes the shared schema plus method aliases. Constructor
arguments stay in Python code. On the policy side, embodia derives required
inputs and output targets directly from the shared `schema:` block. If a policy
needs extra conditioning such as a prompt, put it in `Frame.task`. Robot specs
do not declare task-related capabilities.

### How YAML and your methods relate

`schema:` defines the canonical runtime field names. `method_aliases:` only
tells embodia which of your existing methods should be used for `observe`,
`act`, `reset`, and `infer`. It does not generate implementations and it does
not change your constructor.

If you do not declare Python-side `MODALITY_MAPS`, your native methods should
directly use the names written in YAML. In other words, YAML is the contract.
embodia validates your runtime inputs and outputs against that contract.

The mapping is:

- `schema.images` -> keys that should appear in `frame.images`
- `schema.components.<name>.state` -> keys that should appear in `frame.state`
- `schema.components.<name>` -> `Command.target`
- `schema.components.<name>.command_kinds` -> `Command.kind`
- `schema.task` -> optional policy-side keys inside `frame.task`

That means the common method contracts are:

- `capture()` / `observe()` returns `em.Frame` or a frame-like `dict` with
  `timestamp_ns`, `images`, and `state`. `images` must use the keys from
  `schema.images`. `state` must use the union of all
  `schema.components[*].state` keys.
- `home()` / `reset()` on the robot has the same return contract as
  `capture()`.
- `send_command(action)` / `act(action)` receives an `em.Action`. Its
  `action.commands[*].target` values are the component names from YAML, and
  `action.commands[*].kind` values are the command kinds declared for those
  components.
- `clear_state()` / `reset()` on the policy returns `None`.
- `infer(frame)` receives an `em.Frame`. `frame.images` and `frame.state`
  already follow the YAML schema. If `schema.task` was declared, policy-side
  context is available in `frame.task`. `infer(frame)` should return an
  `em.Action` or an action-like `dict` whose commands target the YAML
  components.

For the example YAML in this repo, the expected runtime shapes are:

```python
# robot.capture() / robot.home() return this shape
{
    "timestamp_ns": 1710000000000000000,
    "images": {
        "front_rgb": ...,
    },
    "state": {
        "joint_positions": [...],
        "position": 0.5,
    },
}

# policy.infer(frame) returns this shape
{
    "commands": [
        {
            "target": "arm",
            "kind": "cartesian_pose_delta",
            "value": [...],
        },
        {
            "target": "gripper",
            "kind": "gripper_position",
            "value": [0.5],
        },
    ],
    "dt": 0.1,
}
```

If your existing project uses different native names, keep that remapping in
Python with `MODALITY_MAPS`. embodia will translate at the boundary:

- robot `observe/reset`: native frame -> embodia `Frame`
- policy `infer`: embodia `Frame` -> native frame
- policy output: native action -> embodia `Action`
- robot `act`: embodia `Action` -> native action

So the recommended rule is simple: YAML defines the standardized structure,
your methods either speak that structure directly, or `MODALITY_MAPS` tells
embodia how to translate.

The normalized action shape is:

```python
{
    "commands": [
        {
            "target": "arm",
            "kind": "cartesian_pose_delta",
            "value": [...],
            "ref_frame": "tool",
        },
        {
            "target": "gripper",
            "kind": "gripper_position",
            "value": [0.5],
        },
    ],
    "dt": 0.1,
}
```

`Action` is a small container of grouped commands. End-effectors such as
grippers, hands, suction tools, or custom actuators are first-class robot
components, not ad-hoc extra channels.

The smallest local inference path is:

```python
result = em.run_step(robot, policy)
```

If you want runtime-side features such as async scheduling or pacing, keep the
same entrypoint and add a runtime object:

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.ASYNC,
    overlap_ratio=0.2,
)

result = em.run_step(robot, policy, runtime=runtime)
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
That shared config defines two components, `arm` and `gripper`, and the
Python examples emit one grouped `Action.commands` payload per control step.

## Design

embodia is centered on unified runtime data flow. The core pieces are
`Frame`, `Action`, `RobotProtocol`, `PolicyProtocol`, `RobotMixin`,
`PolicyMixin`, `run_step()`, and `InferenceRuntime`. The preferred split is that
your robot and policy keep doing their native work, while embodia handles
alignment, remapping, validation, and runtime flow around them.

If you need more detail, the extra docs are [`docs/mixin_guide.md`](./docs/mixin_guide.md),
[`docs/yaml_config_example.yml`](./docs/yaml_config_example.yml), and
[`docs/examples_guide.md`](./docs/examples_guide.md).
