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

If you want the optional remote policy helpers:

```bash
pip install ".[remote]"
```

embodia depends on `numpy` and keeps image/state/action tensors as
`numpy.ndarray` inside the core runtime. If a user project already uses
`torch`, embodia can still accept tensors at the runtime boundary and convert
them into numpy-backed core objects.

## Quickstart

Keep embodia on the outermost layer of your existing classes and keep your
native methods as they are. Any example name prefixed with `YOUR_OWN_` below is
just a placeholder you replace with your real method or field name:

```python
import embodia as em


class YourRobot(em.RobotMixin):
    def YOUR_OWN_get_obs(self): ...
    def YOUR_OWN_send_action(self, action): ...
    def YOUR_OWN_reset(self): ...


class YourPolicy(em.PolicyMixin):
    def YOUR_OWN_clear_state(self): ...
    def YOUR_OWN_infer(self, frame): ...
```

Then load the runtime alignment from YAML:

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
policy = YourPolicy.from_yaml("docs/yaml_config_example.yml")
```

That YAML only describes the shared schema plus method aliases. Constructor
arguments stay in Python code. On the policy side, embodia derives required
inputs and output targets directly from the shared `schema:` block. If a policy
needs extra conditioning such as `YOUR_OWN_prompt`, put it in `Frame.task`.
Robot specs do not declare task-related capabilities.

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
- `schema.components.<name>` -> one shared component key that should appear in
  both `frame.state[<name>]` and `action.commands[<name>]`
- `schema.components.<name>.command` -> allowed values of `Command.command` for
  that component
- `schema.task` -> optional policy-side keys inside `frame.task`

Any schema key that you are expected to rename in your own project should be
written as `YOUR_OWN_*` in embodia's examples and docs.
There is no separate `Command.target` field in the current schema. The target
is the dictionary key on `Action.commands`.

### Method I/O

Write your native methods against one standard observation shape and one
standard action shape:

```python
import numpy as np

obs = {
    # optional, embodia fills them when omitted
    "timestamp_ns": 0,
    "sequence_id": 0,
    "images": {
        "YOUR_OWN_front_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
    },
    "state": {
        "YOUR_OWN_left_arm": np.zeros(6, dtype=np.float32),
        "YOUR_OWN_left_gripper": np.array([0.5], dtype=np.float32),
        "YOUR_OWN_right_arm": np.zeros(6, dtype=np.float32),
        "YOUR_OWN_right_gripper": np.array([0.5], dtype=np.float32),
    },
    # optional, policy-side context only
    "task": {
        "YOUR_OWN_prompt": "fold the cloth",
    },
}

action = {
    "YOUR_OWN_left_arm": {
        "command": "cartesian_pose_delta",
        "value": np.zeros(6, dtype=np.float32),
    },
    "YOUR_OWN_left_gripper": {
        "command": "gripper_position",
        "value": np.array([0.5], dtype=np.float32),
    },
    "YOUR_OWN_right_arm": {
        "command": "cartesian_pose_delta",
        "value": np.zeros(6, dtype=np.float32),
    },
    "YOUR_OWN_right_gripper": {
        "command": "gripper_position",
        "value": np.array([0.5], dtype=np.float32),
    },
}
```

The method contracts are:

- `YOUR_OWN_get_obs()` / `observe()` -> return `em.Frame` or one `obs`-like `dict`
- robot `YOUR_OWN_reset()` / `reset()` -> same return shape as `observe()`
- `YOUR_OWN_infer(frame)` / `infer(frame)` -> receive one `em.Frame`, return `em.Action` or one
  `action`-like `dict`
- `YOUR_OWN_send_action(action)` / `act(action)` -> receive one `em.Action`
- policy `YOUR_OWN_clear_state()` / `reset()` -> return value is ignored

Numeric payloads should be numpy-backed. If your robot omits `timestamp_ns` or
`sequence_id`, embodia fills them automatically.

If your existing project uses different native names, keep that remapping in
Python with `MODALITY_MAPS`. embodia will translate at the boundary:

- robot `observe/reset`: native frame -> embodia `Frame`
- policy `infer`: embodia `Frame` -> native frame
- policy output: native action -> embodia `Action`
- robot `act`: embodia `Action` -> native action

So the recommended rule is simple: YAML defines the standardized structure,
your methods either speak that structure directly, or `MODALITY_MAPS` tells
embodia how to translate.

`Action` is a small container of grouped commands. End-effectors such as
grippers, hands, suction tools, or custom actuators are first-class robot
components, not ad-hoc extra channels. Runtime pacing belongs to
`InferenceRuntime` / `RealtimeController`, so `Action` itself does not carry a
separate `dt` field. If action-level metadata is present, embodia
automatically switches to the wrapped form `{"commands": ..., "meta": ...}`.

The smallest local inference path is:

```python
result = em.run_step(robot, source=policy)
```

If you want runtime-side features such as async scheduling or pacing, keep the
same entrypoint and add a runtime object:

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.ASYNC,
    overlap_ratio=0.2,
)

result = em.run_step(robot, source=policy, runtime=runtime)
```

`source=` is the preferred name because the second side may be a local policy,
a remote policy client, a teleop object exposing `next_action(frame)`, or a
plain callable. `policy=` still works as a compatibility alias. `robot` stays
local-only in embodia's design; remote deployment belongs on the source/policy
side. If one robot class also produces teleop actions itself, it can play both
roles with `run_step(robot, source=robot)`.
For embodia's own remote transport, `RemotePolicy(...)` only needs connection
parameters. It infers action decoding from the remote response or server
metadata instead of asking you to repeat schema fields locally.

Per-step timing is now internal runtime bookkeeping rather than a public
`run_step(...)` output. If you explicitly want timing analysis to choose async
settings, use `profile_sync_inference(...)` instead.

If the remote side is an OpenPI policy server, keep the same outer flow and
adapt only the wire payload at the source boundary:

```python
from embodia.contrib import remote as em_remote

source = em_remote.RemotePolicy(
    host="127.0.0.1",
    port=8000,
    openpi=True,
)

result = em.run_step(robot, source=source)
```

For the default OpenPI path, embodia infers request/response adaptation from
the wrapped robot spec in the background. If you want to reuse the official
OpenPI client object directly, pass it through `runner=...` as long as it
exposes `infer(obs)`.

For normal usage, that is the whole story. `check_*`, `from_config(...)`, and
other lower-level helpers are still available, but they are optional integration
tools rather than the main user path.

## Examples

`examples/` is intentionally fixed to four core scripts:

1. [`examples/01_sync_inference.py`](./examples/01_sync_inference.py)
2. [`examples/02_async_inference.py`](./examples/02_async_inference.py)
3. [`examples/03_data_collection.py`](./examples/03_data_collection.py)
4. [`examples/04_replay_collected_data.py`](./examples/04_replay_collected_data.py)

There is also one optional remote folder:

5. [`examples/remote/serve_embodia_policy.py`](./examples/remote/serve_embodia_policy.py)
6. [`examples/remote/robot_with_embodia_remote_policy.py`](./examples/remote/robot_with_embodia_remote_policy.py)

They all share [`examples/basic_runtime.yml`](./examples/basic_runtime.yml).
That shared config defines two placeholder components,
`YOUR_OWN_arm` and `YOUR_OWN_gripper`, and the Python examples emit one grouped
`Action.commands` mapping per control step.

## Design

embodia is centered on unified runtime data flow. The core pieces are
`Frame`, `Action`, `RobotProtocol`, `PolicyProtocol`, `RobotMixin`,
`PolicyMixin`, `run_step()`, and `InferenceRuntime`. The preferred split is that
your robot and policy keep doing their native work, while embodia handles
alignment, remapping, validation, and runtime flow around them.

If you need more detail, the extra docs are [`docs/mixin_guide.md`](./docs/mixin_guide.md),
[`docs/yaml_config_example.yml`](./docs/yaml_config_example.yml), and
[`docs/examples_guide.md`](./docs/examples_guide.md).
