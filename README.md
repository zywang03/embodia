[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# inferaxis

`inferaxis` is a small Python library for unified runtime interfaces between
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
git clone https://github.com/zywang03/inferaxis.git
cd inferaxis
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

inferaxis depends on `numpy` and keeps image/state/action tensors as
`numpy.ndarray` inside the core runtime. If a user project already uses
`torch`, inferaxis can still accept tensors at the runtime boundary and convert
them into numpy-backed core objects.

## Quickstart

Keep inferaxis on the outermost layer of your existing classes and keep your
native methods as they are. Any example name prefixed with `YOUR_OWN_` below is
just a placeholder you replace with your real method or field name:

```python
import inferaxis as infra


class YourRobot(infra.RobotMixin):
    def YOUR_OWN_get_obs(self): ...
    def YOUR_OWN_send_action(self, action): ...
    def YOUR_OWN_reset(self): ...


class YourPolicy(infra.PolicyMixin):
    def YOUR_OWN_clear_state(self): ...
    def YOUR_OWN_infer(self, frame): ...
```

Then load the runtime alignment from YAML:

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
policy = YourPolicy.from_yaml("docs/yaml_config_example.yml")
```

That YAML only describes the shared schema plus method aliases. Constructor
arguments stay in Python code. On the policy side, inferaxis derives required
inputs and output targets directly from the shared `schema:` block. If a policy
needs extra conditioning such as `YOUR_OWN_prompt`, put it in `Frame.task`.
Robot specs do not declare task-related capabilities.

### How YAML and your methods relate

`schema:` defines the canonical runtime field names. `method_aliases:` only
tells inferaxis which of your existing methods should be used for `observe`,
`act`, `reset`, and `infer`. It does not generate implementations and it does
not change your constructor.

If you do not declare Python-side `MODALITY_MAPS`, your native methods should
directly use the names written in YAML. In other words, YAML is the contract.
inferaxis validates your runtime inputs and outputs against that contract.

The mapping is:

- `schema.images` -> keys that should appear in `frame.images`
- `schema.components.<name>` -> one shared component key that should appear in
  both `frame.state[<name>]` and `action.commands[<name>]`
- `schema.components.<name>.command` -> allowed values of `Command.command` for
  that component
- `schema.task` -> optional policy-side keys inside `frame.task`

Any schema key that you are expected to rename in your own project should be
written as `YOUR_OWN_*` in inferaxis's examples and docs.
There is no separate `Command.target` field in the current schema. The target
is the dictionary key on `Action.commands`.

### Method I/O

Only these method boundaries need to align with the YAML schema:

- `YOUR_OWN_get_obs()` / `observe()` -> output must align to the shared frame structure
- `YOUR_OWN_reset()` / `reset()` -> output must align to the same frame structure as `observe()`
- `YOUR_OWN_infer(frame)` / `infer(frame)` -> input `frame` is already YAML-aligned, and output must align to the shared action structure
- `YOUR_OWN_send_action(action)` / `act(action)` -> input `action` is already YAML-aligned

`YOUR_OWN_clear_state()` / `reset()` is optional policy state cleanup. Its
return value is ignored. Numeric payloads are expected to be numpy-backed.
inferaxis manages frame timestamps and step ids internally.

`command` is not a free-form string. For each component, it must match one of
the entries declared in `schema.components.<name>.command`. inferaxis ships these
built-in command kinds:
`joint_position`, `joint_position_delta`, `joint_velocity`,
`cartesian_pose`, `cartesian_pose_delta`, `cartesian_twist`,
`gripper_position`, `gripper_position_delta`, `gripper_velocity`,
`gripper_open_close`, `hand_joint_position`,
`hand_joint_position_delta`, and `eef_activation`.
If you need a project-specific command, register a `custom:...` kind with
`register_command_kind(...)`.

If your existing project uses different native names, keep that remapping in
Python with `MODALITY_MAPS`. inferaxis will translate at the boundary:

- robot `observe/reset`: native frame -> inferaxis `Frame`
- policy `infer`: inferaxis `Frame` -> native frame
- policy output: native action -> inferaxis `Action`
- robot `act`: inferaxis `Action` -> native action

So the recommended rule is simple: YAML defines the standardized structure,
your methods either speak that structure directly, or `MODALITY_MAPS` tells
inferaxis how to translate.

`Action` is a small container of grouped commands. End-effectors such as
grippers, hands, suction tools, or custom actuators are first-class robot
components, not ad-hoc extra channels. Runtime pacing belongs to
`InferenceRuntime` / `RealtimeController`, so `Action` itself does not carry a
separate `dt` field. If action-level metadata is present, inferaxis
automatically switches to the wrapped form `{"commands": ..., "meta": ...}`.

The smallest local inference path is:

```python
result = infra.run_step(robot, source=policy)
```

If you want runtime-side features such as async scheduling or pacing, keep the
same entrypoint and add a runtime object:

```python
runtime = infra.InferenceRuntime(
    mode=infra.InferenceMode.ASYNC,
    overlap_ratio=0.2,
)

result = infra.run_step(robot, source=policy, runtime=runtime)
```

`source=` is the preferred name because the second side may be a local policy,
a remote policy client, a teleop object exposing `next_action(frame)`, or a
plain callable. `policy=` still works as a compatibility alias. `robot` stays
local-only in inferaxis's design; remote deployment belongs on the source/policy
side. If one robot class also produces teleop actions itself, it can play both
roles with `run_step(robot, source=robot)`.
For inferaxis's own remote transport, `RemotePolicy(...)` only needs connection
parameters. It infers action decoding from the remote response or server
metadata instead of asking you to repeat schema fields locally.

Per-step timing is now internal runtime bookkeeping rather than a public
`run_step(...)` output. If you explicitly want timing analysis to choose async
settings, use `profile_sync_inference(...)` instead.

If the remote side is an OpenPI policy server, keep the same outer flow and
adapt only the wire payload at the source boundary:

```python
from inferaxis.contrib import remote as infra_remote

source = infra_remote.RemotePolicy(
    host="127.0.0.1",
    port=8000,
    openpi=True,
)

result = infra.run_step(robot, source=source)
```

For the default OpenPI path, inferaxis infers request/response adaptation from
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

5. [`examples/remote/serve_inferaxis_policy.py`](./examples/remote/serve_inferaxis_policy.py)
6. [`examples/remote/robot_with_inferaxis_remote_policy.py`](./examples/remote/robot_with_inferaxis_remote_policy.py)

They all share [`examples/basic_runtime.yml`](./examples/basic_runtime.yml).
That shared config defines two placeholder components,
`YOUR_OWN_arm` and `YOUR_OWN_gripper`, and the Python examples emit one grouped
`Action.commands` mapping per control step.

## Design

inferaxis is centered on unified runtime data flow. The core pieces are
`Frame`, `Action`, `RobotProtocol`, `PolicyProtocol`, `RobotMixin`,
`PolicyMixin`, `run_step()`, and `InferenceRuntime`. The preferred split is that
your robot and policy keep doing their native work, while inferaxis handles
alignment, remapping, validation, and runtime flow around them.

If you need more detail, the extra docs are [`docs/mixin_guide.md`](./docs/mixin_guide.md),
[`docs/yaml_config_example.yml`](./docs/yaml_config_example.yml), and
[`docs/examples_guide.md`](./docs/examples_guide.md).
