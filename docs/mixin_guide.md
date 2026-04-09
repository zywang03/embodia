# inferaxis mixin guide

This guide explains the intentionally small integration surface around
`RobotMixin` and `PolicyMixin`.

For most users, the main path should stay small:

1. inherit `RobotMixin` / `PolicyMixin`
2. load a shared runtime schema with `from_yaml(...)`
3. call `run_step(robot, source=...)`
4. add `InferenceRuntime(...)` only when needed

The preferred direction is now:

- YAML describes the shared inferaxis schema only
- your Python class keeps its own constructor and native methods
- if your native names already match the schema, you do not need any remapping
- policy inputs and outputs are inferred from that shared schema
- numeric payloads inside `Frame` / `Action` are numpy arrays in the runtime

inferaxis now keeps its own normalized wrappers on internal `inferaxis_*` methods.
That means your native methods can stay named `YOUR_OWN_infer`,
`YOUR_OWN_get_obs`, `YOUR_OWN_reset`, and so on, while inferaxis still has one
collision-free internal dispatch path.
`task` is policy-side context, not robot capability, so robot specs no longer
declare task-related fields. `robot` stays local-only; if you need remote
deployment, put it on the policy/source side.

## Method aliases

`METHOD_ALIASES` always maps inferaxis method names to your existing method names:

```python
METHOD_ALIASES = {
    "observe": "YOUR_OWN_get_obs",
    "act": "YOUR_OWN_send_action",
    "reset": "YOUR_OWN_reset",
}
```

The same keys can also live in YAML under `robot.method_aliases` or
`policy.method_aliases`. Any value prefixed with `YOUR_OWN_` in these examples
is just a placeholder for your real method name.

## Shared schema

Robot embodiment is described through components:

```python
ROBOT_SPEC = {
    "name": "YOUR_OWN_robot",
    "image_keys": ["YOUR_OWN_front_rgb"],
    "components": [
        {
            "name": "YOUR_OWN_arm",
            "type": "arm",
            "dof": 6,
            "command": ["cartesian_pose_delta"],
        },
        {
            "name": "YOUR_OWN_gripper",
            "type": "gripper",
            "dof": 1,
            "command": ["gripper_position"],
        },
    ],
}
```

Policy outputs should align with the same shared targets and commands:

```python
POLICY_SPEC = {
    "name": "YOUR_OWN_policy",
    "required_image_keys": ["YOUR_OWN_front_rgb"],
    "required_state_keys": ["YOUR_OWN_arm", "YOUR_OWN_gripper"],
    "outputs": [
        {"target": "YOUR_OWN_arm", "command": "cartesian_pose_delta", "dim": 6},
        {
            "target": "YOUR_OWN_gripper",
            "command": "gripper_position",
            "dim": 1,
        },
    ],
}
```

When you use `from_yaml(...)`, you do not repeat that information inside the
`policy:` block. inferaxis derives:

- `required_image_keys` from `schema.images`
- `required_state_keys` from the component names in `schema.components`
- `required_task_keys` from `schema.task`
- `outputs` from `schema.components[*].command`

In the current schema, component names are reused directly across observation
and action: `frame.state[component_name]` and
`action.commands[component_name]`. There is no separate `Command.target`
field anymore.

That YAML auto-derivation expects each component to declare exactly one
command. If a component supports multiple commands and your policy needs a
more specific spec, declare `POLICY_SPEC` in Python instead of relying on YAML
inference.

The normalized runtime action is grouped by control target:

```python
import numpy as np

{
    "YOUR_OWN_arm": {
        "command": "cartesian_pose_delta",
        "value": np.zeros(6, dtype=np.float32),
    },
    "YOUR_OWN_gripper": {
        "command": "gripper_position",
        "value": np.array([0.5], dtype=np.float32),
    },
}
```

If you later add action-level metadata, inferaxis will automatically switch to
the wrapped form `{"commands": ..., "meta": ...}`.

The same numpy-first rule applies to observations: `frame.images[*]` and
`frame.state[*]` should be `numpy.ndarray` in your native methods, and inferaxis
keeps them numpy-backed through validation, runtime execution, remote
adaptation, and export.

## Optional Python-side remapping

YAML no longer carries remapping blocks. If your native class already uses
schema-standard names, that is the simplest setup.

If your native names differ, keep remapping in Python code with `MODALITY_MAPS`:

```python
MODALITY_MAPS = {
    infra.IMAGE_KEYS: {"rgb_front": "YOUR_OWN_front_rgb"},
    infra.STATE_KEYS: {"qpos": "YOUR_OWN_arm", "gripper_pos": "YOUR_OWN_gripper"},
    infra.CONTROL_TARGETS: {"vendor_arm": "YOUR_OWN_arm"},
    infra.COMMAND_KINDS: {"cartesian_delta": "cartesian_pose_delta"},
}
```

That keeps the config file simple while still letting adapters translate
existing projects into the shared schema.

## YAML shape

`from_yaml(...)` reads one top-level `schema:` block plus either `robot:` or
`policy:`:

```yaml
schema:
  images:
    - YOUR_OWN_front_rgb
  components:
    YOUR_OWN_arm:
      type: arm
      dof: 6
      command:
        - cartesian_pose_delta
    YOUR_OWN_gripper:
      type: gripper
      dof: 1
      command:
        - gripper_position

robot:
  name: YOUR_OWN_robot
  method_aliases:
    observe: YOUR_OWN_get_obs
    act: YOUR_OWN_send_action
    reset: YOUR_OWN_reset

policy:
  name: YOUR_OWN_policy
  method_aliases:
    reset: YOUR_OWN_clear_state
    infer: YOUR_OWN_infer
```

The full example lives in [`docs/yaml_config_example.yml`](./yaml_config_example.yml).
