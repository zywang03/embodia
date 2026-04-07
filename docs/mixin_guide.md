# embodia mixin guide

This guide explains the intentionally small integration surface around
`RobotMixin` and `PolicyMixin`.

For most users, the main path should stay small:

1. inherit `RobotMixin` / `PolicyMixin`
2. load a shared runtime schema with `from_yaml(...)`
3. call `run_step(...)`
4. add `InferenceRuntime(...)` only when needed

The preferred direction is now:

- YAML describes the shared embodia schema only
- your Python class keeps its own constructor and native methods
- if your native names already match the schema, you do not need any remapping
- policy inputs and outputs are inferred from that shared schema

embodia now keeps its own normalized wrappers on internal `embodia_*` methods.
That means your native methods can stay named `infer`, `capture`, `home`, and so
on, while embodia still has one collision-free internal dispatch path.
`task` is policy-side context, not robot capability, so robot specs no longer
declare task-related fields.

## Method aliases

`METHOD_ALIASES` always maps embodia method names to your existing method names:

```python
METHOD_ALIASES = {
    "observe": "capture",
    "act": "send_command",
    "reset": "home",
}
```

The same keys can also live in YAML under `robot.method_aliases` or
`policy.method_aliases`. On the policy side, `infer` is the default native method
name, so you only need `policy.method_aliases.infer` when your class uses a
different name.

## Shared schema

Robot embodiment is described through components:

```python
ROBOT_SPEC = {
    "name": "your_robot",
    "image_keys": ["front_rgb"],
    "components": [
        {
            "name": "arm",
            "kind": "arm",
            "dof": 6,
            "supported_command_kinds": ["cartesian_pose_delta"],
            "state_keys": ["joint_positions"],
        },
        {
            "name": "gripper",
            "kind": "gripper",
            "dof": 1,
            "supported_command_kinds": ["gripper_position"],
            "state_keys": ["position"],
        },
    ],
}
```

Policy outputs should align with the same shared targets and command kinds:

```python
POLICY_SPEC = {
    "name": "your_model",
    "required_image_keys": ["front_rgb"],
    "required_state_keys": ["joint_positions", "position"],
    "outputs": [
        {"target": "arm", "command_kind": "cartesian_pose_delta", "dim": 6},
        {"target": "gripper", "command_kind": "gripper_position", "dim": 1},
    ],
}
```

When you use `from_yaml(...)`, you do not repeat that information inside the
`policy:` block. embodia derives:

- `required_image_keys` from `schema.images`
- `required_state_keys` from the union of `schema.components[*].state`
- `required_task_keys` from `schema.task`
- `outputs` from `schema.components[*].command_kinds`

That YAML auto-derivation expects each component to declare exactly one
command kind. If a component supports multiple command kinds and your policy needs a
more specific spec, declare `POLICY_SPEC` in Python instead of relying on YAML
inference.

The normalized runtime action is grouped by control target:

```python
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

## Optional Python-side remapping

YAML no longer carries remapping blocks. If your native class already uses
schema-standard names, that is the simplest setup.

If your native names differ, keep remapping in Python code with `MODALITY_MAPS`:

```python
MODALITY_MAPS = {
    em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
    em.STATE_KEYS: {"qpos": "joint_positions"},
    em.CONTROL_TARGETS: {"vendor_arm": "arm"},
    em.COMMAND_KINDS: {"cartesian_delta": "cartesian_pose_delta"},
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
    - front_rgb
  components:
    arm:
      kind: arm
      dof: 6
      state:
        - joint_positions
      command_kinds:
        - cartesian_pose_delta
    gripper:
      kind: gripper
      dof: 1
      state:
        - position
      command_kinds:
        - gripper_position

robot:
  name: your_robot
  method_aliases:
    observe: capture
    act: send_command
    reset: home

policy:
  name: your_model
  method_aliases:
    reset: clear_state
    infer: predict_action
```

The full example lives in [`docs/yaml_config_example.yml`](./yaml_config_example.yml).
