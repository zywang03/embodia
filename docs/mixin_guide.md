# embodia mixin guide

This guide explains the intentionally small integration surface around
`RobotMixin` and `ModelMixin`.

For most users, the main path should stay small:

1. inherit `RobotMixin` / `ModelMixin`
2. load a shared runtime schema with `from_yaml(...)`
3. call `run_step(...)`
4. add `InferenceRuntime(...)` only when needed

The preferred direction is now:

- YAML describes the shared embodia schema only
- your Python class keeps its own constructor and native methods
- if your native names already match the schema, you do not need any remapping

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
`model.method_aliases`.

## Shared schema

Robot embodiment is described through control groups:

```python
ROBOT_SPEC = {
    "name": "your_robot",
    "image_keys": ["front_rgb"],
    "groups": [
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

Model outputs are declared against the same targets and command kinds:

```python
MODEL_SPEC = {
    "name": "your_model",
    "required_image_keys": ["front_rgb"],
    "required_state_keys": ["joint_positions", "position"],
    "outputs": [
        {"target": "arm", "command_kind": "cartesian_pose_delta", "dim": 6},
        {"target": "gripper", "command_kind": "gripper_position", "dim": 1},
    ],
}
```

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

YAML no longer carries `bindings` or `interface` remapping blocks. If your
native class already uses schema-standard names, that is the simplest setup.

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
`model:`:

```yaml
schema:
  images:
    - front_rgb
  groups:
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
```

The full example lives in [`docs/yaml_config_example.yml`](./yaml_config_example.yml).
