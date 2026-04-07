# embodia mixin guide

This guide explains the small configuration surface around `RobotMixin` and
`ModelMixin`.

For most users, the main path should stay small:

1. inherit `RobotMixin` / `ModelMixin`
2. load interface alignment with `from_yaml(...)`
3. call `run_step(...)`
4. add `InferenceRuntime(...)` only when needed

embodia accepts three equivalent configuration styles:

- class attributes such as `ROBOT_SPEC`, `MODEL_SPEC`, `METHOD_ALIASES`, and `MODALITY_MAPS`
- constructor-time config through `from_config(...)`
- file-based config through `from_yaml(...)`

## Direction rules

`METHOD_ALIASES` always points from embodia method names to your existing
method names:

```python
METHOD_ALIASES = {
    "observe": "capture",
    "act": "send_command",
    "reset": "home",
}
```

`MODALITY_MAPS` always points from native names to embodia-standard names:

```python
MODALITY_MAPS = {
    em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
    em.STATE_KEYS: {"qpos": "joint_positions"},
    em.CONTROL_TARGETS: {"vendor_arm": "arm"},
    em.ACTION_MODES: {"cartesian_delta": "ee_delta"},
}
```

## Robot side

Robot embodiment is declared with grouped control specs, not one global action
mode:

```python
ROBOT_SPEC = {
    "name": "your_robot",
    "image_keys": ["rgb_front"],
    "groups": [
        {
            "name": "arm",
            "kind": "arm",
            "dof": 6,
            "action_modes": ["cartesian_delta"],
            "state_keys": ["qpos"],
        },
        {
            "name": "gripper",
            "kind": "gripper",
            "dof": 1,
            "action_modes": ["gripper_position"],
            "state_keys": ["gripper_pos"],
        },
    ],
}
```

When embodia normalizes one robot action, it becomes:

```python
{
    "commands": [
        {"target": "arm", "mode": "ee_delta", "value": [...]},
        {"target": "gripper", "mode": "scalar_position", "value": [0.5]},
    ],
    "dt": 0.1,
}
```

## Model side

Model outputs are also explicit per control group:

```python
MODEL_SPEC = {
    "name": "your_model",
    "required_image_keys": ["rgb_front"],
    "required_state_keys": ["qpos", "gripper_pos"],
    "required_task_keys": ["instruction"],
    "outputs": [
        {"target": "arm", "mode": "cartesian_delta", "dim": 6},
        {"target": "gripper", "mode": "gripper_position", "dim": 1},
    ],
}
```

If your native model returns native names, keep the body untouched and let
`MODALITY_MAPS` translate them:

```python
MODALITY_MAPS = {
    em.STATE_KEYS: {
        "qpos": "joint_positions",
        "gripper_pos": "position",
    },
    em.CONTROL_TARGETS: {
        "vendor_arm": "arm",
        "vendor_gripper": "gripper",
    },
    em.ACTION_MODES: {
        "cartesian_delta": "ee_delta",
        "gripper_position": "scalar_position",
    },
}
```

## YAML-first shape

If you want the alignment outside Python code entirely, YAML uses a compact
`interface:` block that expands into spec plus remapping tables internally.

Robot-side YAML uses `groups`:

```yaml
robot:
  interface:
    name: your_robot
    images:
      front_rgb: rgb_front
    groups:
      arm:
        native_name: vendor_arm
        kind: arm
        dof: 6
        state:
          joint_positions: qpos
        action_modes:
          ee_delta: cartesian_delta
```

Model-side YAML uses `outputs`:

```yaml
model:
  interface:
    name: your_model
    state:
      joint_positions: qpos
    outputs:
      arm:
        native_name: vendor_arm
        mode: ee_delta
        native_mode: cartesian_delta
        dim: 6
```

The full commented example lives in
[`docs/yaml_config_example.yml`](./yaml_config_example.yml).
