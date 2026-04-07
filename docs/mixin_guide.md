# embodia mixin guide

This guide explains the embodia mixin configuration surface.

embodia accepts two configuration styles:

- class attributes such as `ROBOT_SPEC`, `MODEL_SPEC`, `METHOD_ALIASES`, and
  `MODALITY_MAPS`
- constructor-time config through `from_config(...)`
- file-based config through `from_yaml(...)`

For low-intrusion integrations, `from_yaml(...)` or `from_config(...)` is
usually the best default, because it keeps the wrapped class body focused on
its native methods.

`from_config(...)` matches the Python-side runtime fields directly:

- `*_SPEC` describes your native interface today
- `METHOD_ALIASES` tells embodia which existing methods to call
- `MODALITY_MAPS` renames native names into embodia-standard names
- prefer embodia modality tokens like `em.IMAGE_KEYS` over bare string keys

If you want the declarative dictionaries to read more like named config fields,
you can also use `em.RobotSpecKey`, `em.ModelSpecKey`, and
`em.MethodAliasKey` instead of raw string keys.

## YAML-first shape

If you want the config outside Python code entirely, YAML uses one compact
`interface:` block instead of separate `*_SPEC` and `MODALITY_MAPS` sections:

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
model = YourModel.from_yaml("docs/yaml_config_example.yml")
```

The full commented example lives in
[`docs/yaml_config_example.yml`](./yaml_config_example.yml).

In YAML, the direction is intentionally different from `MODALITY_MAPS`:

| Field | Direction |
| --- | --- |
| `interface.images` | embodia standard key -> your native key |
| `interface.state` | embodia standard key -> your native key |
| `interface.action_modes` | embodia standard mode -> your native mode |
| `interface.output_action_mode` | embodia standard mode |

## Direction rules

| Field | Direction |
| --- | --- |
| `METHOD_ALIASES` | embodia name -> your existing method name |
| `MODALITY_MAPS[em.IMAGE_KEYS]` | native key -> embodia standard key |
| `MODALITY_MAPS[em.STATE_KEYS]` | native key -> embodia standard key |
| `MODALITY_MAPS[em.ACTION_MODES]` | native mode -> embodia standard mode |
| `ROBOT_SPEC.image_keys` | native image keys |
| `ROBOT_SPEC.state_keys` | native state keys |
| `ROBOT_SPEC.action_modes` | native action modes |
| `MODEL_SPEC.required_image_keys` | native image keys |
| `MODEL_SPEC.required_state_keys` | native state keys |
| `MODEL_SPEC.output_action_mode` | native action mode |

## Robot side

Suppose your current robot class already looks like:

```python
class YourRobot:
    def capture(self): ...
    def send_command(self, action): ...
    def home(self): ...
```

and it produces:

```python
{
    "timestamp_ns": ...,
    "images": {"rgb_front": ...},
    "state": {"qpos": ...},
}
```

and accepts an action mode called `cartesian_delta`.

Then:

```python
class YourRobot(em.RobotMixin):
    ROBOT_SPEC = {
        em.RobotSpecKey.NAME: "your_robot",
        em.RobotSpecKey.ACTION_MODES: ["cartesian_delta"],
        em.RobotSpecKey.IMAGE_KEYS: ["rgb_front"],
        em.RobotSpecKey.STATE_KEYS: ["qpos"],
    }
    METHOD_ALIASES = {
        em.MethodAliasKey.OBSERVE: "capture",
        em.MethodAliasKey.ACT: "send_command",
        em.MethodAliasKey.RESET: "home",
    }
    MODALITY_MAPS = {
        em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
        em.STATE_KEYS: {"qpos": "joint_positions"},
        em.ACTION_MODES: {"cartesian_delta": "ee_delta"},
    }
```

Meaning:

- `ROBOT_SPEC["action_modes"]`: the native action mode names your robot accepts
- `ROBOT_SPEC["image_keys"]`: the native image keys your robot outputs
- `ROBOT_SPEC["state_keys"]`: the native state keys your robot outputs
- `METHOD_ALIASES["observe"] = "capture"`: embodia `observe()` should call your `capture()`
- `METHOD_ALIASES["act"] = "send_command"`: embodia `act()` should call your `send_command()`
- `METHOD_ALIASES["reset"] = "home"`: embodia `reset()` should call your `home()`
- `MODALITY_MAPS[em.IMAGE_KEYS]`: rename native image keys to embodia-standard image keys
- `MODALITY_MAPS[em.STATE_KEYS]`: rename native state keys to embodia-standard state keys
- `MODALITY_MAPS[em.ACTION_MODES]`: rename native action mode names to embodia-standard action mode names

## Model side

Suppose your current model class already looks like:

```python
class YourModel:
    def clear_state(self): ...
    def infer(self, frame): ...
```

and it expects native keys like `rgb_front` and `qpos`, and returns an action
mode called `cartesian_delta`.

Then:

```python
class YourModel(em.ModelMixin):
    MODEL_SPEC = {
        em.ModelSpecKey.NAME: "your_model",
        em.ModelSpecKey.REQUIRED_IMAGE_KEYS: ["rgb_front"],
        em.ModelSpecKey.REQUIRED_STATE_KEYS: ["qpos"],
        em.ModelSpecKey.OUTPUT_ACTION_MODE: "cartesian_delta",
    }
    METHOD_ALIASES = {
        em.MethodAliasKey.RESET: "clear_state",
        em.MethodAliasKey.STEP: "infer",
    }
    MODALITY_MAPS = {
        em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
        em.STATE_KEYS: {"qpos": "joint_positions"},
        em.ACTION_MODES: {"cartesian_delta": "ee_delta"},
    }
```

## Minimal template

```python
import embodia as em


class YourRobot(em.RobotMixin):
    ROBOT_SPEC = {
        "name": "your_robot",
        "action_modes": ["your_native_action_mode"],
        "image_keys": ["your_native_image_key"],
        "state_keys": ["your_native_state_key"],
    }
    METHOD_ALIASES = {
        "observe": "your_native_observe_method",
        "act": "your_native_act_method",
        "reset": "your_native_reset_method",
    }
    MODALITY_MAPS = {
        em.IMAGE_KEYS: {
            "your_native_image_key": "your_standard_image_key",
        },
        em.STATE_KEYS: {
            "your_native_state_key": "your_standard_state_key",
        },
        em.ACTION_MODES: {
            "your_native_action_mode": "your_standard_action_mode",
        },
    }


class YourModel(em.ModelMixin):
    MODEL_SPEC = {
        "name": "your_model",
        "required_image_keys": ["your_native_image_key"],
        "required_state_keys": ["your_native_state_key"],
        "output_action_mode": "your_native_action_mode",
    }
    METHOD_ALIASES = {
        "reset": "your_native_reset_method",
        "step": "your_native_step_method",
    }
    MODALITY_MAPS = {
        em.IMAGE_KEYS: {
            "your_native_image_key": "your_standard_image_key",
        },
        em.STATE_KEYS: {
            "your_native_state_key": "your_standard_state_key",
        },
        em.ACTION_MODES: {
            "your_native_action_mode": "your_standard_action_mode",
        },
    }
```

## When you can leave things empty

- If your keys are already embodia-standard, leave the corresponding modality
  maps empty.
- If your action mode is already embodia-standard, leave
  `MODALITY_MAPS[em.ACTION_MODES]` empty.

For direct "edit the outer class in place" integrations, prefer keeping
`METHOD_ALIASES` explicit and pointing at your native method names such as
`capture`, `send_command`, `home`, `clear_state`, or `infer`. That keeps the
embodia wrapper layer visible and avoids mixing the wrapped methods with the
public embodia methods on the same class body.

## Common mistakes

- Do not reverse `METHOD_ALIASES`.
  Write `{"observe": "capture"}`, not `{"capture": "observe"}`.
- Do not reverse `MODALITY_MAPS[em.IMAGE_KEYS]` or `MODALITY_MAPS[em.STATE_KEYS]`.
  Write native -> standard, not standard -> native.
- Do not write standardized names into `*_SPEC` if the native interface still
  uses different names and you also provide a map.
- Do not put `RobotMixin` or `ModelMixin` anywhere except the far-left outer
  position in the base list.
- Do not push heavy preprocessing into your own methods just to satisfy embodia.
  Keep your methods thin and let embodia do remapping and validation.

## Backward compatibility

Legacy top-level attributes such as `IMAGE_KEY_MAP`, `STATE_KEY_MAP`, and
`ACTION_MODE_MAP` are still accepted for now. String keys inside
`MODALITY_MAPS` are also accepted, but embodia modality tokens are the
recommended public configuration shape going forward.
