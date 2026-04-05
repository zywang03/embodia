# embodia mixin guide

This guide explains how to fill:

- `ROBOT_SPEC`
- `MODEL_SPEC`
- `METHOD_ALIASES`
- `IMAGE_KEY_MAP`
- `STATE_KEY_MAP`
- `ACTION_MODE_MAP`

The main rule is:

- `*_SPEC` describes your native interface today
- `*_MAP` renames native names into embodia-standard names
- `METHOD_ALIASES` tells embodia which existing method on your class should be
  used

## Direction rules

| Field | Direction |
| --- | --- |
| `METHOD_ALIASES` | embodia name -> your existing method name |
| `IMAGE_KEY_MAP` | native key -> embodia standard key |
| `STATE_KEY_MAP` | native key -> embodia standard key |
| `ACTION_MODE_MAP` | native mode -> embodia standard mode |
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
        "name": "your_robot",
        "action_modes": ["cartesian_delta"],
        "image_keys": ["rgb_front"],
        "state_keys": ["qpos"],
    }
    METHOD_ALIASES = {
        "observe": "capture",
        "act": "send_command",
        "reset": "home",
    }
    IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
    STATE_KEY_MAP = {"qpos": "joint_positions"}
    ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}
```

Meaning:

- `ROBOT_SPEC["action_modes"]`: the native action mode names your robot accepts
- `ROBOT_SPEC["image_keys"]`: the native image keys your robot outputs
- `ROBOT_SPEC["state_keys"]`: the native state keys your robot outputs
- `METHOD_ALIASES["observe"] = "capture"`: embodia `observe()` should call your `capture()`
- `METHOD_ALIASES["act"] = "send_command"`: embodia `act()` should call your `send_command()`
- `METHOD_ALIASES["reset"] = "home"`: embodia `reset()` should call your `home()`
- `IMAGE_KEY_MAP`: rename native image keys to embodia-standard image keys
- `STATE_KEY_MAP`: rename native state keys to embodia-standard state keys
- `ACTION_MODE_MAP`: rename native action mode names to embodia-standard action mode names

Runtime flow:

```text
capture() -> native keys
-> RobotMixin observe()
-> remap native keys to standard keys
-> standardized Frame
```

and:

```text
standardized Action
-> RobotMixin act()
-> remap standard mode back to native mode
-> send_command(...)
```

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
        "name": "your_model",
        "required_image_keys": ["rgb_front"],
        "required_state_keys": ["qpos"],
        "output_action_mode": "cartesian_delta",
    }
    METHOD_ALIASES = {
        "reset": "clear_state",
        "step": "infer",
    }
    IMAGE_KEY_MAP = {"rgb_front": "front_rgb"}
    STATE_KEY_MAP = {"qpos": "joint_positions"}
    ACTION_MODE_MAP = {"cartesian_delta": "ee_delta"}
```

Meaning:

- `MODEL_SPEC["required_image_keys"]`: native image keys your model expects
- `MODEL_SPEC["required_state_keys"]`: native state keys your model expects
- `MODEL_SPEC["output_action_mode"]`: native action mode your model returns
- `METHOD_ALIASES["reset"] = "clear_state"`: embodia `reset()` should call your `clear_state()`
- `METHOD_ALIASES["step"] = "infer"`: embodia `step()` should call your `infer()`

Runtime flow:

```text
standardized Frame
-> ModelMixin step()
-> remap standard keys to native keys
-> infer(native_frame)
-> native action mode
-> remap native mode to standard mode
-> standardized Action
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
    IMAGE_KEY_MAP = {
        "your_native_image_key": "your_standard_image_key",
    }
    STATE_KEY_MAP = {
        "your_native_state_key": "your_standard_state_key",
    }
    ACTION_MODE_MAP = {
        "your_native_action_mode": "your_standard_action_mode",
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
    IMAGE_KEY_MAP = {
        "your_native_image_key": "your_standard_image_key",
    }
    STATE_KEY_MAP = {
        "your_native_state_key": "your_standard_state_key",
    }
    ACTION_MODE_MAP = {
        "your_native_action_mode": "your_standard_action_mode",
    }
```

## When you can leave things empty

- If your methods are already called `observe`, `act`, `reset`, `step`, leave
  `METHOD_ALIASES` empty.
- If your keys are already embodia-standard, leave key maps empty.
- If your action mode is already embodia-standard, leave `ACTION_MODE_MAP` empty.

## Common mistakes

- Do not reverse `METHOD_ALIASES`.
  Write `{"observe": "capture"}`, not `{"capture": "observe"}`.
- Do not reverse `IMAGE_KEY_MAP` or `STATE_KEY_MAP`.
  Write native -> standard, not standard -> native.
- Do not write standardized names into `*_SPEC` if the native interface still
  uses different names and you also provide a map.
- Do not put `RobotMixin` or `ModelMixin` anywhere except the far-left outer
  position in the base list.
- Do not push heavy preprocessing into your own methods just to satisfy embodia.
  Keep your methods thin and let embodia do remapping and validation.
