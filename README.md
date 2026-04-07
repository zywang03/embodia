[![中文](https://img.shields.io/badge/中文-简体-blue)](./README_CN.md)  
[![English](https://img.shields.io/badge/English-English-green)](./README.md)

# embodia

`embodia` is a small Python library for unified runtime interfaces between
robots and models. It solves one narrow problem: make different robot classes
and model classes speak the same runtime data flow. It does not try to become a
training framework, a network service, a server process, a ROS wrapper, or a
plugin system.

The core idea is simple. `Protocol` expresses compatibility, `Mixin` is the
recommended low-intrusion integration layer, and `check_*` is the runtime
acceptance gate. On top of that core, `run_step()` gives you the minimal
runtime step, and `InferenceRuntime` adds optional inference-system features
like `ActionEnsembler`, `AsyncInference`, and Hz pacing without changing the
main step entrypoint. embodia intentionally does not own episode boundaries or
persistence. You keep your own records and save them however your project
needs.

## Install

Clone the repository and install the core package:

```bash
git clone https://github.com/zywang03/embodia.git
cd embodia
pip install .
```

That gives you the core interface layer, transforms, checks, and inference
runtime. If you want YAML-based config files with `from_yaml(...)`, install
the optional `yaml` extra:

```bash
pip install ".[yaml]"
```

If you want the optional OpenPI remote-policy client/server helpers in
`embodia.contrib.openpi_remote`, install that extra separately as well:

```bash
pip install ".[openpi-remote]"
```

embodia does not require `numpy` or `torch`. If a user project already has
them installed, embodia can accept `ndarray` or `Tensor` values at the runtime
boundary and normalize them into its own core data structures.

## Quickstart

The recommended path is to edit your existing outer class directly, keep
`RobotMixin` or `ModelMixin` on the far left, and keep your own methods thin.
The lowest-intrusion style is usually: leave your native methods as they are,
and move embodia's interface config into `from_yaml(...)` or `from_config(...)`
instead of hard-coding big dictionaries in the class body.

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

Then describe the native interface once at construction time:

```python
robot = YourRobot.from_config(
    robot_spec={
        "name": "your_robot",
        "action_modes": ["cartesian_delta"],
        "image_keys": ["rgb_front"],
        "state_keys": ["qpos"],
    },
    method_aliases={
        "observe": "capture",
        "act": "send_command",
        "reset": "home",
    },
    modality_maps={
        em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
        em.STATE_KEYS: {"qpos": "joint_positions"},
        em.ACTION_MODES: {"cartesian_delta": "ee_delta"},
    },
)
```

If you want that config completely outside Python code, the same structure can
live in YAML through `from_yaml(...)`. A commented example lives in
[`docs/yaml_config_example.yml`](docs/yaml_config_example.yml), and the shared
runtime example config used by several example scripts lives in
[`examples/basic_runtime.yml`](examples/basic_runtime.yml).

```python
robot = YourRobot.from_yaml("docs/yaml_config_example.yml")
model = YourModel.from_yaml("docs/yaml_config_example.yml")
```

Once the class is aligned, you can use embodia as a pure runtime layer. A
robot-only collection flow can be as small as:

```python
import json

robot = YourRobot()

em.check_robot(robot, call_observe=False)

records = []
for _ in range(128):
    result = em.run_step(robot, action_fn=teleop_or_scripted_policy)
    records.append(
        {
            "frame": em.frame_to_dict(result.frame),
            "action": em.action_to_dict(result.action),
        }
    )

with open("data/episode_0000.jsonl", "w", encoding="utf-8") as handle:
    for record in records:
        handle.write(json.dumps(record))
        handle.write("\n")
```

If you also have a model, the same standardized interface can drive inference:

```python
model = YourModel()

em.check_pair(robot, model, sample_frame=robot.reset())
result = em.run_step(robot, model)
```

If your action comes from somewhere else, such as a remote policy service, the
same step entrypoint still works:

```python
robot = YourRobot.from_config(remote_policy={...})

result = em.run_step(robot)
```

embodia will use the configured robot-side remote policy automatically. If you
want to plug in some other action source, that still works, but the common
remote-policy path does not require any extra call on your side.

If you want inference-time optimizers or Hz pacing, keep using the same
`run_step()` API and pass a runtime object:

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.SYNC,
    action_optimizers=[em.ActionEnsembler(window_size=2)],
    realtime_controller=em.RealtimeController(hz=50.0),
)

result = em.run_step(robot, model, runtime=runtime)
```

If your policy natively produces chunks, `AsyncInference` can also manage
async prefetch and overlap in the background, but that is an advanced backend
path rather than the default local example path.

```python
runtime = em.InferenceRuntime(
    mode=em.InferenceMode.ASYNC,
    async_inference=em.AsyncInference(
        chunk_provider=my_chunk_provider,
        condition_steps=3,
        prefetch_steps=3,
    ),
    realtime_controller=em.RealtimeController(hz=50.0),
)
```

`mode` is explicit on purpose: sync and async are different runtime semantics,
so embodia does not infer them indirectly from whether `async_inference` was
provided.

## How embodia is organized

embodia stays centered on unified runtime data flow. The core layer is
`Frame`, `Action`, `RobotProtocol`, `ModelProtocol`, `RobotMixin`,
`ModelMixin`, the transform helpers, and `check_*`. The standardized single
step is `run_step()`. Multi-step rollout logic, episode boundaries, and file
formats stay in user code or in examples. `InferenceRuntime`,
`ActionEnsembler`, `AsyncInference`, and pacing sit on top of that same core
step without making the model wrapper heavier.

The preferred inference split is that the model stays simple and only maps one
observation to one action. If you need action buffering, reuse across control
steps, smoothing, or async overlap-conditioned chunk scheduling, that state
should live in embodia's runtime layer rather than on the model wrapper.

The same separation applies to remote deployment. Robot-side request logic
should stay on the robot side, while model-side serve logic should stay on the
model side. In other words, `RobotMixin` is the natural place to request a
remote policy, and `ModelMixin` is the natural place to expose a local model
through a remote-serving adapter.

The intended integration style is conservative. Do not add an extra wrapper
class unless you really need one. Keep `RobotMixin` or `ModelMixin` as the
outermost runtime layer, for example `class YourRobot(em.RobotMixin): ...` or
`class YourRobot(em.RobotMixin, VendorRobot): ...`. Let embodia handle
remapping, validation, and normalization, instead of pushing heavy data
preprocessing into your own methods. The public examples in this repository all
follow that same "edit the outer class in place" pattern rather than a second
from-scratch adapter style.

If you dislike bare string keys in `ROBOT_SPEC`, `MODEL_SPEC`, or
`METHOD_ALIASES`, embodia also exports `em.RobotSpecKey`,
`em.ModelSpecKey`, and `em.MethodAliasKey`. They are `StrEnum` values, so they
work as readable constants while remaining fully compatible with the existing
string-based config format.

## Where to look next

If your first goal is collection, start with
[`examples/01_robot_data_collection.py`](examples/01_robot_data_collection.py).
If your project already has a robot class and a model class, start with
[`examples/00_mixin_quickstart.py`](examples/00_mixin_quickstart.py). If you
want to understand the shared YAML-first config shape behind several local
examples, read [`examples/basic_runtime.yml`](examples/basic_runtime.yml) and
[`docs/yaml_config_example.yml`](docs/yaml_config_example.yml).
If you want to see the rollout side, read
[`examples/02_rollout_loop.py`](examples/02_rollout_loop.py). If you want the
inference-system layer with framework-side action maintenance, smoothing, and
async overlap-conditioned chunk scheduling, read
[`examples/03_inference_runtime.py`](examples/03_inference_runtime.py). If you
want one example of custom LeRobot-style export, read
[`examples/04_lerobot_bridge.py`](examples/04_lerobot_bridge.py).

The longer guides live in
[`docs/examples_guide.md`](docs/examples_guide.md),
[`docs/mixin_guide.md`](docs/mixin_guide.md). The overall philosophy is to
keep embodia small, predictable, and runtime-focused: unify the data flow,
keep user-side intrusion low, avoid baking in one storage format, and keep
ecosystem-specific bridges optional.
