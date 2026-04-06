# embodia examples

These examples are ordered from the smallest-intrusion path to increasingly
complete sub-applications built on the same interface layer.

## 0. Robot-only data collection

```bash
PYTHONPATH=src python examples/05_robot_data_collection.py
```

Start here if your first goal is standardized robot data collection.

What it shows:

- robot-only collection as a sub-application, not a second framework
- `record_step()` for one normalized sample
- `collect_episode()` for a small standardized episode
- `save_episode_h5()` as the default persistence path
- optional externally supplied actions without changing the robot API

## 1. Recommended quickstart: mixins

```bash
PYTHONPATH=src python examples/00_mixin_quickstart.py
```

Start here if your project already has a robot class and a model class.

What it shows:

- `import embodia as em`
- `RobotMixin` and `ModelMixin` as the primary integration path
- embodia mixin must stay on the far left of the base list
- a short "original structure" sketch before the final code
- declarative `ROBOT_SPEC` and `MODEL_SPEC`
- one `METHOD_ALIASES` table instead of several small wrapper methods
- minimal intrusion into an existing codebase
- one normalized `run_step()` data-flow cycle

## 2. Fresh implementation with mixins

```bash
PYTHONPATH=src python examples/01_mixin_from_scratch.py
```

Use this if you are implementing a fresh robot/model pair and still want to use
the same mixin-based integration style as the rest of the project.

What it shows:

- `RobotMixin` and `ModelMixin` without any vendor base class
- `check_robot`, `check_model`, `check_pair`
- `run_step`
- normalized `Frame` and `Action`

## 3. Wrap existing vendor classes

```bash
PYTHONPATH=src python examples/02_wrap_existing_classes.py
```

Use this if you already have classes from a robot SDK or an existing model
library and want to add embodia without rewriting them.

What it shows:

- `RobotMixin` and `ModelMixin`
- the original class shape in the top docstring, but only final code below
- one alias table, for example `capture -> observe`
- key remapping, for example `rgb_front -> front_rgb`
- mode remapping, for example `cartesian_delta -> ee_delta`
- native vendor data flow on one side and standardized embodia data on the other

## 4. Multi-step rollout

```bash
PYTHONPATH=src python examples/03_rollout_loop.py
```

Use this if you want to understand the main runtime data-flow value of embodia:
observe, normalize, infer, normalize, act, and log everything in one uniform
shape.

What it shows:

- `collect_episode(..., model=...)` for standardized rollout logging
- collecting a standardized episode
- using `episode_to_dict()` for logging/export

## 5. Real pi06star pi05 policy integration

```bash
PYTHONPATH=src python examples/04_pi06star_pi05_policy.py
```

Use this if you want to see how embodia wraps a real third-party policy stack
instead of a toy `YourModel`.

What it shows:

- a lazy `ModelMixin` wrapper around a real `pi06star` pi05 policy
- keeping heavy dependencies optional for the main embodia package
- adapting a chunked policy output to embodia's minimal single-step `Action`
- passing through `check_pair(...)` before attempting real inference

## 6. Optional LeRobot bridge

```bash
PYTHONPATH=src python examples/06_lerobot_bridge.py
```

Use this if you want to hand embodia-collected episodes into a LeRobot-oriented
pipeline without making LeRobot part of embodia core.

What it shows:

- `embodia.contrib.lerobot`
- optional dependency detection
- conservative LeRobot-oriented record export
- JSONL staging output for downstream dataset tooling
