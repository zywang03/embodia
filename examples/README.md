# embodia examples

These examples are ordered from the smallest-intrusion path to the most
complete data-flow demo.

## 0. Recommended quickstart: mixins

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

## 1. Fresh implementation with mixins

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

## 2. Wrap existing vendor classes

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

## 3. Multi-step rollout

```bash
PYTHONPATH=src python examples/03_rollout_loop.py
```

Use this if you want to understand the main runtime data-flow value of embodia:
observe, normalize, infer, normalize, act, and log everything in one uniform
shape.

What it shows:

- repeated `run_step()` calls
- collecting a standardized trajectory
- using `frame_to_dict()` and `action_to_dict()` for logging/export

## 4. Real pi06star pi05 policy integration

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
