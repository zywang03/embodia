# embodia

`embodia` is a small Python library for unified runtime interfaces between
robots and models.

Its core job is interface unification.

It is intentionally narrow:

- no network service
- no server process
- no ROS dependency
- no training framework
- no plugin system

The design is simple:

- `Protocol` is the compatibility standard
- `Mixin` is the recommended integration layer
- `check_*` is the runtime acceptance gate
- `collect_episode()` is one data-collection helper built on top
- `run_step()` is one inference helper built on top

## Install

Core install:

```bash
pip install .
```

If you only want embodia's core interface layer, checks, inference helpers, or
collection in memory, that is enough.

If you want the default H5 collection export:

```bash
pip install ".[h5]"
```

Only install the LeRobot extra when you want the optional bridge in
`embodia.contrib.lerobot`:

```bash
pip install ".[lerobot]"
```

## 30-Second Collection Example

```python
import embodia as em


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
    MODALITY_MAPS = {
        em.IMAGE_KEYS: {"rgb_front": "front_rgb"},
        em.STATE_KEYS: {"qpos": "joint_positions"},
        em.ACTION_MODES: {"cartesian_delta": "ee_delta"},
    }

    def capture(self): ...
    def send_command(self, action): ...
    def home(self): ...
```

Then:

```python
robot = YourRobot()

em.check_robot(robot, call_observe=False)

step = em.record_step(robot)
episode = em.collect_episode(robot, steps=128)
em.save_episode_h5(episode, "data/episode_0000.h5")
```

If you also have a model:

```python
model = YourModel()

em.check_pair(robot, model, sample_frame=robot.reset())
result = em.run_step(robot, model)
rollout = em.collect_episode(robot, steps=128, model=model)
```

## Core and Sub-Applications

embodia should stay small and centered on one pain point: make different robot
and model implementations speak one runtime shape.

The layers are:

- Core interface layer:
  `Frame`, `Action`, `RobotProtocol`, `ModelProtocol`, `RobotMixin`,
  `ModelMixin`, transform helpers, `check_*`
- Collection sub-application:
  `record_step()`, `collect_episode()`, `save_episode_h5()`
- Inference sub-application:
  `run_step()`

## Integration Rules

- Edit your existing outer class directly. Do not add an extra wrapper class unless you really need one.
- Put `RobotMixin` or `ModelMixin` on the far left so it is the outermost runtime layer.
- Keep your own methods thin. Let embodia do remapping, validation, and normalization.
- Prefer declarative class attributes over extra adapter methods.

Correct:

```python
class YourRobot(em.RobotMixin):
    ...
```

Wrong:

```python
class YourRobot(SomeOtherBase, em.RobotMixin):
    ...
```

## What The Class Attributes Mean

- `ROBOT_SPEC` / `MODEL_SPEC`: describe your current native interface
- `METHOD_ALIASES`: embodia method name -> your existing method name
- `MODALITY_MAPS[em.IMAGE_KEYS]`: native image key -> embodia standard key
- `MODALITY_MAPS[em.STATE_KEYS]`: native state key -> embodia standard key
- `MODALITY_MAPS[em.ACTION_MODES]`: native action mode -> embodia standard mode

If your class already uses embodia-standard names, you can leave the maps empty.
The preferred shape is to key `MODALITY_MAPS` with embodia modality tokens like
`em.IMAGE_KEYS`, not bare string names.

Detailed guide:

- [collection_guide.md](/data/embodia/docs/collection_guide.md)
- [mixin_guide.md](/data/embodia/docs/mixin_guide.md)
- [lerobot_bridge.md](/data/embodia/docs/lerobot_bridge.md)

## Examples

```bash
pip install -e .
python examples/05_robot_data_collection.py
python examples/00_mixin_quickstart.py
python examples/02_wrap_existing_classes.py
python examples/03_rollout_loop.py
python examples/01_mixin_from_scratch.py
python examples/04_pi06star_pi05_policy.py
python examples/06_lerobot_bridge.py
```

Suggested reading order:

1. `examples/05_robot_data_collection.py`
2. `examples/00_mixin_quickstart.py`
3. `examples/02_wrap_existing_classes.py`
4. `examples/03_rollout_loop.py`
5. `examples/01_mixin_from_scratch.py`
6. `examples/04_pi06star_pi05_policy.py`
7. `examples/06_lerobot_bridge.py`

## Philosophy

- Focus on unified runtime data flow, not training infrastructure.
- Make third-party code compatible without forcing a framework base class.
- Keep user-side intrusion small and predictable.
- Treat collection and inference as sub-applications of the same unified
  interface layer.
- Default collection file output is H5.
- Keep ecosystem bridges like LeRobot optional and outside the main package
  surface.
