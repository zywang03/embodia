# embodia collection guide

This guide is about the collection side of embodia.

The main idea stays the same:

- embodia's core job is to unify runtime interfaces
- collection is one sub-application built on top of that core
- inference is another sub-application built on top of that core

embodia is not trying to become:

- a teleoperation framework
- a dataset platform
- a training system
- a robot lab orchestration stack

## The collection stack

At the collection layer, embodia gives you three small building blocks:

1. `record_step()`
2. `collect_episode()`
3. `save_episode_h5()`

That is intentionally small.

## The main flows

### 1. Robot-only passive collection

Use this when you want to standardize observations without sending actions.

```python
import embodia as em

robot = YourRobot()
em.check_robot(robot, call_observe=False)

step = em.record_step(robot)
episode = em.collect_episode(robot, steps=100)
em.save_episode_h5(episode, "data/episode_0000.h5")
```

### 2. Robot + external action source

Use this when actions come from teleop, a joystick loop, a scripted policy, or
some external process.

```python
def scripted_action(frame: em.Frame) -> dict[str, object]:
    return {
        "mode": "ee_delta",
        "value": [0.0] * 6,
        "dt": 0.1,
    }


episode = em.collect_episode(
    robot,
    steps=100,
    action_fn=scripted_action,
    execute_actions=True,
)
```

### 3. Robot + model rollout collection

Use this when you already have a model and want the same collection shape.

```python
episode = em.collect_episode(
    robot,
    steps=100,
    model=model,
    execute_actions=True,
)
```

This is the key point:

- robot-only collection
- teleop/scripted collection
- model rollout collection

can all produce the same `Episode` structure.

## What embodia validates for you

During collection, embodia can standardize and validate:

- robot spec shape
- model spec shape
- frame structure
- action structure
- robot/model compatibility
- whether collected frames satisfy the declared robot spec
- whether actions match the declared robot/model action modes

This lets user code stay thin.

## What user code should still do

Your own robot/model code should mainly do two things:

1. expose native methods
2. declare how native names map into embodia-standard names

Try not to push heavy preprocessing into your class methods just to satisfy
embodia. Let embodia handle the structural alignment.

## What belongs in embodia core

Good fit:

- normalized runtime dataclasses
- validation helpers
- remapping helpers
- small rollout/collection helpers
- plain Python export helpers

Not a good fit:

- multi-process recording orchestration
- custom video encoders
- device drivers
- dataset hosting logic
- training configuration systems

## File export

The default collection file export is:

- `save_episode_h5()`

This keeps collection output small and portable, without pulling LeRobot into
embodia core.

The plain-Python export helpers are still available:

- `episode_to_dict()`
- `episode_step_to_dict()`

Those are still useful for:

- JSON serialization
- local logging
- unit tests
- project-specific dataset writers

Optional ecosystem bridges should live outside the main package surface.

Important:

- H5 is the default collection file format
- H5 export uses the optional `h5py` package
- basic embodia collection does not require LeRobot
- `record_step()` and `collect_episode()` stay in the core package
- `save_episode_h5()` is the default persistence helper
- only the optional LeRobot bridge lives in `embodia.contrib.lerobot`

## LeRobot bridge

LeRobot-related helpers live in `embodia.contrib.lerobot`, not in the main
package exports.

That keeps the main package small while still giving users a path to bridge
into LeRobot-oriented data pipelines.

See:

- [lerobot_bridge.md](/data/embodia/docs/lerobot_bridge.md)
