# embodia LeRobot bridge

This document explains embodia's position on LeRobot integration.

## Design rule

LeRobot support should be:

- useful
- optional
- non-invasive

So the bridge lives in:

- `embodia.contrib.lerobot`

and not in:

- `embodia`

## Why

embodia's core problem is runtime interface unification.

LeRobot is an ecosystem integration target, not the core abstraction that
defines embodia.

That means:

- `Frame`, `Action`, `RobotMixin`, `ModelMixin`, `check_*` stay in core
- LeRobot-specific staging/export helpers stay in `contrib`

## What the bridge does today

The current bridge provides:

- `is_lerobot_available()`
- `require_lerobot()`
- `episode_to_lerobot_records()`
- `write_lerobot_jsonl()`

These helpers are intentionally small.

They do not attempt to fully recreate `LeRobotDataset`.

Instead, they give you a clean staging step:

1. collect normalized `Episode` data with embodia
2. convert it into LeRobot-oriented records
3. hand those records to your own dataset build pipeline

## Install

You do not need LeRobot for embodia's core collection helpers.

Only install it if you want the optional bridge available in your environment:

```bash
pip install "embodia[lerobot]"
```

or:

```bash
pip install lerobot
```

## Example

```python
import embodia as em
from embodia.contrib import lerobot as em_lerobot

robot = YourRobot()
episode = em.collect_episode(robot, steps=32)

records = em_lerobot.episode_to_lerobot_records(episode)
em_lerobot.write_lerobot_jsonl(episode, "data/episode_0000.jsonl")
```

## Why the records are still conservative

Different projects want different mappings from embodia data to LeRobot data:

- some want one flat state vector
- some want multiple state fields
- some want raw images
- some want video assets written separately
- some want extra task annotations

So embodia does not force one "true" LeRobot export layout on everyone.

The bridge therefore emits a conservative staging format with fields such as:

- `episode_index`
- `frame_index`
- `timestamp`
- `next.done`
- `observation.images`
- `observation.state`
- `action`

This is enough to build project-specific conversion on top.

## Future direction

A good future extension would be:

- a stricter writer for teams that already fixed their LeRobot feature schema
- optional helpers for video asset writing
- feature-schema declarations that remain outside embodia core

But those should remain optional layers, not become part of embodia's central
interface contract.
