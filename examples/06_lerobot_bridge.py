"""Example 6: optional LeRobot-oriented export without polluting embodia core.

Run with:

    PYTHONPATH=src python examples/06_lerobot_bridge.py
"""

from __future__ import annotations

from pathlib import Path

import embodia as em
from embodia.contrib import lerobot as em_lerobot

from tests.helpers import DummyRobot


def main() -> None:
    robot = DummyRobot()
    episode = em.collect_episode(robot, steps=3)

    records = em_lerobot.episode_to_lerobot_records(episode, episode_index=7)
    output_path = Path("tmp") / "episode_0007.jsonl"
    em_lerobot.write_lerobot_jsonl(episode, output_path, episode_index=7)

    print("lerobot_available:", em_lerobot.is_lerobot_available())
    print("first_record:", records[0])
    print("jsonl_path:", output_path)
    print("example 6 passed.")


if __name__ == "__main__":
    main()
