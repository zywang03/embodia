"""Tests for the optional OpenPI compatibility helpers."""

from __future__ import annotations

import unittest

import inferaxis as em
import numpy as np
from inferaxis.contrib import openpi as em_openpi
from inferaxis.contrib import remote as em_remote

from helpers import assert_array_equal, demo_image


class OpenPITests(unittest.TestCase):
    """Coverage for OpenPI request/response adaptation."""

    def test_frame_to_openpi_obs_remaps_frame_fields(self) -> None:
        frame = em.Frame(
            images={"front_rgb": demo_image()},
            state={"arm": [1.0, 2.0, 3.0], "gripper": 0.5},
            task={"prompt": "fold the cloth"},
            meta={"episode_id": "demo-1"},
        )
        frame.timestamp_ns = 123
        frame.sequence_id = 7

        obs = em_openpi.frame_to_openpi_obs(
            frame,
            image_map={"observation/front_image": "front_rgb"},
            state_map={
                "observation/joint_position": "arm",
                "observation/gripper_position": "gripper",
            },
            task_map={"prompt": "prompt"},
            meta_map={"episode_id": "episode_id"},
            include_timestamp_ns=True,
            include_sequence_id=True,
        )

        assert_array_equal(self, obs["observation/front_image"], demo_image())
        assert_array_equal(self, obs["observation/joint_position"], [1.0, 2.0, 3.0])
        assert_array_equal(self, obs["observation/gripper_position"], [0.5])
        self.assertEqual(obs["prompt"], "fold the cloth")
        self.assertEqual(obs["episode_id"], "demo-1")
        self.assertEqual(obs["timestamp_ns"], 123)
        self.assertEqual(obs["sequence_id"], 7)

    def test_openpi_action_plan_from_response_splits_groups(self) -> None:
        plan = em_openpi.openpi_action_plan_from_response(
            {"actions": [[0.1, 0.2, 0.3, 0.9], [0.4, 0.5, 0.6, 0.8]]},
            action_groups=[
                em_openpi.OpenPIActionGroup(
                    target="arm",
                    command="joint_position",
                    selector=(0, 3),
                ),
                em_openpi.OpenPIActionGroup(
                    target="gripper",
                    command="gripper_position",
                    selector=3,
                ),
            ],
        )

        self.assertEqual(len(plan), 2)
        assert_array_equal(self, plan[0].get_command("arm").value, [0.1, 0.2, 0.3])  # type: ignore[union-attr]
        assert_array_equal(self, plan[0].get_command("gripper").value, [0.9])  # type: ignore[union-attr]
        assert_array_equal(self, plan[1].get_command("arm").value, [0.4, 0.5, 0.6])  # type: ignore[union-attr]
        assert_array_equal(self, plan[1].get_command("gripper").value, [0.8])  # type: ignore[union-attr]

    def test_remote_policy_openpi_mode_drives_run_step_and_chunk(self) -> None:
        class DemoRobot(em.RobotMixin):
            ROBOT_SPEC = {
                "name": "demo_robot",
                "image_keys": ["front_rgb"],
                "components": [
                    {
                        "name": "arm",
                        "type": "arm",
                        "dof": 3,
                        "command": ["joint_position"],
                    },
                    {
                        "name": "gripper",
                        "type": "gripper",
                        "dof": 1,
                        "command": ["gripper_position"],
                    },
                ],
            }

            def __init__(self) -> None:
                self.last_action: em.Action | None = None

            def _observe_impl(self) -> dict[str, object]:
                return {
                    "images": {"front_rgb": demo_image()},
                    "state": {
                        "arm": [0.0, 0.0, 0.0],
                        "gripper": 0.0,
                    },
                    "task": {"prompt": "stack blocks"},
                }

            def _act_impl(self, action: em.Action) -> None:
                self.last_action = action

            def _reset_impl(self) -> dict[str, object]:
                return self._observe_impl()

        class StubRunner:
            def __init__(self) -> None:
                self.last_obs: dict[str, object] | None = None

            def infer(self, obs: dict[str, object]) -> dict[str, object]:
                self.last_obs = dict(obs)
                return {
                    "actions": [
                        [0.1, 0.2, 0.3, 0.9],
                        [0.4, 0.5, 0.6, 0.8],
                    ]
                }

        runner = StubRunner()
        source = em_remote.RemotePolicy(
            runner=runner,
            openpi=True,
        )

        robot = DemoRobot()
        frame = robot.reset()
        result = em.run_step(robot, source=source, frame=frame)
        chunk = source.infer_chunk(frame, object())

        observation = runner.last_obs["observation"]  # type: ignore[index]
        assert isinstance(observation, dict)
        self.assertEqual(runner.last_obs["prompt"], "stack blocks")  # type: ignore[index]
        self.assertEqual(observation["state"], [0.0, 0.0, 0.0, 0.0])
        assert_array_equal(self, result.action.get_command("arm").value, [0.1, 0.2, 0.3])  # type: ignore[union-attr]
        assert_array_equal(self, result.action.get_command("gripper").value, [0.9])  # type: ignore[union-attr]
        self.assertEqual(len(chunk), 2)
        assert_array_equal(self, chunk[1].get_command("arm").value, [0.4, 0.5, 0.6])  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
