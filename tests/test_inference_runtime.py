"""Tests for embodia's inference runtime utilities."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import embodia as em


def arm_value(action: em.Action) -> float:
    """Return the first arm dimension from one action."""

    command = action.get_command("arm")
    assert command is not None
    return float(command.value[0])


class RuntimeRobot(em.RobotMixin):
    """Tiny robot used by inference-runtime tests."""

    def __init__(self) -> None:
        self.last_action: em.Action | None = None

    def _get_spec_impl(self) -> dict[str, object]:
        return {
            "name": "runtime_robot",
            "image_keys": ["front_rgb"],
            "components": [
                {
                    "name": "arm",
                    "kind": "arm",
                    "dof": 6,
                    "supported_command_kinds": ["cartesian_pose_delta"],
                    "state_keys": ["joint_positions"],
                }
            ],
        }

    def _observe_impl(self) -> dict[str, object]:
        return {
            "timestamp_ns": time.time_ns(),
            "images": {"front_rgb": None},
            "state": {"joint_positions": [0.0] * 6},
        }

    def _act_impl(self, action: em.Action) -> None:
        self.last_action = action

    def _reset_impl(self) -> dict[str, object]:
        return self._observe_impl()


class RuntimeModel(em.ModelMixin):
    """Single-step model plus optional chunk hooks for runtime tests."""

    def __init__(self) -> None:
        self.step_index = 0
        self.chunk_seed = 1.0

    def _get_spec_impl(self) -> dict[str, object]:
        return {
            "name": "runtime_model",
            "required_image_keys": ["front_rgb"],
            "required_state_keys": ["joint_positions"],
            "required_task_keys": [],
            "outputs": [
                {
                    "target": "arm",
                    "command_kind": "cartesian_pose_delta",
                    "dim": 6,
                }
            ],
        }

    def _reset_impl(self) -> None:
        self.step_index = 0
        self.chunk_seed = 1.0

    def _step_impl(self, frame: em.Frame) -> em.Action:
        del frame
        value = float(1 + self.step_index * 2)
        self.step_index += 1
        return em.Action.single(
            target="arm",
            kind="cartesian_pose_delta",
            value=[value] * 6,
        )

    def _step_chunk_impl(
        self,
        frame: em.Frame,
        request: em.ChunkRequest,
    ) -> list[em.Action]:
        del frame
        if not request.history_actions:
            start = self.chunk_seed
            self.chunk_seed += 2.0
            return [
                em.Action.single(
                    target="arm",
                    kind="cartesian_pose_delta",
                    value=[start] * 6,
                ),
                em.Action.single(
                    target="arm",
                    kind="cartesian_pose_delta",
                    value=[start + 1.0] * 6,
                ),
            ]

        last = arm_value(request.history_actions[-1])
        return [
            *request.history_actions,
            em.Action.single(
                target="arm",
                kind="cartesian_pose_delta",
                value=[last + 1.0] * 6,
            ),
        ]


class InferenceRuntimeTests(unittest.TestCase):
    """Coverage for action optimizers and sync/async runtime flow."""

    def test_action_ensembler_blends_with_previous_output(self) -> None:
        frame = em.Frame(timestamp_ns=1, images={}, state={})
        ensembler = em.ActionEnsembler(current_weight=0.5)

        first = ensembler(
            em.Action.single(
                target="arm",
                kind="cartesian_pose_delta",
                value=[0.0, 2.0],
            ),
            frame,
        )
        second = ensembler(
            em.Action.single(
                target="arm",
                kind="cartesian_pose_delta",
                value=[2.0, 4.0],
            ),
            frame,
        )

        self.assertEqual(first.get_command("arm").value, [0.0, 2.0])  # type: ignore[union-attr]
        self.assertEqual(second.get_command("arm").value, [1.0, 3.0])  # type: ignore[union-attr]

    def test_action_interpolator_blends_over_one_extra_step(self) -> None:
        frame = em.Frame(timestamp_ns=1, images={}, state={})
        interpolator = em.ActionInterpolator(steps=1)

        first = interpolator(
            em.Action.single(
                target="arm",
                kind="cartesian_pose_delta",
                value=[0.0, 0.0],
            ),
            frame,
        )
        second = interpolator(
            em.Action.single(
                target="arm",
                kind="cartesian_pose_delta",
                value=[2.0, 4.0],
            ),
            frame,
        )
        third = interpolator(
            em.Action.single(
                target="arm",
                kind="cartesian_pose_delta",
                value=[2.0, 4.0],
            ),
            frame,
        )

        self.assertEqual(first.get_command("arm").value, [0.0, 0.0])  # type: ignore[union-attr]
        self.assertEqual(second.get_command("arm").value, [1.0, 2.0])  # type: ignore[union-attr]
        self.assertEqual(third.get_command("arm").value, [2.0, 4.0])  # type: ignore[union-attr]

    def test_sync_runtime_applies_action_optimizers_before_execution(self) -> None:
        robot = RuntimeRobot()
        model = RuntimeModel()
        runtime = em.InferenceRuntime(
            mode=em.InferenceMode.SYNC,
            action_optimizers=[em.ActionEnsembler(current_weight=0.5)],
        )

        first = em.run_step(robot, model, runtime=runtime)
        second = em.run_step(robot, model, runtime=runtime)

        self.assertEqual(arm_value(first.raw_action), 1.0)
        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.raw_action), 3.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(robot.last_action), 2.0)  # type: ignore[arg-type]

    def test_sync_runtime_can_use_overlap_with_plan_provider(self) -> None:
        robot = RuntimeRobot()
        model = RuntimeModel()
        runtime = em.InferenceRuntime(
            mode=em.InferenceMode.SYNC,
            overlap_ratio=0.5,
        )

        first = em.run_step(robot, model, runtime=runtime)
        second = em.run_step(robot, model, runtime=runtime)
        third = em.run_step(robot, model, runtime=runtime)

        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(third.action), 3.0)

    def test_async_runtime_can_use_internal_scheduler(self) -> None:
        robot = RuntimeRobot()
        model = RuntimeModel()
        runtime = em.InferenceRuntime(
            mode=em.InferenceMode.ASYNC,
            overlap_ratio=0.5,
        )

        first = runtime.step(robot, model)
        second = runtime.step(robot, model)
        third = runtime.step(robot, model)

        self.assertTrue(first.plan_refreshed)
        self.assertEqual(arm_value(first.action), 1.0)
        self.assertEqual(arm_value(second.action), 2.0)
        self.assertEqual(arm_value(third.action), 3.0)

    def test_profile_sync_inference_writes_json_report(self) -> None:
        robot = RuntimeRobot()
        model = RuntimeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profile.json"
            profile = em.profile_sync_inference(
                robot,
                model,
                steps=4,
                execute_action=False,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["steps"], 4)
            self.assertEqual(profile.to_dict()["steps"], 4)


if __name__ == "__main__":
    unittest.main()
