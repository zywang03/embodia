"""Tests for inferaxis's plain-object runtime flow."""

from __future__ import annotations

import unittest

import inferaxis as infra
import numpy as np
from inferaxis.core.errors import InterfaceValidationError
from inferaxis.core.schema import (
    Command,
    CommandKindSpec,
    PolicyOutputSpec,
    PolicySpec,
    RobotSpec,
    get_command_kind_spec,
    is_known_command_kind,
    register_command_kind,
    validate_command,
)
from inferaxis.core.transform import (
    action_to_dict,
    coerce_action,
    coerce_command,
    coerce_frame,
    frame_to_dict,
)
from inferaxis.runtime.checks import check_pair, check_policy

from helpers import (
    DummyPolicy,
    DummyRobot,
    assert_array_equal,
    demo_image,
    make_chunk_request,
)


class InterfaceTests(unittest.TestCase):
    """Coverage for the schema and function-first runtime helpers."""

    def test_root_public_api_is_minimal(self) -> None:
        expected_names = {
            "Action",
            "BuiltinCommandKind",
            "ChunkRequest",
            "Command",
            "Frame",
            "InferenceMode",
            "InferenceRuntime",
            "RealtimeController",
            "run_step",
        }

        self.assertEqual(set(infra.__all__), expected_names)
        for removed_name in (
            "InterfaceValidationError",
            "PolicySpec",
            "RobotSpec",
            "action_to_dict",
            "check_pair",
            "coerce_action",
            "coerce_frame",
            "frame_to_dict",
        ):
            self.assertFalse(hasattr(infra, removed_name), removed_name)

    def test_make_chunk_request_rejects_removed_legacy_fields(self) -> None:
        with self.assertRaises(AssertionError) as ctx:
            make_chunk_request(history_start=0)

        self.assertIn("Unexpected ChunkRequest test fields", str(ctx.exception))

    def test_action_roundtrip_uses_grouped_commands(self) -> None:
        action = coerce_action(
            {
                "arm": {
                    "command": "cartesian_pose_delta",
                    "value": [0.1, 0.2, 0.3],
                    "ref_frame": "tool",
                },
                "gripper": {
                    "command": "gripper_position",
                    "value": [0.8],
                },
            }
        )

        self.assertEqual(
            action.get_command("arm").command,  # type: ignore[union-attr]
            "cartesian_pose_delta",
        )
        assert_array_equal(self, action.get_command("gripper").value, [0.8])  # type: ignore[union-attr]
        self.assertEqual(
            action_to_dict(action),
            {
                "arm": {
                    "command": "cartesian_pose_delta",
                    "value": [0.1, 0.2, 0.3],
                    "ref_frame": "tool",
                },
                "gripper": {
                    "command": "gripper_position",
                    "value": [0.8],
                },
            },
        )

    def test_coerce_action_rejects_legacy_command_list(self) -> None:
        action = {
            "commands": [
                {
                    "target": "arm",
                    "command": "cartesian_pose_delta",
                    "value": [0.0] * 6,
                }
            ]
        }

        with self.assertRaises(InterfaceValidationError):
            coerce_action(action)

    def test_command_kind_registry_exposes_builtins(self) -> None:
        spec = get_command_kind_spec("joint_position")
        self.assertEqual(spec.name, "joint_position")
        self.assertTrue(is_known_command_kind("joint_position"))

    def test_builtin_command_kind_enum_is_string_compatible(self) -> None:
        command = Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=[0.1] * 6,
        )

        self.assertEqual(command.command, "cartesian_pose_delta")
        self.assertEqual(
            infra.BuiltinCommandKind.GRIPPER_POSITION,
            "gripper_position",
        )
        self.assertTrue(is_known_command_kind(command.command))

    def test_frame_constructor_reuses_arrays_by_default(self) -> None:
        image = demo_image()
        state = np.zeros(6, dtype=np.float64)

        frame = infra.Frame(
            images={"front_rgb": image},
            state={"arm": state},
        )

        self.assertIs(frame.images["front_rgb"], image)
        self.assertIs(frame.state["arm"], state)

    def test_frame_constructor_copies_arrays_when_requested(self) -> None:
        image = demo_image()
        state = np.zeros(6, dtype=np.float64)

        frame = infra.Frame(
            images={"front_rgb": image},
            state={"arm": state},
            copy=True,
        )

        self.assertIsNot(frame.images["front_rgb"], image)
        self.assertIsNot(frame.state["arm"], state)
        self.assertFalse(np.shares_memory(frame.images["front_rgb"], image))
        self.assertFalse(np.shares_memory(frame.state["arm"], state))

    def test_command_constructor_reuses_float64_1d_array_by_default(self) -> None:
        value = np.zeros(6, dtype=np.float64)

        command = infra.Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=value,
        )

        self.assertIs(command.value, value)

    def test_command_constructor_copies_array_when_requested(self) -> None:
        value = np.zeros(6, dtype=np.float64)

        command = infra.Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=value,
            copy=True,
        )

        self.assertIsNot(command.value, value)
        self.assertFalse(np.shares_memory(command.value, value))

    def test_command_constructor_converts_list_and_non_float64_arrays(self) -> None:
        int_value = np.arange(6, dtype=np.int32)

        from_list = infra.Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=[1, 2, 3],
        )
        from_int_array = infra.Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=int_value,
        )

        self.assertEqual(from_list.value.dtype, np.dtype(np.float64))
        assert_array_equal(self, from_list.value, [1.0, 2.0, 3.0])
        self.assertEqual(from_int_array.value.dtype, np.dtype(np.float64))
        self.assertFalse(np.shares_memory(from_int_array.value, int_value))

    def test_action_single_reuses_1d_array_and_preserves_reshape_behavior(self) -> None:
        value = np.arange(6, dtype=np.float64)
        matrix = np.arange(6, dtype=np.float64).reshape(2, 3)

        action = infra.Action.single(
            target="arm",
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=value,
        )
        flattened = infra.Action.single(
            target="arm",
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=matrix,
        )
        scalar = infra.Action.single(
            target="gripper",
            command=infra.BuiltinCommandKind.GRIPPER_POSITION,
            value=0.5,
        )

        self.assertIs(action.get_command("arm").value, value)  # type: ignore[union-attr]
        self.assertEqual(flattened.get_command("arm").value.shape, (6,))  # type: ignore[union-attr]
        assert_array_equal(self, flattened.get_command("arm").value, [0, 1, 2, 3, 4, 5])  # type: ignore[union-attr]
        assert_array_equal(self, scalar.get_command("gripper").value, [0.5])  # type: ignore[union-attr]

    def test_action_single_copies_array_when_requested(self) -> None:
        value = np.arange(6, dtype=np.float64)

        action = infra.Action.single(
            target="arm",
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=value,
            copy=True,
        )

        command = action.get_command("arm")
        assert command is not None
        self.assertIsNot(command.value, value)
        self.assertFalse(np.shares_memory(command.value, value))

    def test_action_from_commands_trusted_reuses_command_objects(self) -> None:
        value = np.zeros(6, dtype=np.float64)
        command = infra.Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=value,
        )

        commands = {"arm": command}
        action = infra.Action.from_commands(commands, trusted=True)

        self.assertIs(action.commands, commands)
        self.assertIs(action.commands["arm"], command)
        self.assertIs(action.commands["arm"].value, value)

    def test_coerce_frame_and_command_reuse_standard_objects(self) -> None:
        image = demo_image()
        state = np.zeros(6, dtype=np.float64)
        frame = infra.Frame(
            images={"front_rgb": image},
            state={"arm": state},
        )
        command_value = np.zeros(6, dtype=np.float64)
        command = infra.Command(
            command=infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
            value=command_value,
        )

        copied_frame = coerce_frame(frame)
        copied_command = coerce_command(command)

        self.assertIs(copied_frame, frame)
        self.assertIs(copied_frame.images["front_rgb"], image)
        self.assertIs(copied_frame.state["arm"], state)
        self.assertIs(copied_command, command)
        self.assertIs(copied_command.value, command_value)

    def test_coerce_frame_and_command_reuse_arrays_from_mappings(self) -> None:
        image = demo_image()
        state = np.zeros(6, dtype=np.float64)
        command_value = np.zeros(6, dtype=np.float64)

        frame = coerce_frame(
            {
                "images": {"front_rgb": image},
                "state": {"arm": state},
            }
        )
        command = coerce_command(
            {
                "command": infra.BuiltinCommandKind.CARTESIAN_POSE_DELTA,
                "value": command_value,
            }
        )

        self.assertIs(frame.images["front_rgb"], image)
        self.assertIs(frame.state["arm"], state)
        self.assertIs(command.value, command_value)

    def test_register_command_kind_rejects_duplicate_name(self) -> None:
        spec = CommandKindSpec(
            name="custom:test_duplicate_registration",
            description="test kind",
            default_dim=2,
            allowed_component_types=["custom"],
        )
        register_command_kind(spec)

        with self.assertRaises(ValueError):
            register_command_kind(
                CommandKindSpec(name="custom:test_duplicate_registration")
            )

    def test_validate_command_accepts_unregistered_custom_kind(self) -> None:
        validate_command(
            Command(
                command="custom:my_lab_synergy",
                value=[0.1, 0.2],
            )
        )

    def test_validate_command_rejects_unknown_non_custom_kind(self) -> None:
        with self.assertRaises(InterfaceValidationError):
            validate_command(
                Command(
                    command="definitely_unknown_kind",
                    value=[0.0],
                )
            )

    def test_plain_executor_and_policy_pass_checks(self) -> None:
        executor = DummyRobot()
        policy = DummyPolicy()

        sample_frame = executor.reset()
        check_policy(policy, sample_frame=sample_frame)
        check_pair(executor, policy, sample_frame=sample_frame)

    def test_check_pair_is_dry_run_and_observes_only_once(self) -> None:
        class CountingExecutor(DummyRobot):
            def __init__(self) -> None:
                super().__init__()
                self.obs_calls = 0
                self.send_calls = 0

            def get_obs(self) -> infra.Frame:
                self.obs_calls += 1
                return super().get_obs()

            def send_action(self, action: infra.Action) -> None:
                del action
                self.send_calls += 1
                raise AssertionError("check_pair() must not execute send_action().")

        executor = CountingExecutor()
        policy = DummyPolicy()

        check_pair(executor, policy)

        self.assertEqual(executor.obs_calls, 1)
        self.assertEqual(executor.send_calls, 0)

    def test_check_pair_reports_group_mismatch(self) -> None:
        executor = DummyRobot()

        class IncompatiblePolicy:
            def get_spec(self) -> PolicySpec:
                return PolicySpec(
                    name="bad_model",
                    required_image_keys=["front_rgb"],
                    required_state_keys=["arm"],
                    outputs=[
                        PolicyOutputSpec(
                            target="gripper",
                            command="gripper_position",
                            dim=1,
                        )
                    ],
                )

            def reset(self) -> None:
                return None

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> infra.Action:
                del obs, request
                return infra.Action.single(
                    target="gripper",
                    command="gripper_position",
                    value=[1.0],
                )

        with self.assertRaises(InterfaceValidationError) as ctx:
            check_pair(
                executor,
                IncompatiblePolicy(),
                sample_frame=executor.reset(),
            )

        self.assertIn("missing required component", str(ctx.exception))

    def test_run_step_accepts_action_function(self) -> None:
        executor = DummyRobot()

        def scripted(obs: infra.Frame, request: infra.ChunkRequest) -> infra.Action:
            del obs, request
            return infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[1.0] * 6,
            )

        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=scripted,
        )
        assert_array_equal(self, result.action.get_command("arm").value, [1.0] * 6)  # type: ignore[union-attr]
        assert_array_equal(
            self, executor.last_action.get_command("arm").value, [1.0] * 6
        )  # type: ignore[union-attr]

    def test_run_step_accepts_bound_policy_method(self) -> None:
        executor = DummyRobot()
        policy = DummyPolicy()

        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
        )

        self.assertEqual(
            result.action.get_command("arm").command,  # type: ignore[union-attr]
            "cartesian_pose_delta",
        )

    def test_run_step_accepts_plain_local_executor_without_spec(self) -> None:
        class PlainExecutor:
            def __init__(self) -> None:
                self.last_action: infra.Action | None = None

            def get_obs(self) -> infra.Frame:
                return infra.Frame(
                    images={"front_rgb": demo_image()},
                    state={"arm": [0.0] * 6},
                )

            def send_action(self, action: infra.Action) -> None:
                self.last_action = action

            def reset(self) -> infra.Frame:
                return self.get_obs()

        executor = PlainExecutor()
        policy = DummyPolicy()
        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
        )

        assert_array_equal(self, result.action.get_command("arm").value, [0.0] * 6)  # type: ignore[union-attr]
        assert_array_equal(
            self, executor.last_action.get_command("arm").value, [0.0] * 6
        )  # type: ignore[union-attr]

    def test_run_step_accepts_frame_without_local_execution(self) -> None:
        frame = coerce_frame(
            {
                "images": {"front_rgb": demo_image()},
                "state": {"arm": [0.0] * 6},
            }
        )

        policy = DummyPolicy()
        result = infra.run_step(
            frame=frame,
            act_src_fn=policy.infer,
            execute_action=False,
        )

        assert_array_equal(self, result.action.get_command("arm").value, [0.0] * 6)  # type: ignore[union-attr]

    def test_run_step_accepts_generic_source_object(self) -> None:
        executor = DummyRobot()

        class TeleopSource:
            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> infra.Action:
                del obs, request
                return infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[0.3] * 6,
                )

        teleop = TeleopSource()
        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=teleop.infer,
        )
        assert_array_equal(self, result.action.get_command("arm").value, [0.3] * 6)  # type: ignore[union-attr]

    def test_run_step_accepts_executor_as_its_own_source(self) -> None:
        class TeleopExecutor:
            def __init__(self) -> None:
                self.last_action: infra.Action | None = None

            def get_obs(self) -> infra.Frame:
                return infra.Frame(
                    images={"front_rgb": demo_image()},
                    state={"arm": [0.0] * 6},
                )

            def send_action(self, action: infra.Action) -> None:
                self.last_action = action

            def reset(self) -> infra.Frame:
                return self.get_obs()

            def infer(
                self,
                obs: infra.Frame,
                request: infra.ChunkRequest,
            ) -> infra.Action:
                del obs, request
                return infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[0.7] * 6,
                )

        executor = TeleopExecutor()
        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=executor.infer,
        )
        assert_array_equal(self, result.action.get_command("arm").value, [0.7] * 6)  # type: ignore[union-attr]

    def test_run_step_rejects_missing_action_callable(self) -> None:
        executor = DummyRobot()

        with self.assertRaises(InterfaceValidationError):
            infra.run_step(
                observe_fn=executor.get_obs,
                act_fn=executor.send_action,
            )

    def test_run_step_prefers_executor_returned_action(self) -> None:
        class ReturningExecutor(DummyRobot):
            def send_action(self, action: infra.Action) -> infra.Action:
                del action
                self.last_action = infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[0.25] * 6,
                )
                return self.last_action

        executor = ReturningExecutor()

        def scripted(obs: infra.Frame, request: infra.ChunkRequest) -> infra.Action:
            del obs, request
            return infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[1.0] * 6,
            )

        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=scripted,
        )
        assert_array_equal(self, result.action.get_command("arm").value, [0.25] * 6)  # type: ignore[union-attr]
        assert_array_equal(
            self, executor.last_action.get_command("arm").value, [0.25] * 6
        )  # type: ignore[union-attr]

    def test_run_step_result_can_be_exported(self) -> None:
        executor = DummyRobot()
        policy = DummyPolicy()
        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
        )

        self.assertEqual(
            frame_to_dict(result.frame)["state"]["arm"],
            [0.0] * 6,
        )
        self.assertEqual(
            action_to_dict(result.action)["arm"]["command"],
            "cartesian_pose_delta",
        )

    def test_inferaxis_auto_fills_frame_sequence_id(self) -> None:
        executor = DummyRobot()

        reset_frame = executor.reset()
        reset_frame = infra.run_step(
            act_fn=executor.send_action,
            act_src_fn=lambda _obs, _request: infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[0.0] * 6,
            ),
            execute_action=False,
            frame=reset_frame,
        ).frame
        self.assertEqual(reset_frame.sequence_id, 0)

        first = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=lambda _obs, _request: infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[0.0] * 6,
            ),
        )
        second = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=lambda _obs, _request: infra.Action.single(
                target="arm",
                command="cartesian_pose_delta",
                value=[0.0] * 6,
            ),
        )

        self.assertEqual(first.frame.sequence_id, 1)
        self.assertEqual(second.frame.sequence_id, 2)

    def test_coerce_frame_auto_fills_timestamp_ns(self) -> None:
        frame = coerce_frame(
            {
                "timestamp_ns": 1,
                "sequence_id": 99,
                "images": {"front_rgb": demo_image()},
                "state": {"arm": [0.0] * 6},
            }
        )

        self.assertIsInstance(frame.timestamp_ns, int)
        self.assertGreaterEqual(frame.timestamp_ns, 0)
        self.assertNotEqual(frame.timestamp_ns, 1)
        self.assertIsNone(frame.sequence_id)

    def test_run_step_result_keeps_public_shape_small(self) -> None:
        executor = DummyRobot()
        policy = DummyPolicy()

        result = infra.run_step(
            observe_fn=executor.get_obs,
            act_fn=executor.send_action,
            act_src_fn=policy.infer,
        )

        self.assertFalse(hasattr(result, "timing"))
        self.assertTrue(hasattr(result, "control_wait_s"))

    def test_check_policy_does_not_call_reset(self) -> None:
        class ResetTrackingPolicy:
            def __init__(self) -> None:
                self.reset_calls = 0

            def get_spec(self) -> PolicySpec:
                return PolicySpec(
                    name="reset_returning_policy",
                    required_image_keys=["front_rgb"],
                    required_state_keys=["arm"],
                    outputs=[
                        PolicyOutputSpec(
                            target="arm",
                            command="cartesian_pose_delta",
                            dim=6,
                        )
                    ],
                )

            def reset(self) -> str:
                self.reset_calls += 1
                return "ok"

            def infer(
                self, obs: infra.Frame, request: infra.ChunkRequest
            ) -> infra.Action:
                del obs, request
                return infra.Action.single(
                    target="arm",
                    command="cartesian_pose_delta",
                    value=[0.0] * 6,
                )

        policy = ResetTrackingPolicy()
        frame = DummyRobot().reset()

        check_policy(policy, sample_frame=frame)
        self.assertEqual(policy.reset_calls, 0)


if __name__ == "__main__":
    unittest.main()
