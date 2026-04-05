"""Basic tests for the embodia package."""

from __future__ import annotations

import unittest

from embodia import (
    Action,
    Frame,
    InterfaceValidationError,
    check_model,
    check_pair,
    check_robot,
    validate_action,
    validate_frame,
)
from embodia.tests.helpers import DummyModel, DummyRobot


class InterfaceTests(unittest.TestCase):
    """Smoke tests for the minimal runtime interface package."""

    def test_dummy_components_pass_checks(self) -> None:
        robot = DummyRobot()
        model = DummyModel()
        frame = robot.observe()

        check_robot(robot)
        check_model(model, sample_frame=frame)
        check_pair(robot, model)

    def test_validate_frame_rejects_negative_timestamp(self) -> None:
        frame = Frame(timestamp_ns=-1, images={}, state={})

        with self.assertRaises(InterfaceValidationError):
            validate_frame(frame)

    def test_validate_action_rejects_bad_mode(self) -> None:
        action = Action(mode="ee_delta", value=[0.0])
        action.mode = "not_real_mode"  # type: ignore[assignment]

        with self.assertRaises(InterfaceValidationError):
            validate_action(action)

    def test_check_pair_reports_missing_keys(self) -> None:
        class IncompatibleModel(DummyModel):
            def get_spec(self):  # type: ignore[override]
                spec = super().get_spec()
                spec.required_state_keys.append("ee_pose")
                return spec

        robot = DummyRobot()
        model = IncompatibleModel()

        with self.assertRaises(InterfaceValidationError) as ctx:
            check_pair(robot, model)

        self.assertIn("missing state keys", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
