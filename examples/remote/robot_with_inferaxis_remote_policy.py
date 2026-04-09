"""Optional remote example 2: local robot consuming a remote policy source.

For inferaxis remote mode, start the server first:

    PYTHONPATH=src python examples/remote/serve_inferaxis_policy.py

Then run this client:

    PYTHONPATH=src python examples/remote/robot_with_inferaxis_remote_policy.py
"""

from __future__ import annotations

import inferaxis as infra
import numpy as np
from inferaxis.contrib import remote as infra_remote


class YourRobot(infra.RobotMixin):
    """Small local robot that consumes one remote action source."""

    def __init__(self) -> None:
        self.last_native_action: object | None = None

    def YOUR_OWN_get_obs(self) -> dict[str, object]:
        return {
            "images": {"YOUR_OWN_front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            "state": {
                "YOUR_OWN_arm": np.full(6, 0.25, dtype=np.float64),
                "YOUR_OWN_gripper": np.array([0.5], dtype=np.float64),
            },
        }

    def YOUR_OWN_send_action(self, action: object) -> object:
        accepted = infra.coerce_action(action)
        self.last_native_action = accepted
        return accepted

    def YOUR_OWN_reset(self) -> dict[str, object]:
        return self.YOUR_OWN_get_obs()


def main() -> None:
    if not infra_remote.is_remote_available():
        print("remote: skipped, install inferaxis[remote]")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    remote_policy = infra_remote.RemotePolicy(
        host="127.0.0.1",
        port=8000,
        # openpi=True, # whether or not use a openpi policy
        retry_interval_s=0.05,
        connect_timeout_s=2.0,
        connect_immediately=False,
        wait_for_server=True,
    )

    reset_frame = robot.reset()
    infra.check_pair(robot, remote_policy, sample_frame=reset_frame)

    get_server_metadata = getattr(remote_policy, "get_server_metadata", None)
    if callable(get_server_metadata):
        print("server_metadata:", get_server_metadata())
    for step_index in range(3):
        result = infra.run_step(robot, source=remote_policy)
        arm = result.action.get_command("YOUR_OWN_arm")
        assert arm is not None
        print(f"step={step_index} action0={arm.value[0]:.2f}")


if __name__ == "__main__":
    main()
    # main(use_openpi=True)
