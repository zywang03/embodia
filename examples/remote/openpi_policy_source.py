"""Optional remote example 2: adapt an OpenPI-style remote policy as a source.

Requires:

    pip install ".[yaml,remote]"

Run with:

    PYTHONPATH=src python examples/remote/openpi_policy_source.py
"""

from __future__ import annotations

import embodia as em
from embodia.contrib import openpi as em_openpi


class YourRobot(em.RobotMixin):
    """Local robot wrapped by embodia while the policy stays OpenPI-shaped."""

    def __init__(self) -> None:
        self.last_native_action: object | None = None

    def capture(self) -> dict[str, object]:
        return {
            "images": {"front_rgb": None},
            "state": {
                "joint_positions": [0.1] * 6,
                "position": 0.25,
            },
            "task": {"prompt": "fold the cloth"},
        }

    def send_command(self, action: object) -> object:
        accepted = em.coerce_action(action)
        self.last_native_action = accepted
        return accepted

    def home(self) -> dict[str, object]:
        return self.capture()


class StubOpenPIClient:
    """Pretend this is the official OpenPI client exposing infer(obs)."""

    def __init__(self) -> None:
        self.last_obs: dict[str, object] | None = None

    def infer(self, obs: dict[str, object]) -> dict[str, object]:
        self.last_obs = dict(obs)
        joint_positions = obs["observation/joint_position"]
        assert isinstance(joint_positions, list)
        base = float(joint_positions[0])
        return {
            "actions": [
                [base + offset] * 6 + [0.8 - 0.1 * offset]
                for offset in (0.0, 1.0)
            ]
        }


def build_openpi_obs(frame: em.Frame) -> dict[str, object]:
    """Flatten one embodia frame into a small OpenPI-style observation dict."""

    return em_openpi.frame_to_openpi_obs(
        frame,
        image_map={"observation/front_image": "front_rgb"},
        state_map={
            "observation/joint_position": "joint_positions",
            "observation/gripper_position": "position",
        },
        task_map={"prompt": "prompt"},
    )


def main() -> None:
    if not em.is_yaml_available():
        print("yaml_config: skipped, install embodia[yaml] or PyYAML")
        return

    robot = YourRobot.from_yaml("examples/basic_runtime.yml")
    client = StubOpenPIClient()
    openpi_source = em_openpi.OpenPIPolicySource(
        runner=client,
        obs_builder=build_openpi_obs,
        action_groups=[
            em_openpi.OpenPIActionGroup(
                target="arm",
                kind="cartesian_pose_delta",
                selector=(0, 6),
            ),
            em_openpi.OpenPIActionGroup(
                target="gripper",
                kind="gripper_position",
                selector=6,
            ),
        ],
    )

    frame = robot.reset()
    result = em.run_step(robot, source=openpi_source, frame=frame)
    chunk = openpi_source.infer_chunk(frame, object())

    print("openpi_obs:", client.last_obs)
    print("first_action:", em.action_to_dict(result.action))
    print("chunk_length:", len(chunk))
    print("last_native_action:", robot.last_native_action)
    print("remote example 2 passed.")


if __name__ == "__main__":
    main()
