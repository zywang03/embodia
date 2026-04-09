"""Optional remote example 1: serve one inferaxis policy over remote IO.

Requires:

    pip install ".[yaml,remote]"

Run with:

    PYTHONPATH=src python examples/remote/serve_inferaxis_policy.py
"""

from __future__ import annotations

import inferaxis as infra
import numpy as np
from inferaxis.contrib import remote as infra_remote


HOST = "127.0.0.1"
PORT = 8000


class ServedPolicy(infra.PolicyMixin):
    """Small inferaxis policy exposed through inferaxis's remote transport."""

    METHOD_ALIASES = {
        "reset": "YOUR_OWN_clear_state",
        "infer": "YOUR_OWN_infer",
    }
    POLICY_SPEC = {
        "name": "YOUR_OWN_remote_policy",
        "required_image_keys": ["YOUR_OWN_front_rgb"],
        "required_state_keys": ["YOUR_OWN_arm"],
        "required_task_keys": [],
        "outputs": [
            {
                "target": "YOUR_OWN_arm",
                "command": "cartesian_pose_delta",
                "dim": 6,
            }
        ],
    }

    def __init__(self) -> None:
        self.step_index = 0

    def YOUR_OWN_clear_state(self) -> None:
        self.step_index = 0

    def YOUR_OWN_infer(self, frame: infra.Frame) -> dict[str, object]:
        base = float(frame.state["YOUR_OWN_arm"][0] + self.step_index)
        self.step_index += 1
        return {
            "YOUR_OWN_arm": {
                "command": "cartesian_pose_delta",
                "value": np.full(6, base, dtype=np.float64),
            }
        }


def main() -> None:
    if not infra_remote.is_remote_available():
        print("remote: skipped, install inferaxis[remote]")
        return

    policy = ServedPolicy()
    infra.check_policy(
        policy,
        sample_frame=infra.Frame(
            images={"YOUR_OWN_front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={"YOUR_OWN_arm": np.zeros(6, dtype=np.float64)},
        ),
    )

    server = infra_remote.build_remote_policy_server(
        policy,
        host=HOST,
        port=PORT,
    )
    print(f"serving policy on {HOST}:{PORT}")
    print("stop with Ctrl+C")
    server.serve_forever()


if __name__ == "__main__":
    main()
