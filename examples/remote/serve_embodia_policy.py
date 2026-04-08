"""Optional remote example 1: serve one embodia policy over remote IO.

Requires:

    pip install ".[yaml,remote]"

Run with:

    PYTHONPATH=src python examples/remote/serve_embodia_policy.py
"""

from __future__ import annotations

import embodia as em
import numpy as np
from embodia.contrib import remote as em_remote


HOST = "127.0.0.1"
PORT = 8000


class ServedPolicy(em.PolicyMixin):
    """Small embodia policy exposed through embodia's remote transport."""

    POLICY_SPEC = {
        "name": "remote_demo_policy",
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

    def clear_state(self) -> None:
        return None

    def infer(self, frame: em.Frame) -> dict[str, object]:
        base = float(frame.state["joint_positions"][0] + (frame.sequence_id or 0))
        return {
            "arm": {
                "kind": "cartesian_pose_delta",
                "value": [base] * 6,
            }
        }


def main() -> None:
    if not em_remote.is_remote_available():
        print("remote: skipped, install embodia[remote]")
        return

    policy = ServedPolicy()
    em.check_policy(
        policy,
        sample_frame=em.Frame(
            timestamp_ns=1,
            images={"front_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            state={"joint_positions": np.zeros(6, dtype=np.float64)},
        ),
    )

    server = em_remote.build_remote_policy_server(
        policy,
        host=HOST,
        port=PORT,
    )
    print(f"serving policy on {HOST}:{PORT}")
    print("stop with Ctrl+C")
    server.serve_forever()


if __name__ == "__main__":
    main()
