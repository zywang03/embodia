"""Example 4: wrap a real pi06star pi05 policy with embodia.

By default this example only runs embodia-side compatibility checks, so it stays
lightweight and does not load the actual openpi/pi06star policy.

Run the lightweight checks:

    PYTHONPATH=src python examples/04_pi06star_pi05_policy.py

Run one real pi05 inference step:

    PI06STAR_ROOT=/path/to/pi06star \
    EMBODIA_RUN_PI05_INFER=1 \
    PYTHONPATH=src /path/to/pi06star/.venv/bin/python \
    examples/04_pi06star_pi05_policy.py

Notes:

- This uses a local pi06star checkpoint by default if the repo lives next to
  embodia in the same parent directory.
- The wrapped pi05 policy produces an action chunk. embodia's minimal Action
  schema is single-step, so this adapter returns the first action in the chunk
  and keeps the full chunk on ``last_action_chunk`` for advanced use.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import sys
import time
from typing import Any

import embodia as em


def _default_pi06star_root() -> Path:
    """Guess the sibling pi06star repository location."""

    return Path(__file__).resolve().parents[2] / "pi06star"


def _install_pi06star_import_paths(pi06star_root: Path) -> None:
    """Make local pi06star/openpi packages importable."""

    candidates = (
        pi06star_root / "src",
        pi06star_root / "packages" / "openpi-client" / "src",
        pi06star_root / "packages" / "openpi-client",
    )
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


def _numpy() -> Any:
    """Import numpy only when the real policy path needs it."""

    import numpy as np

    return np


def _coerce_chw_uint8(image: Any) -> Any:
    """Convert a frame image into the layout openpi Aloha inference expects."""

    np = _numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(
            f"expected image with 3 dimensions, got shape {array.shape!r}."
        )

    if array.shape[0] == 3:
        chw = array
    elif array.shape[-1] == 3:
        chw = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(
            "expected image shape [3, H, W] or [H, W, 3], "
            f"got {array.shape!r}."
        )

    if np.issubdtype(chw.dtype, np.floating):
        scale = 255.0 if float(np.max(chw, initial=0.0)) <= 1.0 else 1.0
        chw = np.clip(chw * scale, 0.0, 255.0).astype(np.uint8)
    elif chw.dtype != np.uint8:
        chw = np.clip(chw, 0, 255).astype(np.uint8)

    return chw


class DemoAlohaRobot(em.RobotMixin):
    """A minimal embodia robot that exposes Aloha-like observations."""

    ROBOT_SPEC = {
        "name": "demo_aloha_robot",
        "action_modes": ["joint_position"],
        "image_keys": ["front_rgb", "left_wrist_rgb", "right_wrist_rgb"],
        "state_keys": ["joint_positions"],
    }

    def __init__(self, *, prompt: str | None = None) -> None:
        self.prompt = prompt or "take the cloth from the basket and fold the cloth."
        self.last_action: em.Action | None = None

    def _blank_image(self) -> Any:
        if os.environ.get("EMBODIA_RUN_PI05_INFER") == "1":
            return _numpy().zeros((3, 224, 224), dtype=_numpy().uint8)
        return None

    def _get_spec_impl(self) -> dict[str, object]:
        return dict(self.ROBOT_SPEC)

    def _observe_impl(self) -> dict[str, object]:
        return {
            "timestamp_ns": time.time_ns(),
            "images": {
                "front_rgb": self._blank_image(),
                "left_wrist_rgb": self._blank_image(),
                "right_wrist_rgb": self._blank_image(),
            },
            "state": {
                "joint_positions": (
                    [0.0] * 14
                    if os.environ.get("EMBODIA_RUN_PI05_INFER") != "1"
                    else _numpy().zeros((14,), dtype=_numpy().float32)
                ),
            },
            "task": {
                "prompt": self.prompt,
            },
            "meta": {
                "source": "demo_aloha_robot",
            },
        }

    def _act_impl(self, action: em.Action) -> None:
        self.last_action = action
        print(f"[DemoAlohaRobot] execute {action.mode} -> {action.value[:4]} ...")

    def _reset_impl(self) -> dict[str, object]:
        return self._observe_impl()


class Pi06starPi05Model(em.ModelMixin):
    """embodia wrapper around a local pi06star pi05 policy checkpoint.

    This wrapper is intentionally lazy: the real policy is loaded only when
    ``step()`` is called for the first time.
    """

    MODEL_SPEC = {
        "name": "pi06star_pi05_policy",
        "required_image_keys": ["front_rgb", "left_wrist_rgb", "right_wrist_rgb"],
        "required_state_keys": ["joint_positions"],
        "output_action_mode": "joint_position",
    }
    MODALITY_MAPS = {
        em.IMAGE_KEYS: {
            "cam_high": "front_rgb",
            "cam_left_wrist": "left_wrist_rgb",
            "cam_right_wrist": "right_wrist_rgb",
        },
        em.STATE_KEYS: {
            "state": "joint_positions",
        },
    }

    DEFAULT_CONFIG_NAME = "policy_pi05_fold_awbc_iter1"
    DEFAULT_CHECKPOINT_RELATIVE = Path(
        "checkpoints/policy_pi05_fold_awbc_iter1/vlm_fft_20000/9999"
    )

    def __init__(
        self,
        *,
        pi06star_root: Path | None = None,
        config_name: str = DEFAULT_CONFIG_NAME,
        checkpoint_dir: Path | None = None,
        default_prompt: str | None = None,
        dt: float = 0.1,
    ) -> None:
        self.pi06star_root = (pi06star_root or _default_pi06star_root()).resolve()
        self.config_name = config_name
        self.checkpoint_dir = (
            checkpoint_dir
            if checkpoint_dir is not None
            else self.pi06star_root / self.DEFAULT_CHECKPOINT_RELATIVE
        ).resolve()
        self.default_prompt = default_prompt
        self.dt = dt

        self._policy: Any | None = None
        self.last_policy_output: dict[str, Any] | None = None
        self.last_action_chunk: Any | None = None

    def _get_spec_impl(self) -> dict[str, object]:
        return dict(self.MODEL_SPEC)

    def _reset_impl(self) -> None:
        self.last_policy_output = None
        self.last_action_chunk = None

    def _ensure_policy_loaded(self) -> Any:
        """Load the local openpi policy on first use."""

        if self._policy is not None:
            return self._policy

        if not self.pi06star_root.exists():
            raise FileNotFoundError(
                f"pi06star root not found at {self.pi06star_root}. "
                "Set PI06STAR_ROOT to your local pi06star checkout."
            )
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"pi06star checkpoint not found at {self.checkpoint_dir}. "
                "Pass checkpoint_dir=... or set PI06STAR_ROOT to a repo that "
                "contains the expected checkpoint."
            )

        _install_pi06star_import_paths(self.pi06star_root)

        try:
            import jax.numpy as jnp

            from openpi.models import model as openpi_model
            from openpi.policies.policy import Policy
            from openpi.training import config as openpi_config
            import openpi.transforms as openpi_transforms
        except Exception as exc:  # pragma: no cover - optional integration path
            raise RuntimeError(
                "Failed to import pi06star/openpi dependencies. "
                "Run this example inside pi06star's Python environment, for "
                "example with /path/to/pi06star/.venv/bin/python."
            ) from exc

        train_config = openpi_config.get_config(self.config_name)

        # Force norm-stats loading from the local checkpoint instead of the base
        # assets path embedded in the original training config.
        data_factory = dataclasses.replace(
            train_config.data,
            assets=dataclasses.replace(
                train_config.data.assets,
                assets_dir=str(self.checkpoint_dir / "assets"),
            ),
        )
        data_config = data_factory.create(train_config.assets_dirs, train_config.model)

        model = train_config.model.load(
            openpi_model.restore_params(
                self.checkpoint_dir / "params",
                dtype=jnp.bfloat16,
            )
        )

        self._policy = Policy(
            model,
            transforms=[
                openpi_transforms.InjectDefaultPrompt(self.default_prompt),
                *data_config.data_transforms.inputs,
                openpi_transforms.Normalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                openpi_transforms.Unnormalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.data_transforms.outputs,
            ],
            metadata=train_config.policy_metadata,
        )
        return self._policy

    def _frame_to_openpi_obs(self, frame: em.Frame) -> dict[str, Any]:
        """Convert a native-key frame into the dict openpi Policy.infer expects."""

        if "state" not in frame.state:
            raise em.InterfaceValidationError(
                "pi06star pi05 model expects frame.state['state'] after native "
                "remapping, but it was missing."
            )

        obs: dict[str, Any] = {
            "images": {
                name: _coerce_chw_uint8(image)
                for name, image in frame.images.items()
            },
            "state": _numpy().asarray(
                frame.state["state"],
                dtype=_numpy().float32,
            ),
        }

        prompt = None
        if frame.task is not None:
            raw_prompt = frame.task.get("prompt")
            if raw_prompt is not None:
                prompt = str(raw_prompt)
        if prompt is not None:
            obs["prompt"] = prompt

        return obs

    def _step_impl(self, frame: em.Frame) -> dict[str, object]:
        """Run one pi05 policy step and adapt the action chunk to embodia."""

        policy = self._ensure_policy_loaded()
        policy_output = policy.infer(self._frame_to_openpi_obs(frame))
        np = _numpy()
        actions = np.asarray(policy_output["actions"], dtype=np.float32)

        if actions.ndim == 1:
            first_action = actions
        elif actions.ndim == 2 and actions.shape[0] > 0:
            first_action = actions[0]
        else:
            raise em.InterfaceValidationError(
                "pi06star policy returned actions with unsupported shape "
                f"{actions.shape!r}; expected [action_dim] or "
                "[action_horizon, action_dim]."
            )

        self.last_policy_output = policy_output
        self.last_action_chunk = actions
        return {
            "mode": "joint_position",
            "value": first_action.astype(float).tolist(),
            "dt": self.dt,
        }


def main() -> None:
    robot = DemoAlohaRobot()
    model = Pi06starPi05Model(
        pi06star_root=Path(
            os.environ.get("PI06STAR_ROOT", str(_default_pi06star_root()))
        ),
        default_prompt="take the cloth from the basket and fold the cloth.",
    )

    em.check_robot(robot)
    em.check_model(model)
    em.check_pair(robot, model)
    print("embodia checks passed.")

    if os.environ.get("EMBODIA_RUN_PI05_INFER") != "1":
        print(
            "skipped real pi05 inference. "
            "Set EMBODIA_RUN_PI05_INFER=1 to load pi06star and run one step."
        )
        return

    frame = robot.reset()
    em.check_model(model, sample_frame=frame)
    result = em.run_step(robot, model, frame=frame)

    print("standardized_frame:", em.frame_to_dict(result.frame))
    print("standardized_action:", em.action_to_dict(result.action))
    if model.last_action_chunk is not None:
        print("pi05_action_chunk_shape:", tuple(model.last_action_chunk.shape))
    print("example 4 passed.")


if __name__ == "__main__":
    main()
