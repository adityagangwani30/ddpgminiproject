"""Live inference demo for DDPG adaptive power allocation.

Purpose:
- Load a trained policy checkpoint.
- Run deterministic inference in the wireless environment.
- Visualize, in real time, how power allocation responds to channel changes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG

import config as project_config
from environment import UplinkPowerControlEnv

LOGGER = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure project logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_model_path(model_arg: str | None) -> Path:
    """Resolve a model path from CLI input or default names."""
    candidates: list[Path] = []
    if model_arg:
        user_path = Path(model_arg)
        candidates.append(user_path)
        if user_path.suffix != ".zip":
            candidates.append(Path(f"{model_arg}.zip"))
    else:
        candidates.append(Path(f"{project_config.DEPLOY_MODEL_NAME}.zip"))
        candidates.append(Path(f"{project_config.MODEL_NAME}.zip"))

    for path in candidates:
        if path.exists():
            return path

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No model file found. Tried: {tried}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for interactive demo."""
    parser = argparse.ArgumentParser(description="Live DDPG power-allocation demo.")
    parser.add_argument("--model", type=str, default=None, help="Model path or base name.")
    parser.add_argument("--users", type=int, default=project_config.N_USERS, help="Number of users.")
    parser.add_argument("--p-max", type=float, default=project_config.P_MAX, help="Maximum power.")
    parser.add_argument(
        "--noise-power", type=float, default=project_config.NOISE_POWER, help="Noise power."
    )
    parser.add_argument(
        "--lambda-power",
        type=float,
        default=project_config.LAMBDA_POWER,
        help="Power penalty coefficient.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=project_config.EPISODE_LENGTH,
        help="Episode horizon.",
    )
    parser.add_argument("--seed", type=int, default=project_config.RANDOM_SEED, help="Random seed.")
    parser.add_argument("--steps", type=int, default=200, help="Demo timesteps to run.")
    parser.add_argument("--pause", type=float, default=0.1, help="Pause duration per frame.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def run_live_demo(model: DDPG, env: UplinkPowerControlEnv, steps: int, pause: float) -> None:
    """Run live visualization of channel gains and DDPG power allocations."""
    obs, _ = env.reset()
    user_ids = np.arange(1, env.n_users + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.ion()
    fig, (ax_gain, ax_power) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Initial deterministic action seeds the first frame before loop updates.
    initial_action, _ = model.predict(obs, deterministic=True)
    initial_power = np.clip(np.asarray(initial_action, dtype=np.float32), 0.0, env.p_max)

    gain_bars = ax_gain.bar(user_ids, obs, color="#4C72B0", edgecolor="black")
    power_bars = ax_power.bar(user_ids, initial_power, color="#55A868", edgecolor="black")

    ax_gain.set_ylabel("Channel gain |h|^2")
    ax_power.set_ylabel("Allocated power")
    ax_power.set_xlabel("User index")
    ax_power.set_ylim(0.0, max(env.p_max * 1.1, 1e-3))

    # Frame-by-frame inference loop for live adaptation visualization.
    for step_idx in range(1, steps + 1):
        # Deterministic inference shows the learned policy behavior directly.
        action, _ = model.predict(obs, deterministic=True)
        obs_next, reward, terminated, truncated, info = env.step(action)

        # Use detailed info from env when available; fallback values keep demo robust.
        gains = np.asarray(info.get("gains", obs), dtype=np.float32)
        powers = np.asarray(
            info.get("powers", np.clip(action, 0.0, env.p_max)),
            dtype=np.float32,
        )

        for bar, value in zip(gain_bars, gains):
            bar.set_height(float(value))
        for bar, value in zip(power_bars, powers):
            bar.set_height(float(value))

        ax_gain.set_ylim(0.0, max(float(np.max(gains)) * 1.2, 1e-3))
        ax_gain.set_title(
            f"Channel Gains (Step {step_idx}) | Sum Rate: {info['sum_rate']:.3f}"
        )
        # Reward and fairness are displayed to connect actions with performance.
        ax_power.set_title(
            f"DDPG Power Allocation | Reward: {reward:.3f} | Fairness: {info['fairness']:.3f}"
        )

        fig.canvas.draw_idle()
        # Small pause yields real-time animation effect.
        plt.pause(pause)

        obs = obs_next
        if terminated or truncated:
            # Restart episode to keep demo running for requested number of steps.
            obs, _ = env.reset()

    plt.ioff()
    plt.show()


def main() -> None:
    """Entry point for the live demo script."""
    args = parse_args()
    setup_logging(args.log_level)

    model_path = resolve_model_path(args.model)
    LOGGER.info("Loading model from: %s", model_path.resolve())

    env = UplinkPowerControlEnv(
        n_users=args.users,
        p_max=args.p_max,
        noise_power=args.noise_power,
        lambda_power=args.lambda_power,
        episode_length=args.episode_length,
        detailed_info=True,
        seed=args.seed,
    )
    model = DDPG.load(str(model_path), env=env)

    run_live_demo(model=model, env=env, steps=args.steps, pause=args.pause)


if __name__ == "__main__":
    main()
