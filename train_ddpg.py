"""Train a DDPG agent for adaptive uplink transmit power control.

This script orchestrates end-to-end learning:
1) Build the custom wireless environment.
2) Configure DDPG (actor, critic, replay buffer, target networks, exploration).
3) Run training and save model artifacts.
4) Produce reward diagnostics and optional quick evaluation.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

import config as project_config
from environment import UplinkPowerControlEnv

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters and artifact paths for DDPG training."""

    n_users: int = project_config.N_USERS
    p_max: float = project_config.P_MAX
    noise_power: float = project_config.NOISE_POWER
    lambda_power: float = project_config.LAMBDA_POWER
    episode_length: int = project_config.EPISODE_LENGTH
    total_timesteps: int = project_config.TRAINING_TIMESTEPS
    learning_rate: float = project_config.LEARNING_RATE
    seed: int = project_config.RANDOM_SEED

    buffer_size: int = 80_000
    learning_starts: int = 500
    batch_size: int = 128
    train_freq_steps: int = 4
    gradient_steps: int = 1

    hidden_size: int = 128
    verbose: int = 0
    device: str = "auto"
    cpu_threads: int = 1
    report_steps: int = 800

    # Primary and deployment-oriented model names.
    model_path: str = project_config.MODEL_NAME
    deploy_model_path: str = project_config.DEPLOY_MODEL_NAME

    training_curve_path: Path = field(
        default_factory=lambda: project_config.results_path(
            project_config.TRAINING_CURVE_FILENAME
        )
    )
    rewards_path: Path = field(
        default_factory=lambda: project_config.results_path(
            project_config.TRAINING_REWARDS_FILENAME
        )
    )

    legacy_training_curve_path: Path = field(
        default_factory=lambda: project_config.legacy_path(
            project_config.TRAINING_CURVE_FILENAME
        )
    )
    legacy_rewards_path: Path = field(
        default_factory=lambda: project_config.legacy_path(
            project_config.TRAINING_REWARDS_FILENAME
        )
    )


def setup_logging(log_level: str) -> None:
    """Configure project logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_random_seeds(seed: int) -> None:
    """Set NumPy and PyTorch random seeds for reproducibility."""
    # Seeds are set in one place so train/eval runs are easier to reproduce.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_if_needed(source: Path, destination: Path) -> None:
    """Copy source to destination only when paths differ."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def plot_training_rewards(episode_rewards: list[float], output_path: Path) -> None:
    """Plot episodic reward convergence with moving-average smoothing."""
    if not episode_rewards:
        LOGGER.warning("No episode rewards recorded. Skipping reward plot.")
        return

    episodes = np.arange(1, len(episode_rewards) + 1)
    window = min(20, len(episode_rewards))
    # Moving average helps visualize trend under noisy episodic returns.
    smooth = np.convolve(np.array(episode_rewards), np.ones(window) / window, mode="valid")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, episode_rewards, alpha=0.35, label="Episode reward", color="#4C72B0")
    ax.plot(
        np.arange(window, len(episode_rewards) + 1),
        smooth,
        linewidth=2.0,
        label=f"Moving average ({window})",
        color="#C44E52",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("DDPG Training Reward Convergence")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_training_artifacts(episode_rewards: list[float], config: TrainConfig) -> None:
    """Persist training artifacts under results/ and legacy root paths."""
    rewards_array = np.array(episode_rewards, dtype=np.float32)

    config.rewards_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.rewards_path, rewards_array)
    # Keep root-level copies for older workflows that expect legacy paths.
    copy_if_needed(config.rewards_path, config.legacy_rewards_path)

    plot_training_rewards(episode_rewards, config.training_curve_path)
    if config.training_curve_path.exists():
        copy_if_needed(config.training_curve_path, config.legacy_training_curve_path)
    else:
        LOGGER.warning(
            "Training curve file was not generated (likely no completed episodes): %s",
            config.training_curve_path,
        )


def train_ddpg(config: TrainConfig) -> tuple[DDPG, list[float]]:
    """Create the environment, train DDPG, and save model artifacts."""
    set_random_seeds(config.seed)

    if config.cpu_threads > 0:
        torch.set_num_threads(config.cpu_threads)

    env = UplinkPowerControlEnv(
        n_users=config.n_users,
        p_max=config.p_max,
        noise_power=config.noise_power,
        lambda_power=config.lambda_power,
        episode_length=config.episode_length,
        detailed_info=False,
        seed=config.seed,
    )
    # Monitor records episodic reward/length metadata for diagnostics.
    monitored_env = Monitor(env)

    action_dim = monitored_env.action_space.shape[0]
    # Exploration process:
    # DDPG policy is deterministic, so Gaussian action noise is injected during
    # training to explore nearby continuous actions in power space.
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim, dtype=np.float32),
        # Exploration noise scale is proportional to maximum power.
        sigma=(0.1 * np.ones(action_dim, dtype=np.float32) * config.p_max),
    )

    # DDPG internals configured here:
    # - policy="MlpPolicy": actor and critic are neural MLPs.
    # - buffer_size/learning_starts/batch_size: replay buffer behavior.
    # - tau: target-network soft update coefficient.
    # - train_freq + gradient_steps: update cadence.
    model = DDPG(
        policy="MlpPolicy",
        env=monitored_env,
        # Same hidden-layer template for actor/critic network backbones.
        policy_kwargs={"net_arch": [config.hidden_size, config.hidden_size]},
        learning_rate=config.learning_rate,
        # Replay buffer controls off-policy sample reuse.
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        # Target-network smoothing for critic/actor stability.
        tau=0.005,
        gamma=0.99,
        # How often training updates occur relative to data collection.
        train_freq=(config.train_freq_steps, "step"),
        gradient_steps=config.gradient_steps,
        action_noise=action_noise,
        verbose=config.verbose,
        seed=config.seed,
        device=config.device,
    )

    # SB3 training loop (conceptually):
    # collect transition -> store in replay buffer -> sample mini-batch ->
    # update critic (TD target) -> update actor (policy gradient via critic) ->
    # soft-update target networks.
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    # Pull episodic rewards from monitor wrapper for plotting/reporting.
    episode_rewards = [float(x) for x in monitored_env.get_episode_rewards()]

    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.deploy_model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)
    model.save(config.deploy_model_path)

    save_training_artifacts(episode_rewards, config)
    return model, episode_rewards


def quick_policy_report(model: DDPG, config: TrainConfig, n_eval_steps: int) -> None:
    """Run a fast deterministic sanity-check report after training."""
    env = UplinkPowerControlEnv(
        n_users=config.n_users,
        p_max=config.p_max,
        noise_power=config.noise_power,
        lambda_power=config.lambda_power,
        episode_length=config.episode_length,
        detailed_info=False,
        seed=config.seed + 123,
    )

    obs, _ = env.reset()
    sum_rates: list[float] = []
    total_powers: list[float] = []
    fairness_values: list[float] = []
    rewards: list[float] = []

    # Deterministic actions evaluate the learned policy without exploration noise.
    for _ in range(n_eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        sum_rates.append(float(info["sum_rate"]))
        total_powers.append(float(info["total_power"]))
        fairness_values.append(float(info["fairness"]))
        if terminated or truncated:
            obs, _ = env.reset()

    LOGGER.info("Final learned DDPG policy performance (quick report):")
    LOGGER.info("  Average reward      : %.4f", float(np.mean(rewards)))
    LOGGER.info("  Average sum rate    : %.4f", float(np.mean(sum_rates)))
    LOGGER.info("  Average total power : %.4f", float(np.mean(total_powers)))
    LOGGER.info("  Average fairness    : %.4f", float(np.mean(fairness_values)))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for configurable training."""
    parser = argparse.ArgumentParser(description="Train DDPG for uplink power control.")
    parser.add_argument(
        "--mode",
        choices=["quick", "fast", "balanced", "full"],
        default="balanced",
        help="Preset for speed/quality trade-off.",
    )
    parser.add_argument("--users", type=int, default=None, help="Number of uplink users (3-5).")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps.")
    parser.add_argument("--learning-rate", type=float, default=None, help="DDPG learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--p-max", type=float, default=None, help="Maximum power per user.")
    parser.add_argument("--noise-power", type=float, default=None, help="AWGN noise power.")
    parser.add_argument("--lambda-power", type=float, default=None, help="Power penalty coefficient.")
    parser.add_argument("--episode-length", type=int, default=None, help="Episode horizon.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Torch CPU threads (1 is often fastest for this problem size).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Override quick-report evaluation steps.",
    )
    parser.add_argument("--skip-report", action="store_true", help="Skip post-training quick report.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def apply_mode_preset(cfg: TrainConfig, mode: str) -> None:
    """Apply predefined training presets."""
    if mode in ("quick", "fast"):
        cfg.total_timesteps = 8_000
        cfg.episode_length = 300
        cfg.buffer_size = 50_000
        cfg.learning_starts = 200
        cfg.batch_size = 64
        cfg.train_freq_steps = 8
        cfg.gradient_steps = 1
        cfg.hidden_size = 64
        cfg.verbose = 0
        cfg.report_steps = 400
    elif mode == "balanced":
        cfg.total_timesteps = project_config.TRAINING_TIMESTEPS
        cfg.episode_length = project_config.EPISODE_LENGTH
        cfg.buffer_size = 80_000
        cfg.learning_starts = 500
        cfg.batch_size = 128
        cfg.train_freq_steps = 4
        cfg.gradient_steps = 1
        cfg.hidden_size = 128
        cfg.verbose = 0
        cfg.report_steps = 800
    elif mode == "full":
        cfg.total_timesteps = 50_000
        cfg.episode_length = 300
        cfg.buffer_size = 100_000
        cfg.learning_starts = 1_000
        cfg.batch_size = 128
        cfg.train_freq_steps = 4
        cfg.gradient_steps = 1
        cfg.hidden_size = 128
        cfg.verbose = 1
        cfg.report_steps = 1_000
    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_cli_overrides(cfg: TrainConfig, args: argparse.Namespace) -> None:
    """Override config defaults from command-line arguments."""
    if args.users is not None:
        cfg.n_users = int(args.users)
    if args.timesteps is not None:
        cfg.total_timesteps = int(args.timesteps)
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.p_max is not None:
        cfg.p_max = float(args.p_max)
    if args.noise_power is not None:
        cfg.noise_power = float(args.noise_power)
    if args.lambda_power is not None:
        cfg.lambda_power = float(args.lambda_power)
    if args.episode_length is not None:
        cfg.episode_length = int(args.episode_length)
    if args.eval_steps is not None:
        cfg.report_steps = int(args.eval_steps)

    cfg.device = args.device
    cfg.cpu_threads = max(1, int(args.cpu_threads))


def main() -> None:
    """Entry point for DDPG training."""
    args = parse_args()
    setup_logging(args.log_level)

    cfg = TrainConfig()
    # Preset first, then allow explicit CLI flags to override final values.
    apply_mode_preset(cfg, args.mode)
    apply_cli_overrides(cfg, args)

    LOGGER.info(
        "Training config | users=%d timesteps=%d lr=%.5f seed=%d",
        cfg.n_users,
        cfg.total_timesteps,
        cfg.learning_rate,
        cfg.seed,
    )

    start = time.perf_counter()
    trained_model, episodic_rewards = train_ddpg(cfg)
    elapsed = time.perf_counter() - start

    LOGGER.info("Training complete. Episodes recorded: %d", len(episodic_rewards))
    LOGGER.info("Training time (s): %.2f", elapsed)
    LOGGER.info("Saved model: %s", Path(f"{cfg.model_path}.zip").resolve())
    LOGGER.info("Saved deploy model: %s", Path(f"{cfg.deploy_model_path}.zip").resolve())
    LOGGER.info("Saved training curve: %s", cfg.training_curve_path.resolve())
    LOGGER.info("Saved reward history: %s", cfg.rewards_path.resolve())

    if not args.skip_report:
        quick_policy_report(trained_model, cfg, n_eval_steps=cfg.report_steps)


if __name__ == "__main__":
    main()
