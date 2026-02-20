"""
Train a DDPG agent for adaptive uplink transmit power control.

Core learning idea:
- Actor network proposes continuous power vector actions.
- Critic network evaluates action quality (Q-value).
- Replay buffer stores transitions for off-policy updates.
- Target networks stabilize learning via slow parameter tracking.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from environment import UplinkPowerControlEnv


@dataclass
class TrainConfig:
    """
    Hyperparameter container for training and reporting.

    Values are grouped to make experiment settings explicit and reproducible.
    """

    n_users: int = 4
    p_max: float = 1.0
    noise_power: float = 1e-2
    lambda_power: float = 0.1
    episode_length: int = 400
    total_timesteps: int = 20_000
    seed: int = 42

    # Replay buffer parameters:
    # - buffer_size controls memory of past transitions.
    # - learning_starts delays updates until enough experience is collected.
    # - batch_size controls sample size per gradient step.
    buffer_size: int = 80_000
    learning_starts: int = 500
    batch_size: int = 128

    # Update schedule:
    # - train_freq_steps defines how often to trigger optimization.
    # - gradient_steps defines how many updates to run when training triggers.
    train_freq_steps: int = 4
    gradient_steps: int = 1

    # Two-layer MLP width for actor and critic.
    hidden_size: int = 128
    verbose: int = 0
    device: str = "auto"
    cpu_threads: int = 1
    report_steps: int = 800
    model_path: str = "ddpg_power_control_model"
    training_curve_path: str = "training_reward_curve.png"
    rewards_path: str = "training_episode_rewards.npy"


def plot_training_rewards(episode_rewards: list[float], output_path: str) -> None:
    """
    Plot and save the convergence curve of episodic training rewards.

    The raw curve may be noisy due to fading randomness and exploration noise,
    so we additionally plot a moving-average trend.
    """
    if not episode_rewards:
        return

    episodes = np.arange(1, len(episode_rewards) + 1)
    window = min(20, len(episode_rewards))
    smooth = np.convolve(
        np.array(episode_rewards), np.ones(window) / window, mode="valid"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, episode_rewards, alpha=0.35, label="Episode reward")
    plt.plot(
        np.arange(window, len(episode_rewards) + 1),
        smooth,
        linewidth=2.0,
        label=f"Moving average ({window})",
    )
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("DDPG Training Reward Convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_ddpg(config: TrainConfig) -> tuple[DDPG, list[float]]:
    """
    Build environment and train DDPG with the configured hyperparameters.

    Notes on DDPG mechanics used by Stable-Baselines3:
    - Replay buffer:
      transitions (s, a, r, s') are stored up to buffer_size and sampled
      uniformly for off-policy learning.
    - Target networks:
      SB3 maintains target actor/critic and performs Polyak averaging with
      coefficient tau (set below) to stabilize bootstrapped Q-updates.
    - Exploration:
      Gaussian action noise is added to actor outputs during training.
    """
    # For this compact network/problem, one CPU thread is usually faster.
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
    # Monitor records episodic returns and lengths for diagnostics.
    env = Monitor(env)

    action_dim = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim, dtype=np.float32),
        # Exploration scale: 10% of maximum power per action dimension.
        sigma=(0.1 * np.ones(action_dim, dtype=np.float32) * config.p_max),
    )

    model = DDPG(
        policy="MlpPolicy",
        env=env,
        # Actor and critic are MLPs with this shared architecture template.
        policy_kwargs={"net_arch": [config.hidden_size, config.hidden_size]},
        learning_rate=1e-3,
        # Replay buffer and sampling controls:
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        # Target network update rate (Polyak averaging coefficient).
        tau=0.005,
        gamma=0.99,
        # Optimization schedule:
        train_freq=(config.train_freq_steps, "step"),
        gradient_steps=config.gradient_steps,
        action_noise=action_noise,
        verbose=config.verbose,
        seed=config.seed,
        device=config.device,
    )

    # The internal training loop alternates:
    # collect step -> store in replay buffer -> (if due) sample batch -> update.
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)

    # Episode rewards are extracted from Monitor for convergence plotting.
    episode_rewards = [float(x) for x in env.get_episode_rewards()]

    model.save(config.model_path)
    np.save(config.rewards_path, np.array(episode_rewards, dtype=np.float32))
    plot_training_rewards(episode_rewards, config.training_curve_path)
    return model, episode_rewards


def quick_policy_report(model: DDPG, config: TrainConfig, n_eval_steps: int) -> None:
    """
    Compute a lightweight post-training report on unseen channel realizations.

    This is a fast sanity-check report (not a full benchmark script).
    """
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

    for _ in range(n_eval_steps):
        # Deterministic action removes exploration noise during reporting.
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        sum_rates.append(float(info["sum_rate"]))
        total_powers.append(float(info["total_power"]))
        fairness_values.append(float(info["fairness"]))
        if terminated or truncated:
            obs, _ = env.reset()

    print("\nFinal learned DDPG policy performance (quick report):")
    print(f"  Average reward      : {np.mean(rewards):.4f}")
    print(f"  Average sum rate    : {np.mean(sum_rates):.4f}")
    print(f"  Average total power : {np.mean(total_powers):.4f}")
    print(f"  Average fairness    : {np.mean(fairness_values):.4f}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for convenient experiment control.
    """
    parser = argparse.ArgumentParser(description="Train DDPG for uplink power control.")
    parser.add_argument(
        "--mode",
        choices=["quick", "fast", "balanced", "full"],
        default="balanced",
        help="quick/fast: fastest, balanced: speed-quality tradeoff, full: 50k steps.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device for DDPG.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Torch CPU threads (1 is often fastest for this small model).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Override number of steps used in final quick policy report.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip final policy report to reduce runtime.",
    )
    return parser.parse_args()


def apply_mode_preset(cfg: TrainConfig, mode: str) -> None:
    """
    Apply predefined training presets.

    Presets alter timesteps and optimization intensity while preserving
    the same environment and algorithmic structure.
    """
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
        cfg.total_timesteps = 20_000
        cfg.episode_length = 400
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


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig()
    apply_mode_preset(cfg, args.mode)
    cfg.device = args.device
    cfg.cpu_threads = max(1, int(args.cpu_threads))

    if args.timesteps is not None:
        cfg.total_timesteps = int(args.timesteps)

    if args.eval_steps is not None:
        cfg.report_steps = int(args.eval_steps)

    start = time.perf_counter()
    trained_model, episodic_rewards = train_ddpg(cfg)
    elapsed = time.perf_counter() - start
    print(f"\nTraining complete. Episodes recorded: {len(episodic_rewards)}")
    print(f"Training time (s): {elapsed:.2f}")
    print(f"Saved model: {os.path.abspath(cfg.model_path)}.zip")
    print(f"Saved training curve: {os.path.abspath(cfg.training_curve_path)}")
    if not args.skip_report:
        quick_policy_report(trained_model, cfg, n_eval_steps=cfg.report_steps)
