"""Centralized project configuration for power-control experiments."""

from __future__ import annotations

from pathlib import Path

# Environment defaults
N_USERS: int = 4
P_MAX: float = 1.0
NOISE_POWER: float = 1e-2
LAMBDA_POWER: float = 0.1
EPISODE_LENGTH: int = 400

# Training defaults
TRAINING_TIMESTEPS: int = 20_000
LEARNING_RATE: float = 1e-3
RANDOM_SEED: int = 42

# Evaluation defaults
EVAL_STEPS: int = 10_000
EVAL_RUNS: int = 3

# Artifact names
MODEL_NAME: str = "ddpg_power_control_model"
DEPLOY_MODEL_NAME: str = "trained_ddpg_model"

TRAINING_REWARDS_FILENAME: str = "training_episode_rewards.npy"
TRAINING_CURVE_FILENAME: str = "training_reward_curve.png"
TRAINING_TIMESTEPS_PLOT_FILENAME: str = "training_reward_vs_timesteps.png"
SUM_RATE_PLOT_FILENAME: str = "sum_rate_comparison.png"
POWER_EFFICIENCY_PLOT_FILENAME: str = "power_efficiency_comparison.png"
POWER_USAGE_PLOT_FILENAME: str = "power_usage_comparison.png"
FAIRNESS_PLOT_FILENAME: str = "fairness_comparison.png"
RATE_CDF_PLOT_FILENAME: str = "user_rate_cdf.png"
EVAL_SUMMARY_CSV_FILENAME: str = "evaluation_summary.csv"

RESULTS_DIR: Path = Path("results")
LEGACY_OUTPUT_DIR: Path = Path(".")


def ensure_results_dir() -> Path:
    """Create the results directory if needed and return it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def results_path(filename: str) -> Path:
    """Return a path inside the results directory."""
    return ensure_results_dir() / filename


def legacy_path(filename: str) -> Path:
    """Return a backward-compatible artifact path in project root."""
    return LEGACY_OUTPUT_DIR / filename
