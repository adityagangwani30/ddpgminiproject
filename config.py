"""Centralized project configuration for power-control experiments.

Why this file exists:
1) Keep all default hyperparameters in one place for reproducibility.
2) Avoid hard-coding values repeatedly across training/evaluation/demo scripts.
3) Provide consistent output naming and directory structure for artifacts.

Design note:
- Script-level CLI arguments may override these defaults at runtime.
- Values here represent the baseline experiment configuration.
"""

from __future__ import annotations

from pathlib import Path

# Environment defaults used by UplinkPowerControlEnv
N_USERS: int = 4
P_MAX: float = 1.0
NOISE_POWER: float = 1e-2
LAMBDA_POWER: float = 0.1
EPISODE_LENGTH: int = 400

# Training defaults used by train_ddpg.py
TRAINING_TIMESTEPS: int = 20_000
LEARNING_RATE: float = 1e-3
RANDOM_SEED: int = 42

# Evaluation defaults used by evaluation.py
EVAL_STEPS: int = 10_000
EVAL_RUNS: int = 3

# Model artifact names
MODEL_NAME: str = "ddpg_power_control_model"
DEPLOY_MODEL_NAME: str = "trained_ddpg_model"

# Figure and summary file names used across scripts
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
    """Create the results directory if needed and return it.

    Keeping this in one helper avoids duplicate mkdir logic in many scripts.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def results_path(filename: str) -> Path:
    """Return a path inside the results directory.

    This is the preferred location for newly generated artifacts.
    """
    return ensure_results_dir() / filename


def legacy_path(filename: str) -> Path:
    """Return a backward-compatible artifact path in project root.

    Some existing workflows expect files in the repository root, so scripts
    optionally mirror outputs here as a compatibility layer.
    """
    return LEGACY_OUTPUT_DIR / filename
