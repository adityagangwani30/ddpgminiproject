"""Evaluate DDPG and baseline power-control policies with publication-quality outputs."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG

import config as project_config
from baselines import (
    equal_power_allocation,
    fractional_power_control,
    greedy_sinr_allocation,
)
from environment import UplinkPowerControlEnv, compute_sinr_rates, jain_fairness

LOGGER = logging.getLogger(__name__)
METHOD_ORDER = ["Equal", "Fractional", "Greedy", "DDPG"]


def setup_logging(log_level: str) -> None:
    """Configure project logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_plot_style() -> None:
    """Apply a clean, publication-oriented Matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 120,
        }
    )


def copy_if_needed(source: Path, destination: Path) -> None:
    """Copy a file when source and destination are different paths."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def generate_channel_sequence(n_users: int, n_steps: int, seed: int) -> np.ndarray:
    """Generate a deterministic Rayleigh-fading gain sequence for fair comparisons."""
    rng = np.random.default_rng(seed)
    h_real = rng.normal(0.0, np.sqrt(0.5), size=(n_steps, n_users))
    h_imag = rng.normal(0.0, np.sqrt(0.5), size=(n_steps, n_users))
    h = h_real + 1j * h_imag
    gains = np.abs(h) ** 2
    return gains.astype(np.float32)


def summarize_metrics(
    sum_rates: list[float], total_powers: list[float], fairness_values: list[float]
) -> dict[str, float]:
    """Aggregate step-level metrics into method-level averages."""
    avg_sum_rate = float(np.mean(sum_rates))
    avg_total_power = float(np.mean(total_powers))
    avg_fairness = float(np.mean(fairness_values))
    power_efficiency = float(avg_sum_rate / (avg_total_power + 1e-12))
    return {
        "avg_sum_rate": avg_sum_rate,
        "avg_total_power": avg_total_power,
        "avg_fairness": avg_fairness,
        "power_efficiency": power_efficiency,
    }


def baseline_powers(policy_name: str, gains: np.ndarray, p_max: float) -> np.ndarray:
    """Return baseline power allocation for one channel realization."""
    if policy_name == "equal":
        return equal_power_allocation(n_users=int(gains.size), p_max=p_max)
    if policy_name == "fractional":
        return fractional_power_control(gains=gains, p_max=p_max, alpha=0.5)
    if policy_name == "greedy":
        return greedy_sinr_allocation(gains=gains, p_max=p_max)
    raise ValueError(f"Unknown policy_name: {policy_name}")


def evaluate_baseline_on_channels(
    channel_gains: np.ndarray,
    policy_name: str,
    p_max: float,
    noise_power: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate one baseline policy on a fixed channel sequence."""
    sum_rates: list[float] = []
    total_powers: list[float] = []
    fairness_values: list[float] = []
    rate_samples: list[np.ndarray] = []

    for gains in channel_gains:
        powers = baseline_powers(policy_name=policy_name, gains=gains, p_max=p_max)
        _, rates = compute_sinr_rates(gains=gains, powers=powers, noise_power=noise_power)
        sum_rates.append(float(np.sum(rates)))
        total_powers.append(float(np.sum(powers)))
        fairness_values.append(jain_fairness(rates))
        rate_samples.append(rates.astype(np.float32))

    return summarize_metrics(sum_rates, total_powers, fairness_values), np.vstack(rate_samples)


def evaluate_ddpg_on_channels(
    model: DDPG,
    channel_gains: np.ndarray,
    p_max: float,
    noise_power: float,
    return_rates: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """Evaluate deterministic DDPG actions on a fixed channel sequence."""
    sum_rates: list[float] = []
    total_powers: list[float] = []
    fairness_values: list[float] = []
    rate_samples: list[np.ndarray] = []

    for gains in channel_gains:
        action, _ = model.predict(gains, deterministic=True)
        powers = np.clip(np.asarray(action, dtype=np.float32), 0.0, p_max)
        _, rates = compute_sinr_rates(gains=gains, powers=powers, noise_power=noise_power)
        sum_rates.append(float(np.sum(rates)))
        total_powers.append(float(np.sum(powers)))
        fairness_values.append(jain_fairness(rates))
        rate_samples.append(rates.astype(np.float32))

    metrics = summarize_metrics(sum_rates, total_powers, fairness_values)
    if return_rates:
        return metrics, np.vstack(rate_samples)
    return metrics


def aggregate_run_metrics(
    per_run_metrics: list[dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    """Average method metrics across multiple independent evaluation runs."""
    aggregated: dict[str, dict[str, float]] = {}
    for method in METHOD_ORDER:
        aggregated[method] = {
            "avg_sum_rate": float(np.mean([run[method]["avg_sum_rate"] for run in per_run_metrics])),
            "avg_total_power": float(
                np.mean([run[method]["avg_total_power"] for run in per_run_metrics])
            ),
            "avg_fairness": float(np.mean([run[method]["avg_fairness"] for run in per_run_metrics])),
            "power_efficiency": float(
                np.mean([run[method]["power_efficiency"] for run in per_run_metrics])
            ),
        }
    return aggregated


def resolve_model_path(model_arg: str | None) -> Path:
    """Resolve model path from CLI input or project defaults."""
    candidates: list[Path] = []
    if model_arg:
        model_path = Path(model_arg)
        candidates.append(model_path)
        if model_path.suffix != ".zip":
            candidates.append(Path(f"{model_arg}.zip"))
    else:
        candidates.append(Path(f"{project_config.DEPLOY_MODEL_NAME}.zip"))
        candidates.append(Path(f"{project_config.MODEL_NAME}.zip"))

    for path in candidates:
        if path.exists():
            return path

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No model file found. Tried: {tried}")


def build_table(results: dict[str, dict[str, float]]) -> str:
    """Build a professional fixed-width comparison table."""
    lines = [
        "---------------------------------------------",
        "Method      | Sum Rate | Power | Fairness",
        "---------------------------------------------",
    ]
    for method in METHOD_ORDER:
        metrics = results[method]
        lines.append(
            f"{method:<11} | {metrics['avg_sum_rate']:>8.4f} | {metrics['avg_total_power']:>5.4f} | {metrics['avg_fairness']:>8.4f}"
        )
    lines.append("---------------------------------------------")
    return "\n".join(lines)


def print_results(results: dict[str, dict[str, float]]) -> None:
    """Log tabular comparison and power-efficiency values."""
    table = build_table(results)
    LOGGER.info("\n%s", table)

    LOGGER.info("Power Efficiency (Sum Rate / Total Power):")
    for method in METHOD_ORDER:
        LOGGER.info("  %-10s : %.4f", method, results[method]["power_efficiency"])


def save_results_csv(results: dict[str, dict[str, float]], output_path: Path) -> None:
    """Save averaged evaluation metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "method",
                "avg_sum_rate",
                "avg_total_power",
                "avg_fairness",
                "power_efficiency",
            ],
        )
        writer.writeheader()
        for method in METHOD_ORDER:
            row = {"method": method}
            row.update(results[method])
            writer.writerow(row)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save each figure to results/ and backward-compatible root path."""
    results_output = project_config.results_path(filename)
    legacy_output = project_config.legacy_path(filename)

    results_output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(results_output, dpi=300)
    copy_if_needed(results_output, legacy_output)
    plt.close(fig)


def plot_bar_metric(
    results: dict[str, dict[str, float]],
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
    color: str,
) -> None:
    """Create a publication-style bar chart for one metric."""
    values = [results[m][metric_key] for m in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, values, color=color, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(fig, filename)


def plot_rate_cdf(rate_samples: dict[str, np.ndarray], filename: str) -> None:
    """Plot CDF of per-user rates for all methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "Equal": "#4C72B0",
        "Fractional": "#55A868",
        "Greedy": "#C44E52",
        "DDPG": "#8172B3",
    }

    for method in METHOD_ORDER:
        rates = np.sort(rate_samples[method].ravel())
        if rates.size == 0:
            continue
        cdf = np.arange(1, rates.size + 1) / rates.size
        ax.plot(rates, cdf, label=method, linewidth=2.0, color=colors[method])

    ax.set_xlabel("Per-user spectral efficiency (bps/Hz)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of User Rates")
    ax.legend()
    save_figure(fig, filename)


def load_training_rewards() -> np.ndarray | None:
    """Load episodic reward history from results/ or root fallback."""
    candidates = [
        project_config.results_path(project_config.TRAINING_REWARDS_FILENAME),
        project_config.legacy_path(project_config.TRAINING_REWARDS_FILENAME),
    ]
    for path in candidates:
        if path.exists():
            rewards = np.load(path)
            if rewards.size > 0:
                return rewards.astype(np.float32)
    return None


def plot_training_reward_vs_timesteps(episode_length: int, filename: str) -> None:
    """Plot training reward against approximate timesteps using episode horizon."""
    rewards = load_training_rewards()
    if rewards is None:
        LOGGER.warning(
            "No non-empty training reward history found; skipping training reward vs timesteps plot."
        )
        return

    timesteps = np.arange(1, rewards.size + 1, dtype=np.int64) * int(episode_length)
    window = min(20, rewards.size)
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smooth_timesteps = timesteps[window - 1 :]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timesteps, rewards, alpha=0.35, label="Episode reward", color="#4C72B0")
    ax.plot(smooth_timesteps, smooth, linewidth=2.0, label=f"Moving average ({window})", color="#C44E52")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Training Reward vs Timesteps")
    ax.legend()

    save_figure(fig, filename)


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DDPG and baseline policies.")
    parser.add_argument("--users", type=int, default=project_config.N_USERS, help="Number of users.")
    parser.add_argument(
        "--p-max", type=float, default=project_config.P_MAX, help="Maximum power per user."
    )
    parser.add_argument(
        "--noise-power", type=float, default=project_config.NOISE_POWER, help="Noise power."
    )
    parser.add_argument(
        "--lambda-power",
        type=float,
        default=project_config.LAMBDA_POWER,
        help="Reward power-penalty coefficient.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=project_config.EPISODE_LENGTH,
        help="Episode horizon used for reward-vs-timestep mapping.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=project_config.EVAL_STEPS,
        help="Evaluation channel realizations per run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=project_config.EVAL_RUNS,
        help="Number of evaluation runs for averaging.",
    )
    parser.add_argument("--seed", type=int, default=project_config.RANDOM_SEED, help="Base random seed.")
    parser.add_argument("--model", type=str, default=None, help="Model path or base name.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for evaluation and result visualization."""
    args = parse_args()
    setup_logging(args.log_level)
    set_plot_style()

    model_path = resolve_model_path(args.model)
    LOGGER.info("Loading model from: %s", model_path.resolve())

    model = DDPG.load(
        str(model_path),
        env=UplinkPowerControlEnv(
            n_users=args.users,
            p_max=args.p_max,
            noise_power=args.noise_power,
            lambda_power=args.lambda_power,
            episode_length=args.episode_length,
            seed=args.seed,
        ),
    )

    per_run_metrics: list[dict[str, dict[str, float]]] = []
    per_method_rates: dict[str, list[np.ndarray]] = {method: [] for method in METHOD_ORDER}

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        channel_seq = generate_channel_sequence(
            n_users=args.users,
            n_steps=args.eval_steps,
            seed=run_seed,
        )

        ddpg_metrics, ddpg_rates = evaluate_ddpg_on_channels(
            model=model,
            channel_gains=channel_seq,
            p_max=args.p_max,
            noise_power=args.noise_power,
            return_rates=True,
        )
        equal_metrics, equal_rates = evaluate_baseline_on_channels(
            channel_gains=channel_seq,
            policy_name="equal",
            p_max=args.p_max,
            noise_power=args.noise_power,
        )
        fractional_metrics, fractional_rates = evaluate_baseline_on_channels(
            channel_gains=channel_seq,
            policy_name="fractional",
            p_max=args.p_max,
            noise_power=args.noise_power,
        )
        greedy_metrics, greedy_rates = evaluate_baseline_on_channels(
            channel_gains=channel_seq,
            policy_name="greedy",
            p_max=args.p_max,
            noise_power=args.noise_power,
        )

        run_metrics = {
            "Equal": equal_metrics,
            "Fractional": fractional_metrics,
            "Greedy": greedy_metrics,
            "DDPG": ddpg_metrics,
        }
        per_run_metrics.append(run_metrics)

        per_method_rates["Equal"].append(equal_rates)
        per_method_rates["Fractional"].append(fractional_rates)
        per_method_rates["Greedy"].append(greedy_rates)
        per_method_rates["DDPG"].append(ddpg_rates)

        LOGGER.info("Completed evaluation run %d/%d with seed=%d", run_idx + 1, args.runs, run_seed)

    averaged_results = aggregate_run_metrics(per_run_metrics)
    merged_rates = {
        method: np.concatenate(rate_batches, axis=0)
        for method, rate_batches in per_method_rates.items()
    }

    print_results(averaged_results)

    plot_training_reward_vs_timesteps(
        episode_length=args.episode_length,
        filename=project_config.TRAINING_TIMESTEPS_PLOT_FILENAME,
    )
    plot_bar_metric(
        results=averaged_results,
        metric_key="avg_sum_rate",
        title="Sum Rate Comparison",
        ylabel="Average sum rate (bps/Hz)",
        filename=project_config.SUM_RATE_PLOT_FILENAME,
        color="#4C72B0",
    )
    plot_bar_metric(
        results=averaged_results,
        metric_key="avg_total_power",
        title="Power Usage Comparison",
        ylabel="Average total transmit power",
        filename=project_config.POWER_USAGE_PLOT_FILENAME,
        color="#55A868",
    )
    plot_bar_metric(
        results=averaged_results,
        metric_key="avg_fairness",
        title="Jain Fairness Comparison",
        ylabel="Average Jain fairness index",
        filename=project_config.FAIRNESS_PLOT_FILENAME,
        color="#C44E52",
    )
    plot_bar_metric(
        results=averaged_results,
        metric_key="power_efficiency",
        title="Power Efficiency Comparison",
        ylabel="Sum rate / total power",
        filename=project_config.POWER_EFFICIENCY_PLOT_FILENAME,
        color="#8172B3",
    )
    plot_rate_cdf(merged_rates, filename=project_config.RATE_CDF_PLOT_FILENAME)

    csv_path = project_config.results_path(project_config.EVAL_SUMMARY_CSV_FILENAME)
    save_results_csv(averaged_results, csv_path)

    LOGGER.info("Saved plots (results + legacy root copy):")
    LOGGER.info("  %s", project_config.TRAINING_TIMESTEPS_PLOT_FILENAME)
    LOGGER.info("  %s", project_config.SUM_RATE_PLOT_FILENAME)
    LOGGER.info("  %s", project_config.POWER_USAGE_PLOT_FILENAME)
    LOGGER.info("  %s", project_config.FAIRNESS_PLOT_FILENAME)
    LOGGER.info("  %s", project_config.POWER_EFFICIENCY_PLOT_FILENAME)
    LOGGER.info("  %s", project_config.RATE_CDF_PLOT_FILENAME)
    LOGGER.info("Saved CSV summary: %s", csv_path.resolve())


if __name__ == "__main__":
    main()
