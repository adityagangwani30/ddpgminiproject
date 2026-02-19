"""
Evaluate trained DDPG against baseline power-control methods.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from stable_baselines3 import DDPG

from baselines import evaluate_static_policy_on_channels
from environment import UplinkPowerControlEnv, compute_sinr_rates, jain_fairness


def generate_channel_sequence(
    n_users: int, n_steps: int, seed: int = 2026
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h_real = rng.normal(0.0, np.sqrt(0.5), size=(n_steps, n_users))
    h_imag = rng.normal(0.0, np.sqrt(0.5), size=(n_steps, n_users))
    h = h_real + 1j * h_imag
    gains = np.abs(h) ** 2
    return gains.astype(np.float32)


def evaluate_ddpg_on_channels(
    model: DDPG,
    channel_gains: np.ndarray,
    p_max: float,
    noise_power: float,
) -> dict[str, float]:
    sum_rates = []
    total_powers = []
    fairness_values = []

    for gains in channel_gains:
        action, _ = model.predict(gains, deterministic=True)
        powers = np.clip(np.asarray(action, dtype=np.float32), 0.0, p_max)
        _, rates = compute_sinr_rates(gains=gains, powers=powers, noise_power=noise_power)
        sum_rates.append(float(np.sum(rates)))
        total_powers.append(float(np.sum(powers)))
        fairness_values.append(jain_fairness(rates))

    avg_sum_rate = float(np.mean(sum_rates))
    avg_total_power = float(np.mean(total_powers))
    avg_fairness = float(np.mean(fairness_values))
    power_eff = float(avg_sum_rate / (avg_total_power + 1e-12))

    return {
        "avg_sum_rate": avg_sum_rate,
        "avg_total_power": avg_total_power,
        "avg_fairness": avg_fairness,
        "power_efficiency": power_eff,
    }


def plot_metric_bars(results: dict[str, dict[str, float]]) -> None:
    methods = list(results.keys())
    sum_rates = [results[m]["avg_sum_rate"] for m in methods]
    power_eff = [results[m]["power_efficiency"] for m in methods]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, sum_rates)
    plt.ylabel("Average sum rate (bps/Hz)")
    plt.title("Sum Rate Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("sum_rate_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(methods, power_eff)
    plt.ylabel("Power efficiency (sum-rate / power)")
    plt.title("Power Efficiency Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("power_efficiency_comparison.png", dpi=300)
    plt.close()


def print_results(results: dict[str, dict[str, float]]) -> None:
    print("\nPerformance comparison:")
    for method, metrics in results.items():
        print(f"\n{method}")
        print(f"  Average sum rate    : {metrics['avg_sum_rate']:.4f}")
        print(f"  Average total power : {metrics['avg_total_power']:.4f}")
        print(f"  Jain fairness index : {metrics['avg_fairness']:.4f}")
        print(f"  Power efficiency    : {metrics['power_efficiency']:.4f}")


if __name__ == "__main__":
    n_users = 4
    p_max = 1.0
    noise_power = 1e-2
    n_eval_steps = 10_000

    channel_seq = generate_channel_sequence(
        n_users=n_users, n_steps=n_eval_steps, seed=2026
    )

    model_path = Path("ddpg_power_control_model.zip")
    if not model_path.exists():
        raise FileNotFoundError(
            "Model file not found: ddpg_power_control_model.zip. "
            "Run train_ddpg.py first."
        )

    model = DDPG.load(
        str(model_path),
        env=UplinkPowerControlEnv(
            n_users=n_users,
            p_max=p_max,
            noise_power=noise_power,
            lambda_power=0.1,
            episode_length=200,
            seed=0,
        ),
    )

    results = {
        "DDPG": evaluate_ddpg_on_channels(
            model=model,
            channel_gains=channel_seq,
            p_max=p_max,
            noise_power=noise_power,
        ),
        "Equal Power": evaluate_static_policy_on_channels(
            channel_gains=channel_seq,
            policy_name="equal",
            p_max=p_max,
            noise_power=noise_power,
        ),
        "Fractional PC": evaluate_static_policy_on_channels(
            channel_gains=channel_seq,
            policy_name="fractional",
            p_max=p_max,
            noise_power=noise_power,
        ),
        "Greedy SINR": evaluate_static_policy_on_channels(
            channel_gains=channel_seq,
            policy_name="greedy",
            p_max=p_max,
            noise_power=noise_power,
        ),
    }

    print_results(results)
    plot_metric_bars(results)
    print("\nSaved plots:")
    print("  sum_rate_comparison.png")
    print("  power_efficiency_comparison.png")
