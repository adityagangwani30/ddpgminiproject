"""
Baseline power-control policies for uplink power allocation.
"""

from __future__ import annotations

import numpy as np

from environment import compute_sinr_rates, jain_fairness


def equal_power_allocation(n_users: int, p_max: float) -> np.ndarray:
    """
    Equal power for all users at full-scale level.
    """
    return np.full(shape=(n_users,), fill_value=p_max, dtype=np.float32)


def fractional_power_control(
    gains: np.ndarray, p_max: float, alpha: float = 0.5
) -> np.ndarray:
    """
    Fractional power control based on inverse channel gain.
    alpha = 0 -> equal power, alpha = 1 -> full channel inversion trend.
    """
    gains = np.asarray(gains, dtype=np.float64)
    inv_term = np.power(gains + 1e-12, -alpha)
    inv_term = inv_term / (np.max(inv_term) + 1e-12)
    powers = p_max * inv_term
    return np.clip(powers, 0.0, p_max).astype(np.float32)


def greedy_sinr_allocation(gains: np.ndarray, p_max: float) -> np.ndarray:
    """
    Greedy SINR-based rule:
    Allocate maximum power to the strongest user and minimal power to others.
    """
    gains = np.asarray(gains, dtype=np.float64)
    n_users = gains.size
    powers = np.full(n_users, 0.05 * p_max, dtype=np.float64)
    best_user = int(np.argmax(gains))
    powers[best_user] = p_max
    return np.clip(powers, 0.0, p_max).astype(np.float32)


def evaluate_static_policy_on_channels(
    channel_gains: np.ndarray,
    policy_name: str,
    p_max: float,
    noise_power: float,
) -> dict[str, float]:
    """
    Evaluate one baseline policy over a pre-generated channel sequence.
    """
    sum_rates = []
    total_powers = []
    fairness_values = []

    for gains in channel_gains:
        n_users = gains.size
        if policy_name == "equal":
            powers = equal_power_allocation(n_users=n_users, p_max=p_max)
        elif policy_name == "fractional":
            powers = fractional_power_control(gains=gains, p_max=p_max, alpha=0.5)
        elif policy_name == "greedy":
            powers = greedy_sinr_allocation(gains=gains, p_max=p_max)
        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")

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
