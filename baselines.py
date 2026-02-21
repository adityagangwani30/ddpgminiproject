"""
Baseline power-allocation rules for comparison against DDPG.

This module contains simple non-learning policies that map channel gains
to transmit powers. They provide interpretable references for:
- throughput efficiency,
- power consumption,
- and fairness behavior.
"""

from __future__ import annotations

import numpy as np

from environment import compute_sinr_rates, jain_fairness


def equal_power_allocation(n_users: int, p_max: float) -> np.ndarray:
    """
    Equal power baseline: all users transmit at the same maximum power.

    Rationale:
    - Simplest feasible allocation.
    - Often achieves reasonable throughput in symmetric settings.
    - Usually ignores fairness-through-interference tradeoffs.
    """
    # Every user gets identical power; no channel-awareness is used.
    return np.full(shape=(n_users,), fill_value=p_max, dtype=np.float32)


def fractional_power_control(
    gains: np.ndarray, p_max: float, alpha: float = 0.5
) -> np.ndarray:
    """
    Fractional power control (FPC) baseline.

    Idea:
    - Users with weak channel gains receive relatively higher power.
    - Users with strong channel gains receive relatively lower power.
    - Controlled by exponent alpha:
      alpha=0 behaves like equal power; alpha=1 approximates full inversion.

    Implementation:
    1) Compute (|h_i|^2)^(-alpha).
    2) Normalize by maximum value to keep outputs in [0, 1].
    3) Scale by p_max and clip for numerical safety.
    """
    gains = np.asarray(gains, dtype=np.float64)
    # Inverse-gain weighting compensates weaker channels.
    inv_term = np.power(gains + 1e-12, -alpha)
    # Normalize to keep relative profile while respecting [0, p_max] scale.
    inv_term = inv_term / (np.max(inv_term) + 1e-12)
    powers = p_max * inv_term
    return np.clip(powers, 0.0, p_max).astype(np.float32)


def greedy_sinr_allocation(gains: np.ndarray, p_max: float) -> np.ndarray:
    """
    Greedy SINR-style baseline.

    Rule:
    - Assign full power to the user with the strongest channel.
    - Assign small residual power (5% of p_max) to all others.

    Interpretation:
    - Aggressively maximizes immediate strongest-link throughput.
    - Can produce high sum rate in interference-limited scenarios.
    - Typically sacrifices fairness among users.
    """
    gains = np.asarray(gains, dtype=np.float64)
    n_users = gains.size
    # Start with small residual powers for all users.
    powers = np.full(n_users, 0.05 * p_max, dtype=np.float64)
    # Allocate full power to the strongest user to maximize immediate gain.
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
    Evaluate one baseline policy over a fixed sequence of channel gains.

    Using the same channel sequence for all methods is essential for fair
    comparison: differences in performance then come from policy behavior,
    not from random channel luck.

    Reported metrics:
    - average sum rate,
    - average total power,
    - average Jain fairness index (computed over per-user rates),
    - power efficiency = average_sum_rate / average_total_power.
    """
    sum_rates: list[float] = []
    total_powers: list[float] = []
    fairness_values: list[float] = []

    # Evaluate each channel realization independently and accumulate statistics.
    for gains in channel_gains:
        n_users = gains.size
        # Dispatch to the requested rule-based policy.
        if policy_name == "equal":
            powers = equal_power_allocation(n_users=n_users, p_max=p_max)
        elif policy_name == "fractional":
            powers = fractional_power_control(gains=gains, p_max=p_max, alpha=0.5)
        elif policy_name == "greedy":
            powers = greedy_sinr_allocation(gains=gains, p_max=p_max)
        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")

        # Compute per-user rates under the chosen baseline action.
        _, rates = compute_sinr_rates(
            gains=gains, powers=powers, noise_power=noise_power
        )
        sum_rates.append(float(np.sum(rates)))
        total_powers.append(float(np.sum(powers)))
        fairness_values.append(jain_fairness(rates))

    # Reduce step-wise metrics into one summary dictionary.
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
