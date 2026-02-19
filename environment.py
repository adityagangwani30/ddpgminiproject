"""
Custom Gymnasium environment for DDPG-based uplink power control.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def compute_sinr_rates(
    gains: np.ndarray, powers: np.ndarray, noise_power: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-user SINR and spectral efficiency for a single-cell uplink.
    """
    gains = np.asarray(gains, dtype=np.float32)
    powers = np.asarray(powers, dtype=np.float32)
    noise_power = np.float32(noise_power)

    received_signal = powers * gains
    total_received = np.sum(received_signal, dtype=np.float32)
    interference = total_received - received_signal
    sinr = received_signal / (interference + noise_power + 1e-12)
    rates = np.log2(1.0 + sinr)
    return sinr, rates


def jain_fairness(values: np.ndarray) -> float:
    """
    Compute Jain's fairness index.
    """
    values = np.asarray(values, dtype=np.float64)
    numerator = np.square(np.sum(values))
    denominator = values.size * np.sum(np.square(values)) + 1e-12
    return float(numerator / denominator)


class UplinkPowerControlEnv(gym.Env):
    """
    State: channel gains |h_i|^2 for K users.
    Action: continuous user powers P_i in [0, P_max].
    Reward: sum_i log2(1 + SINR_i) - lambda_power * sum_i P_i.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_users: int = 4,
        p_max: float = 1.0,
        noise_power: float = 1e-2,
        lambda_power: float = 0.1,
        episode_length: int = 200,
        detailed_info: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if n_users < 3 or n_users > 5:
            raise ValueError("n_users must be between 3 and 5.")

        self.n_users = n_users
        self.p_max = float(p_max)
        self.noise_power = float(noise_power)
        self.lambda_power = float(lambda_power)
        self.episode_length = int(episode_length)
        self.detailed_info = bool(detailed_info)

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(self.n_users,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=self.p_max,
            shape=(self.n_users,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self.current_gains = np.zeros(self.n_users, dtype=np.float32)
        self.timestep = 0

    def _sample_channel_gains(self) -> np.ndarray:
        # If h_i ~ CN(0,1), then |h_i|^2 ~ Exp(mean=1). Sampling directly is equivalent.
        gains = self.rng.exponential(scale=1.0, size=self.n_users)
        return gains.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.timestep = 0
        self.current_gains = self._sample_channel_gains()
        return self.current_gains.copy(), {}

    def step(self, action: np.ndarray):
        self.timestep += 1
        powers = np.clip(np.asarray(action, dtype=np.float32), 0.0, self.p_max)

        sinr, rates = compute_sinr_rates(
            gains=self.current_gains, powers=powers, noise_power=self.noise_power
        )
        sum_rate = float(np.sum(rates))
        total_power = float(np.sum(powers))
        reward = sum_rate - self.lambda_power * total_power

        fairness = jain_fairness(rates)
        info = {
            "sum_rate": sum_rate,
            "total_power": total_power,
            "fairness": fairness,
        }
        if self.detailed_info:
            info["sinr"] = sinr
            info["rates"] = rates
            info["powers"] = powers
            info["gains"] = self.current_gains.copy()

        terminated = self.timestep >= self.episode_length
        truncated = False

        self.current_gains = self._sample_channel_gains()
        return self.current_gains.copy(), float(reward), terminated, truncated, info
