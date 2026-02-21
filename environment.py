"""
Wireless uplink environment for DDPG-based power control.

System model summary:
- Single-cell uplink with one base station and K users (3 <= K <= 5).
- Flat fading channel with per-user coefficient h_i.
- Rayleigh fading assumption: h_i ~ CN(0, 1), therefore |h_i|^2 ~ Exp(mean=1).
- Continuous transmit powers with box constraints: 0 <= P_i <= P_max.

At each time step:
1) A power vector is selected by the agent.
2) SINR and spectral efficiency are computed for all users.
3) Reward is returned as
   sum_i log2(1 + SINR_i) - lambda_power * sum_i P_i.
4) A new independent fading sample is generated for the next state.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as project_config


def compute_sinr_rates(
    gains: np.ndarray, powers: np.ndarray, noise_power: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-user SINR and spectral efficiency for one uplink slot.

    Mathematical model for user i:
        SINR_i = (P_i * |h_i|^2) / (sum_{j != i} P_j * |h_j|^2 + sigma^2)
        R_i    = log2(1 + SINR_i)

    Args:
        gains: Vector [|h_1|^2, ..., |h_K|^2].
        powers: Vector [P_1, ..., P_K].
        noise_power: AWGN variance sigma^2.

    Returns:
        sinr: Per-user SINR values.
        rates: Per-user spectral efficiencies in bps/Hz.
    """
    # Convert to float32 so numerical behavior matches the Gym action/state dtype.
    # This also reduces memory bandwidth compared with float64.
    gains = np.asarray(gains, dtype=np.float32)
    powers = np.asarray(powers, dtype=np.float32)
    noise_power = np.float32(noise_power)

    # Received power per user at the base station: P_i * g_i.
    # We compute total received once and subtract each user's own signal to
    # obtain interference from all other users efficiently.
    received_signal = powers * gains
    total_received = np.sum(received_signal, dtype=np.float32)
    interference = total_received - received_signal

    # SINR follows the uplink expression:
    #   signal / (interference + noise)
    # Small epsilon is a numerical guard for degenerate corner cases.
    sinr = received_signal / (interference + noise_power + 1e-12)
    # Spectral efficiency uses Shannon mapping in bps/Hz.
    rates = np.log2(1.0 + sinr)
    return sinr, rates


def jain_fairness(values: np.ndarray) -> float:
    """
    Compute Jain's fairness index for a non-negative performance vector.

    For x = [x_1, ..., x_K]:
        J(x) = (sum_i x_i)^2 / (K * sum_i x_i^2)

    Interpretation:
    - J close to 1: highly fair distribution.
    - J close to 1/K: highly unequal distribution.

    Args:
        values: Typically per-user rates.

    Returns:
        Jain fairness index in (0, 1].
    """
    values = np.asarray(values, dtype=np.float64)
    numerator = np.square(np.sum(values))
    denominator = values.size * np.sum(np.square(values)) + 1e-12
    return float(numerator / denominator)


class UplinkPowerControlEnv(gym.Env):
    """
    Gymnasium environment for centralized uplink power allocation.

    RL formulation:
    - State s_t: channel gain vector [|h_1|^2, ..., |h_K|^2].
    - Action a_t: power vector [P_1, ..., P_K], each bounded in [0, P_max].
    - Reward r_t: sum spectral efficiency minus weighted total power.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_users: int = project_config.N_USERS,
        p_max: float = project_config.P_MAX,
        noise_power: float = project_config.NOISE_POWER,
        lambda_power: float = project_config.LAMBDA_POWER,
        episode_length: int = project_config.EPISODE_LENGTH,
        detailed_info: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Initialize environment configuration and Gym spaces.

        Args:
            n_users: Number of uplink users (restricted to 3-5 in this project).
            p_max: Maximum transmit power per user.
            noise_power: AWGN variance sigma^2.
            lambda_power: Reward penalty coefficient for power usage.
            episode_length: Number of time steps before episode termination.
            detailed_info: If True, return richer per-step debug statistics.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        if n_users < 3 or n_users > 5:
            raise ValueError("n_users must be between 3 and 5.")

        self.n_users = n_users
        self.p_max = float(p_max)
        self.noise_power = float(noise_power)
        self.lambda_power = float(lambda_power)
        self.episode_length = int(episode_length)
        self.detailed_info = bool(detailed_info)

        # State construction:
        #   observation = [g_1, ..., g_K] with g_i = |h_i|^2 >= 0.
        # Infinite theoretical upper bound is represented with float32 max.
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(self.n_users,),
            dtype=np.float32,
        )
        # Action construction:
        #   action = [P_1, ..., P_K] with each P_i constrained in [0, P_max].
        # This encodes physical transmit-power limits directly in the space.
        self.action_space = spaces.Box(
            low=0.0,
            high=self.p_max,
            shape=(self.n_users,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        # Current channel realization s_t; updated each step/reset.
        self.current_gains = np.zeros(self.n_users, dtype=np.float32)
        # Episode time index used for fixed-horizon termination.
        self.timestep = 0

    def _sample_channel_gains(self) -> np.ndarray:
        """
        Sample one channel-gain vector for the next state.

        Instead of explicitly sampling complex Gaussian h_i and squaring magnitude,
        we directly sample |h_i|^2 from Exp(1), which is equivalent under
        h_i ~ CN(0,1). This is computationally cheaper and mathematically exact.
        """
        gains = self.rng.exponential(scale=1.0, size=self.n_users)
        return gains.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Start a new episode and return the initial state.

        Args:
            seed: Optional seed to reinitialize RNG.
            options: Gym API placeholder (unused in this environment).

        Returns:
            observation: Initial channel-gain vector.
            info: Empty dictionary by design.
        """
        super().reset(seed=seed)
        # Optional reseeding makes repeated experiments deterministic when needed.
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Episode always starts with a fresh channel draw.
        self.timestep = 0
        self.current_gains = self._sample_channel_gains()
        return self.current_gains.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Advance one time step using the selected power allocation.

        Step logic:
        1) Action clipping enforces physical power constraints.
        2) SINR and rates are computed using the current channel realization.
        3) Reward combines throughput gain and power penalty.
        4) Termination is checked by fixed horizon.
        5) Next state is sampled as a fresh independent fading realization.
        """
        self.timestep += 1

        # Even with exploration noise, clipping guarantees physically valid powers.
        powers = np.clip(np.asarray(action, dtype=np.float32), 0.0, self.p_max)

        # Compute instantaneous link quality and per-user rates for this slot.
        sinr, rates = compute_sinr_rates(
            gains=self.current_gains, powers=powers, noise_power=self.noise_power
        )
        sum_rate = float(np.sum(rates))
        total_power = float(np.sum(powers))

        # Reward logic:
        # - maximize spectral efficiency (sum_rate),
        # - penalize high energy usage (lambda_power * total_power).
        # This scalar reward defines the throughput-energy tradeoff objective.
        reward = sum_rate - self.lambda_power * total_power

        # Fairness is logged for analysis; it is not directly optimized here.
        fairness = jain_fairness(rates)
        info: dict[str, Any] = {
            "sum_rate": sum_rate,
            "total_power": total_power,
            "fairness": fairness,
        }
        # Optional debug payloads are useful for analysis scripts and live demos.
        if self.detailed_info:
            info["sinr"] = sinr
            info["rates"] = rates
            info["powers"] = powers
            info["gains"] = self.current_gains.copy()

        terminated = self.timestep >= self.episode_length
        truncated = False

        # Independent block fading model: new sample every slot.
        self.current_gains = self._sample_channel_gains()
        return self.current_gains.copy(), float(reward), terminated, truncated, info
