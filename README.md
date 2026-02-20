# DDPG Based Adaptive Transmit Power Control in a Single-Cell Wireless Network

## Project Overview
This project studies adaptive uplink transmit power control for a single-cell wireless network using Deep Deterministic Policy Gradient (DDPG).  
The base station acts as a centralized controller and decides continuous transmit powers for all users at each time step.

The objective is to maximize total spectral efficiency while penalizing excessive transmit power, which creates a practical throughput-versus-energy tradeoff.

## Directory Structure
```text
Code/
|-- environment.py
|-- baselines.py
|-- train_ddpg.py
|-- evaluation.py
|-- requirements.txt
|-- run_on_colab.ipynb
|-- README.md
|-- .gitignore
|-- .gitattributes
|-- ddpg_power_control_model.zip            # generated after training
|-- training_episode_rewards.npy            # generated after training
|-- training_reward_curve.png               # generated after training
|-- sum_rate_comparison.png                 # generated after evaluation
|-- power_efficiency_comparison.png         # generated after evaluation
```

## System Model
We consider one base station and \(K\) single-antenna uplink users (\(3 \le K \le 5\)).

### Channel Model (Rayleigh Fading)
The complex channel coefficient for user \(i\) is:

$$
h_i \sim \mathcal{CN}(0,1)
$$

Hence, the channel power gain is \(|h_i|^2\), and channels are resampled every time step (time-varying fading).

### SINR
For user \(i\), the uplink SINR is:

$$
\mathrm{SINR}_i
=
\frac{P_i |h_i|^2}
{\sum_{j \ne i} P_j |h_j|^2 + \sigma^2}
$$

where:
- \(P_i\): transmit power of user \(i\),
- \(\sigma^2\): AWGN noise power.

### Spectral Efficiency
Per-user spectral efficiency is:

$$
R_i = \log_2(1+\mathrm{SINR}_i)
$$

### Reward Function
The RL reward at each time step is:

$$
\mathrm{Reward}
=
\sum_{i=1}^{K} R_i
- \lambda \sum_{i=1}^{K} P_i
$$

The first term pushes the controller toward high throughput, while the penalty term discourages unnecessary power usage.

### Fairness Metric (Jain's Index)
Fairness over user rates is measured by:

$$
J
=
\frac{\left(\sum_{i=1}^{K} R_i\right)^2}
{K \sum_{i=1}^{K} R_i^2}
$$

Values close to 1 indicate fair rate distribution; lower values indicate imbalance.

## RL Formulation
- Agent: centralized power controller at the base station.
- State: \([|h_1|^2, |h_2|^2, \dots, |h_K|^2]\).
- Action: continuous power vector \([P_1, P_2, \dots, P_K]\).
- Action bounds: \(0 \le P_i \le P_{\max}\).

## DDPG Explanation
DDPG is an off-policy actor-critic algorithm for continuous control.

### Actor Network
- Input: current state (channel gains).
- Output: continuous action (power allocation vector).
- Role: learns a deterministic policy \(\mu(s)\).

### Critic Network
- Input: state-action pair \((s,a)\).
- Output: Q-value \(Q(s,a)\), i.e., expected long-term return.
- Role: evaluates actor decisions.

### Replay Buffer
- Stores transitions \((s_t, a_t, r_t, s_{t+1})\).
- Breaks temporal correlation by sampling random mini-batches.
- Improves data efficiency via off-policy reuse.

### Target Networks
- DDPG uses slowly updated target actor and target critic.
- Soft update (Polyak averaging) stabilizes training:
  - online parameters change quickly,
  - target parameters track them smoothly.

### Policy Update
- Critic is updated by TD error minimization.
- Actor is updated through deterministic policy gradient to maximize critic-estimated Q-values.
- Gaussian action noise is added during training for exploration.

## Baseline Methods
The following non-learning methods are implemented for comparison:

### 1. Equal Power Allocation
All users transmit at the same power (here, \(P_{\max}\)).

### 2. Fractional Power Control
Power is inversely adjusted according to channel gain with exponent \(\alpha=0.5\), balancing compensation and stability.

### 3. Greedy SINR Method
Full power is assigned to the strongest channel user, while other users get a small residual power.

## Code Modules
### `environment.py`
- Custom Gymnasium environment.
- Channel sampling, state generation, power clipping, SINR/rate computation, and reward calculation.
- Includes Jain fairness utility.

### `baselines.py`
- Baseline power-control policies.
- Baseline evaluation routine over shared channel realizations.

### `train_ddpg.py`
- DDPG model creation and training loop via Stable-Baselines3.
- Uses replay buffer, target network updates, and normal action noise.
- Saves trained model, reward history, and convergence plot.

### `evaluation.py`
- Loads trained model and compares with baselines.
- Reports average sum rate, power usage, fairness, and power efficiency.
- Generates comparison plots.

## Results and Outputs
After running training and evaluation, the project produces:

### 1. Convergence Curve
- `training_reward_curve.png`
- Shows episode reward trend across training.

### 2. Sum Rate Comparison
- `sum_rate_comparison.png`
- Bar chart comparing average system sum rate across DDPG and baselines.

### 3. Power Efficiency Comparison
- `power_efficiency_comparison.png`
- Bar chart comparing throughput-per-power performance.

### 4. Fairness Comparison
- Jain fairness index is printed for each method in evaluation logs.
- This allows direct fairness comparison alongside throughput and power metrics.

## Installation and Run Instructions
Use Python 3.10+.

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train DDPG model
```bash
python train_ddpg.py
```

Optional modes:
```bash
python train_ddpg.py --mode quick
python train_ddpg.py --mode balanced
python train_ddpg.py --mode full
```

### 3) Evaluate DDPG against baselines
```bash
python evaluation.py
```

## Google Colab
Use the provided notebook:
- `run_on_colab.ipynb`

Direct link:
- https://colab.research.google.com/github/adityagangwani30/ddpgminiproject/blob/main/run_on_colab.ipynb

## Notes for Reproducibility
- Fixed random seeds are used in training/evaluation scripts.
- For fair baseline comparison, all methods are evaluated on the same channel sequence.
- Ensure the same Python environment is used for installation and execution.
