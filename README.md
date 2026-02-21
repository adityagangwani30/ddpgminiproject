# DDPG Based Adaptive Transmit Power Control in a Single-Cell Wireless Network

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.3%2B-orange.svg)](https://github.com/DLR-RM/stable-baselines3)

## Project Overview
This project addresses adaptive uplink transmit power control in a single-cell wireless network.  
A centralized controller (base station) observes channel conditions and allocates user transmit powers to optimize network performance.

The objective is to maximize spectral efficiency while penalizing excessive power usage.  
Adaptive power control is important because wireless channels vary over time and static policies cannot consistently balance throughput, energy usage, and fairness.

Deep Deterministic Policy Gradient (DDPG) is used because the action space (per-user transmit powers) is continuous, and DDPG is well-suited for continuous control with actor-critic learning.

## System Model
We consider a single-cell uplink scenario with \(K\) users.

Channel fading:
\[
h_i \sim \mathcal{CN}(0,1)
\]

SINR for user \(i\):
\[
\mathrm{SINR}_i = \frac{P_i |h_i|^2}{\sum_{j \ne i} P_j |h_j|^2 + \sigma^2}
\]

Spectral efficiency:
\[
R_i = \log_2(1 + \mathrm{SINR}_i)
\]

Reward:
\[
\mathrm{Reward} = \sum_{i=1}^{K} R_i - \lambda \sum_{i=1}^{K} P_i
\]

Jain's fairness index:
\[
J = \frac{\left(\sum_{i=1}^{K} R_i\right)^2}{K \sum_{i=1}^{K} R_i^2}
\]

## DDPG Formulation
- `Agent`: centralized controller at the base station.
- `State`: current channel gain vector \([|h_1|^2, \dots, |h_K|^2]\).
- `Action`: continuous transmit power vector \([P_1, \dots, P_K]\), clipped to \([0, P_{\max}]\).
- `Reward`: throughput-power tradeoff shown above.
- `Actor-Critic`: actor outputs deterministic power allocation; critic estimates \(Q(s,a)\) to guide policy updates through deterministic policy gradients.

## Project Structure
```text
project/
|
|-- environment.py
|-- baselines.py
|-- train_ddpg.py
|-- evaluation.py
|-- demo.py
|-- config.py
|-- requirements.txt
`-- README.md
```

## Installation
```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

## Run Instructions
Training:
```bash
python train_ddpg.py
```

Evaluation:
```bash
python evaluation.py
```

Demo (live channel/power adaptation visualization):
```bash
python demo.py
```

## Google Colab
A Colab-ready notebook is provided as:
- `Run_Project_On_Colab.ipynb`

Typical usage in Colab:
1. Open the notebook in Google Colab.
2. Set repository path (clone or uploaded project).
3. Run all cells from top to bottom to install dependencies, train, evaluate, and visualize results.

## Results
The repository evaluation pipeline reports and visualizes:
- `Convergence`: training reward trend over episodes/timesteps.
- `Sum-rate comparison`: DDPG vs Equal/Fractional/Greedy baselines.
- `Fairness`: Jain's fairness index comparison across methods.
- `Power efficiency`: \(\text{Sum Rate} / \text{Total Power}\).
- `User-rate CDF`: distribution-level behavior for each method.

## Future Work
- Extend to multi-cell interference coordination.
- Include time-correlated fading and mobility models.
- Compare with additional RL algorithms (TD3, SAC, PPO).
- Add confidence intervals across more random seeds.
- Explore constrained and multi-objective RL formulations.

## License
This project is released under the MIT License.
