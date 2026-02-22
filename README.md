# DDPG Based Adaptive Transmit Power Control in a Single-Cell Wireless Network

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.3%2B-orange.svg)](https://github.com/DLR-RM/stable-baselines3)

## Project Overview
This repository implements an adaptive uplink transmit power controller for a single-cell wireless network using Deep Deterministic Policy Gradient (DDPG).

In uplink communication, each user's transmission creates interference for other users. Fixed power rules (always max power, fixed heuristics) cannot consistently handle channel fluctuations and interference coupling. This project learns a channel-aware policy that allocates continuous transmit powers to balance:
- throughput (sum spectral efficiency),
- energy usage (total transmit power),
- fairness (Jain's fairness index).

DDPG is used because power control is a continuous-action problem and actor-critic methods are effective for such settings.

## Why Adaptive Power Control
- Wireless channels are stochastic and time-varying due to fading.
- A static policy cannot exploit favorable channel states or conserve energy in poor states.
- Adaptive control improves long-term operating efficiency under realistic channel variability.

## System Model
We consider one base station and $K$ uplink users ($3 \le K \le 5$).

### 1) Channel Model (Rayleigh Fading)
$$
h_i \sim \mathcal{CN}(0,1), \qquad g_i = |h_i|^2
$$

### 2) Uplink SINR
$$
\mathrm{SINR}_i = \frac{P_i |h_i|^2}{\sum_{j \neq i} P_j |h_j|^2 + \sigma^2}
$$

### 3) Spectral Efficiency
$$
R_i = \log_2(1 + \mathrm{SINR}_i)
$$

### 4) Reward Function
$$
\mathrm{Reward} = \sum_{i=1}^{K} R_i - \lambda \sum_{i=1}^{K} P_i
$$

### 5) Jain's Fairness Index
$$
J = \frac{\left(\sum_{i=1}^{K} R_i\right)^2}{K \sum_{i=1}^{K} R_i^2}
$$

## RL/DDPG Formulation
- `Agent`: centralized controller at the base station.
- `State`: channel gain vector $[|h_1|^2, \dots, |h_K|^2]$.
- `Action`: continuous power vector $[P_1, \dots, P_K]$, clipped into $[0, P_{\max}]$.
- `Reward`: throughput-power tradeoff defined above.
- `Actor`: deterministic policy network $\mu(s)$ producing powers.
- `Critic`: Q-network $Q(s,a)$ evaluating action quality.
- `Training mechanics`: replay buffer, target networks, action noise for exploration.

## Repository Structure
```text
project/
|
|-- environment.py
|-- baselines.py
|-- train_ddpg.py
|-- evaluation.py
|-- generate_results_report.py
|-- demo.py
|-- config.py
|-- requirements.txt
|-- Run_Project_On_Colab.ipynb
|-- README.md
`-- results/
    |-- Results_Analysis_Report.pdf
    |-- evaluation_summary.csv
    `-- *.png  (all generated plots)
```

## Installation
### 1) Clone
```bash
git clone <repo-url>
cd project
```

### 2) (Recommended) Create virtual environment
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
Linux/macOS:
```bash
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run (Step-by-Step)
### Step 1: Train DDPG
Default training:
```bash
python train_ddpg.py
```

Custom training example:
```bash
python train_ddpg.py --users 4 --timesteps 100000 --learning-rate 1e-3 --seed 42
```

Quick smoke run:
```bash
python train_ddpg.py --mode quick --timesteps 8000
```

### Step 2: Evaluate DDPG vs Baselines
Default evaluation:
```bash
python evaluation.py
```

Custom evaluation example:
```bash
python evaluation.py --users 4 --eval-steps 10000 --runs 3 --seed 42
```

This prints a comparison table and saves figures/CSV.

### Step 3: Run Live Inference Demo
```bash
python demo.py --steps 200 --pause 0.1
```

The demo shows two real-time plots:
- top: current channel gains,
- bottom: DDPG allocated powers.

### Step 4: (Optional) Run in Google Colab
Open and run:
- `Run_Project_On_Colab.ipynb`

The notebook includes dependency installation, training, evaluation, inline plot display, and artifact export.

## Outputs
Training and evaluation generate artifacts such as:
- model checkpoints (`ddpg_power_control_model.zip`, `trained_ddpg_model.zip`),
- reward traces and convergence figures,
- sum-rate, power-usage, fairness, and power-efficiency comparisons,
- user-rate CDF plot,
- CSV summary (`evaluation_summary.csv`),
- **Results & Analysis PDF report** (`Results_Analysis_Report.pdf`).

All outputs are stored in the `results/` folder.

## Results Interpretation
- `Convergence`: increasing/saturating reward trend indicates policy stabilization.
- `Sum rate`: aggregate throughput of the network.
- `Power efficiency`: throughput per unit transmit power.
- `Fairness`: Jain's index shows how evenly rates are distributed.

High sum rate alone may come with poor fairness; reward-aware DDPG aims for a practical tradeoff.

## 📊 Automated Results Report Generation

After running `evaluation.py`, the system **automatically generates a professional PDF report** containing a complete analysis of evaluation results. The report is saved as `results/Results_Analysis_Report.pdf`.

### What the Report Contains

The generated report includes the following sections with embedded plots and computed metric values:

| Section | Description |
|---------|-------------|
| Training Convergence | Reward curve behavior, convergence, and learning stability |
| Sum Rate Comparison | DDPG vs Equal, Fractional, and Greedy baselines |
| Power Efficiency | Transmit power usage and reward-power tradeoff |
| Fairness Analysis | Jain's Fairness Index results and interpretation |
| CDF of User Rates | Distribution of per-user spectral efficiency |
| Final Observations | Summary of key findings and conclusions |

> **Note:** This report contains **results analysis only** and does not replace the full project documentation. It is designed for quick review and presentation of evaluation outcomes.

### Usage

```bash
python train_ddpg.py
python evaluation.py
```

After evaluation completes, open `results/Results_Analysis_Report.pdf` directly to view the report.

The PDF is generated using pure Python (via the `fpdf2` library) — no external tools are required.

## 🚀 Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

You can run the entire project in Google Colab without any local setup.

### Steps

1. **Open Google Colab** at [colab.research.google.com](https://colab.research.google.com/).

2. **Upload the project ZIP** using the Colab file browser (folder icon on the left sidebar).

3. **Extract and navigate to the project:**
   ```python
   !unzip project.zip
   %cd project
   ```

4. **Install dependencies:**
   ```python
   !pip install -r requirements.txt
   ```

5. **Train the DDPG model:**
   ```python
   !python train_ddpg.py
   ```

6. **Evaluate and generate the PDF report:**
   ```python
   !python evaluation.py
   ```

7. **Download the report:** The generated PDF will be available at `results/Results_Analysis_Report.pdf`. Use the Colab file browser to locate and download it.

> **Tip:** You can also use the pre-built notebook `Run_Project_On_Colab.ipynb` which includes all the above steps with inline plot display.

## Math Rendering Note
If equations do not render in your editor preview, open the README on GitHub web UI, which supports Markdown math rendering.

## Future Work
- Multi-cell extension with inter-cell interference.
- Time-correlated fading and mobility-aware dynamics.
- Comparison with TD3/SAC/PPO under identical settings.
- Constrained RL for QoS and fairness guarantees.

## License
This project is released under the MIT License.
