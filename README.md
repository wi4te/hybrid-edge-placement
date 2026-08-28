<div align="center">

# Hybrid Service Placement in Edge Computing

**Final-year ML project — adaptive microservice placement using GA, PSO & Multi-Agent RL with GAT**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=flat-square)](LICENSE)
[![Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-6366F1?style=flat-square&logo=github)](https://wi4te.github.io/hybrid-edge-placement/)
[![IIEST](https://img.shields.io/badge/IIEST%20Shibpur-Student%20Project-F59E0B?style=flat-square)](https://www.iiests.ac.in/)

### 🌐 [**Interactive Live Demo →**](https://wi4te.github.io/hybrid-edge-placement/)
*Simulate the full system in your browser — no install needed*

</div>

---

## About This Project

I built this as a personal ML research project while studying at IIEST Shibpur. The problem I wanted to solve: **when a mobile user's app (made of multiple microservices) needs to be deployed on nearby edge servers, which server should get which service?**

This turns out to be a hard combinatorial optimization problem — you want low latency, good load balance, low energy, and privacy all at once. I implemented and compared three different algorithms to solve it, then combined them into a hybrid system that picks the best one per situation.

The project is based on two research papers (full citations at the bottom):
- Kazmi et al. (2025) — *Dynamic Service Placement in Edge Computing*
- Geng et al. (KDD'23) — *GAL-VNE: Graph Attention Learning for VNE*

> **Note:** This is a simulation study, not a real deployment. All results are from a Python simulation of a 4×5 grid of edge servers.

---

## What It Does

The system places microservices from different application types onto a grid of 20 edge servers, optimizing for 4 objectives simultaneously:

| Objective | Description | Algorithm specialized for this |
|-----------|-------------|-------------------------------|
| **Latency** | Sum of communication delays between consecutive services | PSO |
| **Privacy** | Fewer unique servers used = less data spread | GA |
| **Load Balance** | Standard deviation of server utilization | MARL + GAT |
| **Energy** | Total Watts consumed across the network | EnergyAwareScheduler |

---

## Algorithms Implemented

### 1. Genetic Algorithm (GA) — Privacy-optimized
Encodes a placement as a chromosome (list of server IDs, one per service). Evolves a population of 60 candidates over 40 generations using tournament selection, single-point crossover, and mutation. Biased toward solutions that consolidate services onto fewer servers.

### 2. Particle Swarm Optimization (PSO) — Latency-optimized
40 particles explore a continuous position space (discretized to server IDs). Each particle's velocity is updated by its personal best and the global best position. Converges fast and produces low-latency placements.

### 3. Multi-Agent RL with Graph Attention Network (MARL + GAT)
This is the most complex part. There's one DQN agent per region (4 regions for the 4×5 grid). Each agent uses a Graph Attention Network (Velickovic et al., 2018) to encode server topology — neighboring servers share information through learned attention weights.

Training uses **curriculum learning**: start with 1-service apps, gradually increase to 8 services. First pretrain on a static environment, then finetune on a dynamic one with server failures and flash crowds. Finally enable online learning during deployment.

### 4. Hybrid Engine
Routes each incoming request to the best algorithm based on:
- App type (LTS → GA, MMA → PSO, others → MARL)
- Current network load (if avg util > 75%, always use MARL)
- Deadline (tight deadline → PSO)
- Fallback chain: if primary fails, tries next (GA→PSO→MARL)

---

## Additional Features (beyond the papers)

These are things I added on top of the paper implementations:

- **`EnergyAwareScheduler`** — wraps any solver, tries to find a greener placement that doesn't sacrifice much fitness
- **`SLAMonitor`** — tracks latency SLA violations per application type, prints a violation report
- **`AdaptiveWeightTuner`** — if SLA violations increase, automatically shifts objective weights toward latency using EMA
- **`FailoverManager`** — when a server fails, migrates its services to the next-best available server
- **`BenchmarkRunner`** — adds random and first-fit greedy baselines so the comparison is actually fair

---

## Live Demo

> **[https://wi4te.github.io/hybrid-edge-placement/](https://wi4te.github.io/hybrid-edge-placement/)**

The browser demo re-implements the core placement logic in JavaScript on an HTML5 Canvas. You can:

- Switch between Hybrid / GA / PSO / MARL modes
- Click any server to toggle its failure state
- Trigger flash crowd events (5 heavy requests at once)
- Adjust number of services and CPU requirement per request
- Watch live metrics update: acceptance rate, avg latency, energy, load balance

> The JS demo is a simplified version for visualization. The real GA/PSO/MARL runs in Python.

---

## Results

Running `python hybridplacement.py` on 100 requests (4×5 grid, seed=42):

| Algorithm | Success Rate | Avg Latency | Avg Servers Used | Avg Time/req |
|-----------|:---:|:---:|:---:|:---:|
| Random | ~55% | very high | high | < 1ms |
| First-Fit Greedy | ~70% | high | high | < 1ms |
| GA | ~82% | moderate | **lowest** | ~800ms |
| PSO | ~84% | **lowest** | moderate | ~400ms |
| MARL+GAT | ~85% | moderate | moderate | ~10ms |
| **HYBRID** | **~88%** | balanced | balanced | varies |

*Results vary with random seed and network conditions.*

---

## How to Run

```bash
git clone https://github.com/wi4te/hybrid-edge-placement.git
cd hybrid-edge-placement

pip install -r requirements.txt

python hybridplacement.py
```

**Output files generated:**
- `dashboard.png` — 6-panel analytics dashboard
- `research_plots.png` — sensitivity analysis (acceptance ratio vs arrival rate, network size, etc.)
- `routing_decisions.csv` — every placement decision logged

**If you don't have PyTorch** (MARL will use greedy fallback):
```bash
pip install numpy matplotlib
python hybridplacement.py
```

---

## Project Structure

```
hybrid-edge-placement/
├── hybridplacement.py         # Main simulation (~2700 lines)
│   ├── Sections 1–5          # Data models, network, GA, PSO
│   ├── Sections 6–8          # GAT layers, replay buffer, MARL agent
│   ├── Sections 9–13         # Workload, logger, dashboard, evaluator
│   ├── Sections 14–16        # EnergyScheduler, SLAMonitor, WeightTuner
│   └── Sections 17–18        # FailoverManager, BenchmarkRunner + baselines
├── docs/
│   └── index.html             # GitHub Pages interactive demo (pure JS)
├── .github/
│   └── workflows/
│       └── pages.yml          # Auto-deploy docs/ on push to main
├── requirements.txt
└── README.md
```

---

## References

This project is based on, and cites, the following papers. I implemented the core ideas from scratch in Python — no official code was used.

**[1] Kazmi, S. H. A., et al. (2025)**
*"Dynamic Service Placement in Edge Computing"*
— Introduced the 4-objective placement framework (latency, distance, privacy, load balance) and the MARL formulation used in Sections 8 and 12.

**[2] Geng, X., et al. (2023)**
*"GAL-VNE: Solving the VNE Problem with Global Reinforcement Learning and Local Graph Attention"*
KDD '23: Proceedings of the 29th ACM SIGKDD Conference
— Inspired the Graph Attention Network architecture used in the Q-network (Section 6) and the GNN-based server scoring approach.

**[3] Velickovic, P., et al. (2018)**
*"Graph Attention Networks"*
International Conference on Learning Representations (ICLR)
— The original GAT paper. The `GATLayer` class (Section 6) is a direct implementation of this architecture.

**[4] Kennedy, J. & Eberhart, R. (1995)**
*"Particle swarm optimization"*
Proceedings of ICNN'95 — the original PSO paper referenced for the PSO implementation.

---

## About Me

Student at IIEST Shibpur (2024 batch, ITB). Interested in ML systems, edge computing, and reinforcement learning. This project was done independently as a way to deeply understand optimization algorithms and graph neural networks by implementing them from scratch rather than using existing libraries.

Feel free to open an issue or reach out if you have questions about any part of the code.

---

<div align="center">

*If this was useful to you, consider leaving a ⭐*

</div>
