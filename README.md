# project_collective_behavior

This project is part of the **Collective Behaviour course at the Faculty of Computer Science.**.  
Our goal is to study and expand upon existing models of **collective animal behaviour**.

## Project Overview
This repository implements a predator–prey multi-agent system inspired by Li et al. (2023), with the objective of studying how collective behaviours emerge under survival pressure.

Our implementation focuses on three core aspects:

- **Environment**  
  A 2D continuous predator–prey environment, available in two configurations:
  - a **toroidal world** (periodic boundary conditions),
  - a **bounded world with walls**, including different wall interaction regimes.

- **Learning framework**  
  Agents are trained using **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**.
  Predators are always controlled via reinforcement learning, while prey can either:
  - follow **Couzin-style deterministic interaction rules**, or
  - be trained via **reinforcement learning** in a co-evolution setting.

- **Analysis and visualisation**  
  Emergent behaviours are quantitatively analysed using collective metrics such as:
  - Degree of Sparsity (DoS),
  - Degree of Alignment (DoA),
  and qualitatively explored through animated visualisations of the simulations.

This framework allows systematic comparisons between:
- toroidal and bounded environments,
- rule-based prey (Couzin) and learning-based prey (RL),
- different boundary interaction models.

The project builds upon the following reference work:  
*Predator–prey survival pressure is sufficient to evolve swarming behaviors* [New Journal of Physics, 25 (2023) 093024](https://iopscience.iop.org/article/10.1088/1367-2630/acf33a)

## Summary of the paper
Li et al. (2023) investigate how complex collective behaviours such as flocking, milling, and swirling can emerge purely from survival-driven interactions, without explicitly programmed social rules.

The authors propose a minimal predator–prey co-evolution framework based on multi-agent reinforcement learning.  
Predators receive a positive reward when catching prey, while prey receive a negative reward when caught. No explicit incentives for cohesion, alignment, or dispersion are provided.

Despite this simplicity, the system gives rise to rich emergent dynamics. Prey develop cohesive group structures that reduce predation risk, while predators exhibit coordinated hunting strategies, including dispersion and edge-focused attacks.

These behaviours are quantified using collective metrics such as the Degree of Sparsity (DoS) and the Degree of Alignment (DoA), demonstrating that survival pressure alone can be sufficient to generate coordinated group motion.  
This work provides a compelling perspective on the emergence of collective animal behaviour and serves as a foundation for both biological modelling and swarm robotics research.



## Team Members
-  Rafaëlle Lacraz [rafaellelac](https://github.com/rafaellelac)
-  Titouan Steyer [titouansteyer](https://github.com/titouansteyer)

## Repository Structure

```text
project_collective_behavior/
│
├── env.py                   # Toroidal predator–prey environment
├── env_border_strong.py     # Bounded environment with walls (strong / pure reflection)
│
├── maddpg.py                # MADDPG implementation
├── metrics.py               # DoS and DoA collective behaviour metrics
│
├── train.py                 # Training script (configurable: torus/walls, couzin/RL)
├── visualize.py             # Simulation and GIF generation
│
├── models/                  # Saved trained models (actors)
│   ├── actor_pred_*.pth
│   └── actor_prey_*.pth
│
├── simulation/              # Generated GIF simulations
│   ├── simulation_torus_*.gif
│   └── simulation_walls_*.gif
│
├── README.md
```

##  Project Plan
### **Milestone 1 — First Report**
- Read and summarize **Li et al. (2023)** and related work on collective behavior.  
- Reproduce the **baseline predator–prey MARL model** using  
  [`xxnnnnn/PredatorPrey_RL_Reproduction`](https://github.com/xxnnnnn/PredatorPrey_RL_Reproduction).  
- Verify that environment, rewards, and parameters match the paper.  
- if we have time : Run first simulations and measure **DoS** and **DoA** to confirm swarming emergence.  
- Document key differences from the paper and write **Report 1** (IMRAD, 4 pages).  

### **Milestone 2 — Second Report**
- Improve **Report 1** based on feedback.  
- Stabilise the environment so that it matches the behaviour described in Li et al. (2023) and produces consistent baseline results.
- Improve the current reinforcement learning training.
- Analyse emergent behaviours using the DoS and DoA metrics to compare random motion, deterministic rules, and early RL policies.
- Investigate the effect of simple parameters, such as the number of predators/prey or the observation radius.  

### **Milestone 3 — Final Report**
- Produce the final polished report (≤ 4 pages / 2000 words).
- Compare:
  - Torus vs bounded environments (walls).
  - Prey controlled by Couzin rules vs RL.
- Discuss limitations and future work (obstacles, scalability, heterogeneous speeds).
- Prepare presentation slides (≤ 20 minutes).
- Finalise GitHub repository (clean structure, reproducibility).
  
##  Deadlines / Milestones
- **Report 1**: 2025-11-16  
- **Report 2**: 2025-12-07
- **Final Report**: 2026-01-11


## Installation

### Requirements
- Python 3.10+  
- Recommended: virtual environment (`venv` or `conda`)

### Setup

```bash
git clone https://github.com/titouansteyer/project_collective_behavior.git
cd project_collective_behavior
```

## How to Run : Visualisation 
## Visualisation Configuration

The script `visualize.py` allows you to generate animated simulations (GIFs) of trained models.

### Environment choice
At the top of `visualize.py`, you can select the environment:

```python
MODE = "torus"   # "torus" or "walls"
PREY_MODE = "couzin"  # "couzin" or "rl"
```

```bash
python3 visualize.py
```
