# project_collective_behavior

## Project Description
This project is part of the **Collective Behaviour course at the Faculty of Computer Science.**.  
Our goal is to study and expand upon existing models of **collective animal behaviour**.

## Project Overview
This repository implements a predator–prey multi-agent system inspired by Li et al. (2023), with the goal of analysing how collective behaviours emerge under survival pressure.

The project is organised around three main components:

- **Environment**  
  A 2D continuous predator–prey environment, implemented both as:
  - a **toroidal world** (infinite periodic space),
  - a **bounded world with walls**, including physical wall interactions.

- **Learning framework**  
  Predators (and optionally prey) are trained using **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**.
  The reward structure is minimal and purely survival-based.

- **Analysis and visualisation**  
  Emergent collective behaviours are analysed using:
  - Degree of Sparsity (DoS),
  - Degree of Alignment (DoA),
  and visualised through animated simulations.

The framework allows us to compare:
- torus vs bounded environments,
- prey controlled by **Couzin-style rules** vs **reinforcement learning**,
- and different wall interaction regimes.

We use as a starting point the paper:  
*Predator–prey survival pressure is sufficient to evolve swarming behaviors*  
[New Journal of Physics, 25 (2023) 093024](https://iopscience.iop.org/article/10.1088/1367-2630/acf33a)

## Summary of the paper
This paper explores how complex collective behaviors such as flocking and swirling can emerge purely from survival dynamics, without any predefined social rules.
Li et al. (2023) propose a minimal predator–prey co-evolution framework using multi-agent reinforcement learning (MARL).
Agents (predators and prey) receive only a simple survival-based reward: predators gain +1 when catching prey, and prey receive −1 when caught.
Despite this minimal setup, the system naturally evolves rich emergent behaviors — prey develop cohesive flocking and swirling, while predators exhibit dispersion tactics, confusion effects, and edge predation.
The authors measure these phenomena using quantitative metrics such as the Degree of Sparsity (DoS) and Degree of Alignment (DoA), showing that survival pressure alone is sufficient to produce coordinated group motion.
This framework provides valuable insights into the evolution of collective animal behavior and offers a foundation for swarm robotics research.

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
```markdown
## Visualisation Configuration

The script `visualize.py` allows you to generate animated simulations (GIFs) of trained models.

### Environment choice
At the top of `visualize.py`, you can select the environment:

```python
MODE = "torus"   # "torus" or "walls"
PREY_MODE = "couzin"  # "couzin" or "rl"

```bash
python3 visualize.py
