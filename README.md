# project_collective_behavior

## Project Description
This project is part of the **Collective Behaviour course at the Faculty of Computer Science.**.  
Our goal is to study and expand upon existing models of **collective animal behaviour**.

We use as a starting point the paper:  
*Predator–prey survival pressure is sufficient to evolve swarming behaviors*  
[New Journal of Physics, 25 (2023) 093024](https://iopscience.iop.org/article/10.1088/1367-2630/acf33a)

## Summary
This paper explores how complex collective behaviors such as flocking and swirling can emerge purely from survival dynamics, without any predefined social rules.
Li et al. (2023) propose a minimal predator–prey co-evolution framework using multi-agent reinforcement learning (MARL).
Agents (predators and prey) receive only a simple survival-based reward: predators gain +1 when catching prey, and prey receive −1 when caught.
Despite this minimal setup, the system naturally evolves rich emergent behaviors — prey develop cohesive flocking and swirling, while predators exhibit dispersion tactics, confusion effects, and edge predation.
The authors measure these phenomena using quantitative metrics such as the Degree of Sparsity (DoS) and Degree of Alignment (DoA), showing that survival pressure alone is sufficient to produce coordinated group motion.
This framework provides valuable insights into the evolution of collective animal behavior and offers a foundation for swarm robotics research.

## Team Members
-  Rafaëlle Lacraz [rafaellelac](https://github.com/rafaellelac)
-  Titouan Steyer [titouansteyer](https://github.com/titouansteyer)

##  Project Plan
### **Milestone 1 — First Report**
In the first phase, we will:  
- Read and summarize **Li et al. (2023)** and related work on collective behavior.  
- Reproduce the **baseline predator–prey MARL model** using  
  [`xxnnnnn/PredatorPrey_RL_Reproduction`](https://github.com/xxnnnnn/PredatorPrey_RL_Reproduction).  
- Verify that environment, rewards, and parameters match the paper.  
- if we have time : Run first simulations and measure **DoS** and **DoA** to confirm swarming emergence.  
- Document key differences from the paper and write **Report 1** (IMRAD, 4 pages).  

### **Milestone 2 — Second Report**
- Improve **Report 1** based on feedback.  
- Add details about the **MADDPG algorithm**, training, and verification.  
- Conduct **parameter sweeps** (predator number, perception range, speed ratio).  
- Report intermediate results and update the **README.md**.  

### **Milestone 3 — Final Report**
- Produce the **final polished report** (≤ 4 pages / 2000 words).  
- Compare baseline and extended models (quantitative DoS/DoA results).  
- Discuss limitations and propose future work.  
- Prepare and attach **presentation slides** (≤ 20 min).  
- Finalize GitHub: cleaned structure, clear README, and run instructions.  
  
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

## How to Run
### 1. Train the agents
To launch a training run:

```bash
python3 train.py
```

### 2. Visualisation
```bash
python3 visualize.py
