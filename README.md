# project_collective_behavior

## Project Description
This project is part of the **Collective Behaviour course at the Faculty of Computer Science.**.  
Our goal is to study and expand upon existing models of **collective animal behaviour**.

We use as a starting point the paper:  
*Predator–prey survival pressure is sufficient to evolve swarming behaviors*  
New Journal of Physics, 25 (2023) 093024  (https://iopscience.iop.org/article/10.1088/1367-2630/acf33a)

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
- Read and summarize the chosen paper and related references.  
- Review concepts and existing models of swarming in predator–prey interactions.  
- Recreate the **baseline swarming model** described in the paper.  
- Run **initial simulations** to verify baseline behaviours (emergence of swarming under predator pressure).  
- Document our intended approach for extending the model later in the course.  
- Write Report 1 (IMRAD structure, 4 pages max) and push both **source + PDF** into the GitHub repository.  


### **Milestone 2 — Second Report**
- Polish the first report based on received comments.  
- Add **details about methods and proposed methodology for verification**.  
- Follow **IMRAD structure**.  
- Page limit: **4 pages**.  
- Explicitly specify **individual contributions** of each team member (who wrote/implemented what).  
- **Deliverables**: Source + PDF of updated report in repository.
- Update README.md 

### **Milestone 3 — Final Report**
- Polished version of previous reports with improvements.  
- Concise **introduction with relevant research** and problem statement.  
- Methods: clear, reproducible description of our model & evaluation.  
- Results: include **comparisons with previous research**.  
- Discussion: interpret results, link with literature, propose future work.  
- Follow **IMRAD structure**.  
- Page limit: **4 pages / 2000 words max**.  
- Must include **individual contributions** again.  
- Attach **slides for presentation** (≤20 min) covering:  
  a) overview of the problem & motivation,  
  b) initial goals & achieved results,  
  c) main challenges & solutions,  
  d) what we would do differently.  
- Finalize GitHub repository with:  
  - A clear **README.md** (one-paragraph project description, main paper, goals, level achieved).  
  - Detailed **instructions to run simulations**.  
- **Deliverables**: Final report (source + PDF) + slides in repository.
- Update README.md 
  
##  Deadlines / Milestones
- **Report 1**: 2025-11-16  
- **Report 2**: 2025-12-07
- **Final Report**: 2026-01-11
