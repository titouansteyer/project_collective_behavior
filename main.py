import numpy as np
import matplotlib.pyplot as plt
from env import PredatorPreyEnv
from metrics import degree_of_sparsity, degree_of_alignment

# --- Simulation parameters ---
N_STEPS = 1500
env = PredatorPreyEnv(n_prey=30, n_predators=3, world_size=10)

DoS_values = []
DoA_values = []

# --- Simulation loop ---
for t in range(N_STEPS):
    (prey_pos, pred_pos), (r_pred, r_prey) = env.step()

    dos = degree_of_sparsity(prey_pos)
    doa = degree_of_alignment(env.prey_velocities)
    DoS_values.append(dos)
    DoA_values.append(doa)


plt.plot(DoS_values, label="Degree of Sparsity (DoS)", color='blue')
plt.plot(DoA_values, label="Degree of Alignment (DoA)", color='orange')
plt.xlabel("Timestep")
plt.ylabel("Metric value")
plt.title("Evolution of Collective Metrics (Couzin + Deterministic Predators)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------- Simulation with Random Baseline -------------------------------------
# --- Simulation parameters ---
#N_STEPS = 200          # number of timesteps
#WORLD_SIZE = 10
#N_PREY = 30
#N_PRED = 3

#env = PredatorPreyEnv(n_prey=N_PREY, n_predators=N_PRED, world_size=WORLD_SIZE)

# --- Data storage ---
#DoS_values = []
#DoA_values = []

# --- Simulation loop ---
#for t in range(N_STEPS):
    # Random movement
#    predator_actions = np.random.uniform(-1, 1, (env.n_predators, 2))
#    prey_actions = np.random.uniform(-1, 1, (env.n_prey, 2))

#    (prey_pos, pred_pos), (r_pred, r_prey) = env.step(predator_actions, prey_actions)

    #prey + predator positions & velocities
#    all_pos = np.vstack((prey_pos, pred_pos))
#    all_vel = np.vstack((prey_actions, predator_actions))
#
    #metrics
#    dos = degree_of_sparsity(all_pos)
#    doa = degree_of_alignment(all_vel)

#    DoS_values.append(dos)
#    DoA_values.append(doa)
#    print(f"Step {t:03d}: DoS={dos:.3f}, DoA={doa:.3f}, Predator reward={r_pred.sum():.2f}")


# -------------------------------------Exemple fictif pour premier test des métriques -------------------------------
#positions = np.random.rand(10, 2) * 10
#velocities = np.random.randn(10, 2)

#dos = degree_of_sparsity(positions)
#doa = degree_of_alignment(velocities)

#print(f"DoS = {dos:.3f}, DoA = {doa:.3f}")


# -------------------------------------Exemple fictif pour test de l'environnement -----------------------------------
#env = PredatorPreyEnv()
#for t in range(10):
#    predator_actions = np.random.uniform(-1, 1, (env.n_predators, 2))
#    prey_actions = np.random.uniform(-1, 1, (env.n_prey, 2))
#    (prey, predators), (r_pred, r_prey) = env.step(predator_actions, prey_actions)
#    print(f"Step {t}: predator reward {r_pred.sum():.2f}, prey reward {r_prey.sum():.2f}")
