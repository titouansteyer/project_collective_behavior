import numpy as np
import torch
import matplotlib.pyplot as plt
from maddpg import MADDPGAgent
from env import PredatorPreyEnv  # à adapter à la signature correcte

# ============================================================
# Configuration "light"
# ============================================================

EPISODES = 300            # au lieu de 500
STEPS_PER_EPISODE = 80    # au lieu de 200
N_PREDATORS = 3
N_PREYS = 5

STATE_DIM = 40 
ACTION_DIM = 2   # [ax, ay]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Initialisation
# ============================================================

# Attention au nom des arguments dans ton env :
# dans ton env précédent c'était n_prey, n_predators.
# Je suppose ici un constructeur : PredatorPreyEnv(n_prey, n_predators)
env = PredatorPreyEnv(n_prey=N_PREYS, n_predators=N_PREDATORS)

agent = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=N_PREDATORS,
    device=DEVICE
)

rewards_history = []
DoS_values, DoA_values = [], []

total_steps = 0

# ============================================================
# Boucle d'entraînement
# ============================================================

for episode in range(EPISODES):
    obs = env.reset()  # obs doit être de dimension (N_PREDATORS, STATE_DIM)
    total_reward = 0.0
    last_metrics = {}

    for step in range(STEPS_PER_EPISODE):
        # Sélection d'action pour chaque prédateur
        actions = np.zeros((N_PREDATORS, ACTION_DIM))
        for i in range(N_PREDATORS):
            actions[i] = agent.select_action(obs[i], i)

        # Env.step : à toi de faire en sorte qu'il prenne ce format
        next_obs, rewards, done, metrics = env.step(actions)

        # Stockage dans le replay buffer
        agent.store_transition(obs, actions, rewards, next_obs)

        # Mise à jour (pas à chaque step au début)
        total_steps += 1
        if total_steps > 1000 and total_steps % 5 == 0:
            agent.update()

        total_reward += np.mean(rewards)
        obs = next_obs
        last_metrics = metrics

        if done:
            break

    rewards_history.append(total_reward)
    DoS_values.append(last_metrics.get("DoS", np.nan))
    DoA_values.append(last_metrics.get("DoA", np.nan))

    print(
        f"Episode {episode:03d} | "
        f"Total Reward: {total_reward:.2f} | "
        f"DoS={DoS_values[-1]:.3f}, DoA={DoA_values[-1]:.3f}"
    )

# ============================================================
# Sauvegarde du modèle
# ============================================================

import os
os.makedirs("models", exist_ok=True)
torch.save(agent.actor.state_dict(), "models/actor_predator_shared.pth")
print("Shared predator actor saved in /models/actor_predator_shared.pth")
# ============================================================
# Tracés
# ============================================================

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_history)
plt.title("Average Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1, 2, 2)
plt.plot(DoS_values, label="Degree of Sparsity")
plt.plot(DoA_values, label="Degree of Alignment")
plt.legend()
plt.title("Collective Behavior Metrics")
plt.xlabel("Episode")

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()
