import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from maddpg import MADDPGAgent
from env import PredatorPreyEnv

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

EPISODES = 1000
STEPS_PER_EPISODE = 80
N_PREDATORS = 3
N_PREYS = 20     # cohérent avec env par défaut

STATE_DIM = 40   # doit matcher _get_predator_obs()
ACTION_DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Init
# ------------------------------------------------------------

env = PredatorPreyEnv(n_prey=N_PREYS, n_predators=N_PREDATORS, world_size=7.0)

agent = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=N_PREDATORS,
    device=DEVICE
)

rewards_history = []
DoS_values, DoA_values = [], []
total_steps = 0

# ------------------------------------------------------------
# Entraînement
# ------------------------------------------------------------

for episode in range(EPISODES):
    obs = env.reset()   # shape (N_PREDATORS, STATE_DIM)
    total_reward = 0.0
    last_metrics = {}

    for step in range(STEPS_PER_EPISODE):
        actions = np.zeros((N_PREDATORS, ACTION_DIM))
        for i in range(N_PREDATORS):
            actions[i] = agent.select_action(obs[i], i, noise_scale=0.1)

        next_obs, rewards, done, metrics = env.step(actions)

        agent.store_transition(obs, actions, rewards, next_obs)

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

# ------------------------------------------------------------
# Sauvegarde modèle
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)
torch.save(agent.actor.state_dict(), "models/actor_predator_shared.pth")
print("Saved shared predator actor in models/actor_predator_shared.pth")

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

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
