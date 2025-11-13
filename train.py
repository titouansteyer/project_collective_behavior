import numpy as np
import torch
import matplotlib.pyplot as plt
from maddpg import MADDPGAgent
from env import PredatorPreyEnv

# ============================================================
# 🔹 Configuration
# ============================================================

EPISODES = 500
STEPS_PER_EPISODE = 200
N_PREDATORS = 3
N_PREYS = 5

STATE_DIM = 4    # [x, y, vx, vy]
ACTION_DIM = 2   # [ax, ay]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 🔹 Initialisation
# ============================================================

env = PredatorPreyEnv(n_predators=N_PREDATORS, n_preys=N_PREYS)
agent = MADDPGAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, n_agents=N_PREDATORS, device=DEVICE)

rewards_history = []
DoS_values, DoA_values = [], []

# ============================================================
# 🔹 Boucle d'entraînement
# ============================================================

for episode in range(EPISODES):
    obs = env.reset()
    total_reward = 0

    for step in range(STEPS_PER_EPISODE):
        actions = [agent.select_action(obs[i], i) for i in range(N_PREDATORS)]
        next_obs, rewards, done, metrics = env.step(actions)

        agent.store_transition(obs, actions, rewards, next_obs)
        agent.update()

        total_reward += np.mean(rewards)
        obs = next_obs

        if done:
            break

    rewards_history.append(total_reward)
    DoS_values.append(metrics.get("DoS", np.nan))
    DoA_values.append(metrics.get("DoA", np.nan))

    print(f"Episode {episode:03d} | Total Reward: {total_reward:.2f} | DoS={DoS_values[-1]:.3f}, DoA={DoA_values[-1]:.3f}")

# ============================================================
# 🔹 Sauvegarde du modèle
# ============================================================

for i, actor in enumerate(agent.actors):
    torch.save(actor.state_dict(), f"models/actor_predator_{i}.pth")
print("✅ Models saved in /models")

# ============================================================
# 🔹 Tracés
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
