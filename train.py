import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from maddpg import MADDPGAgent

# Choisis l'env :
#from env import PredatorPreyEnv  # bords infinis (tore)
#EXP_NAME = "walls"
from env_border_strong import PredatorPreyEnvReflect as PredatorPreyEnv  # bords solides
EXP_NAME = "walls"

# ------------------------------------------------------------
# Hyperparameters

EPISODES = 500
STEPS_PER_EPISODE = 100

N_PREDATORS = 3
N_PREYS = 10  # comme Li pendant training (ensuite tu peux tester avec 50 en eval)

STATE_DIM = 40
ACTION_DIM = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exploration
NOISE_PRED = 0.15
NOISE_PREY = 0.15

# fréquence d'update
WARMUP_STEPS = 2000
UPDATE_EVERY = 5

# ------------------------------------------------------------
# Initialization

env = PredatorPreyEnv(
    n_prey=N_PREYS,
    n_predators=N_PREDATORS,
    world_size=7.0
)

agent_pred = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=N_PREDATORS,
    device=DEVICE
)

agent_prey = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=N_PREYS,
    device=DEVICE
)

rewards_pred_history = []
rewards_prey_history = []
DoS_values, DoA_values = [], []
total_steps = 0

# ------------------------------------------------------------
# Helper: safe unpack reset output

def reset_env():
    out = env.reset()
    # On accepte:
    # - (pred_obs, prey_obs)
    # - pred_obs seul (ancien env), auquel cas on récupère prey_obs via méthode privée si dispo
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    else:
        pred_obs = out
        if hasattr(env, "_get_prey_obs"):
            prey_obs = env._get_prey_obs()
        else:
            raise RuntimeError("Env reset() ne renvoie pas prey_obs et env._get_prey_obs() absent.")
        return pred_obs, prey_obs


# ------------------------------------------------------------
# Training loop

for episode in range(EPISODES):
    pred_obs, prey_obs = reset_env()

    ep_pred_reward = 0.0
    ep_prey_reward = 0.0
    last_metrics = {}

    for step in range(STEPS_PER_EPISODE):
        # --- Actions predators ---
        pred_actions = np.zeros((N_PREDATORS, ACTION_DIM))
        for i in range(N_PREDATORS):
            pred_actions[i] = agent_pred.select_action(pred_obs[i], noise_scale=NOISE_PRED)

        # --- Actions prey ---
        prey_actions = np.zeros((N_PREYS, ACTION_DIM))
        for j in range(N_PREYS):
            prey_actions[j] = agent_prey.select_action(prey_obs[j], noise_scale=NOISE_PREY)

        # --- Env step (coevolution) ---
        (pred_next_obs, prey_next_obs), (pred_rewards, prey_rewards), done, metrics = env.step(
            pred_actions, prey_actions
        )

        # --- Store transitions (1 transition per agent) ---
        agent_pred.store_transition(pred_obs, pred_actions, pred_rewards, pred_next_obs)
        agent_prey.store_transition(prey_obs, prey_actions, prey_rewards, prey_next_obs)

        pred_obs = pred_next_obs
        prey_obs = prey_next_obs

        total_steps += 1
        last_metrics = metrics

        ep_pred_reward += float(np.mean(pred_rewards))
        ep_prey_reward += float(np.mean(prey_rewards))

        # --- Update networks ---
        if total_steps > WARMUP_STEPS and total_steps % UPDATE_EVERY == 0:
            agent_pred.update()
            agent_prey.update()

        if done:
            break

    rewards_pred_history.append(ep_pred_reward)
    rewards_prey_history.append(ep_prey_reward)
    DoS_values.append(last_metrics.get("DoS", np.nan))
    DoA_values.append(last_metrics.get("DoA", np.nan))

    print(
        f"Episode {episode:04d} | "
        f"R_pred={ep_pred_reward:.3f} | R_prey={ep_prey_reward:.3f} | "
        f"DoS={DoS_values[-1]:.3f}, DoA={DoA_values[-1]:.3f}"
    )

# ------------------------------------------------------------
# Save models

os.makedirs("models", exist_ok=True)
torch.save(
    agent_pred.actor.state_dict(),
    f"models/actor_pred_{EXP_NAME}.pth"
)

torch.save(
    agent_prey.actor.state_dict(),
    f"models/actor_prey_{EXP_NAME}.pth"
)
print("Saved:")
print(f" - models/actor_pred_{EXP_NAME}.pth")
print(f" - models/actor_prey_{EXP_NAME}.pth")

# ------------------------------------------------------------
# Plotting results


def rolling_mean(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def rolling_std(x, w=50):
    x = np.asarray(x, dtype=float)
    out = []
    for i in range(w-1, len(x)):
        out.append(np.std(x[i-w+1:i+1]))
    return np.array(out)

W = 50  # fenêtre de lissage (essaie 50 ou 100)

# --- Rewards predators ---
r = np.array(rewards_pred_history)  # ou rewards_history si predators seuls
rm = rolling_mean(r, W)
rs = rolling_std(r, W)
x = np.arange(len(rm))

plt.figure(figsize=(10,4))
plt.plot(x, rm)
plt.fill_between(x, rm-rs, rm+rs, alpha=0.2)
plt.title("Predators reward (rolling mean ± std)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig("predator_reward.png")
plt.show()

# --- Rewards prey (si coevolution) ---
if "rewards_prey_history" in globals():
    r = np.array(rewards_prey_history)
    rm = rolling_mean(r, W)
    rs = rolling_std(r, W)
    x = np.arange(len(rm))

    plt.figure(figsize=(10,4))
    plt.plot(x, rm)
    plt.fill_between(x, rm-rs, rm+rs, alpha=0.2)
    plt.title("Prey reward (rolling mean ± std)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("prey_reward.png")
    plt.show()

# --- DoS / DoA ---
dos = np.array(DoS_values)
doa = np.array(DoA_values)

dos_m = rolling_mean(dos, W)
doa_m = rolling_mean(doa, W)
x = np.arange(len(dos_m))

plt.figure(figsize=(10,4))
plt.plot(x, dos_m, label="DoS (smoothed)")
plt.plot(x, doa_m, label="DoA (smoothed)")
plt.ylim(0, 1)
plt.title("Collective metrics (smoothed)")
plt.xlabel("Episode")
plt.legend()
plt.tight_layout()
plt.savefig("collective_metrics.png")
plt.show()