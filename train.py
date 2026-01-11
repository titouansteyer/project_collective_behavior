"""
train.py

Training script for predator–prey collective behavior using a MADDPG-style algorithm.

This script:
- Trains predator agents (and optionally prey agents) in a 2D environment
- Supports two environments:
    * Toroidal world (periodic boundaries)
    * Bounded world with strong reflective walls
- Supports two prey modes:
    * "couzin": rule-based prey behavior
    * "rl": learned prey behavior (coevolution)
- Logs episode rewards and collective behavior metrics (DoS, DoA)
- Saves trained actor models and training curves

This file is intended for long training runs and offline analysis.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from maddpg import MADDPGAgent

# ============================================================
# Environment selection
# Uncomment ONE of the following blocks

# from env import PredatorPreyEnv  # infinite borders (torus)
# EXP_NAME = "torus"

from env_border_strong import PredatorPreyEnvReflect as PredatorPreyEnv  # solid walls
EXP_NAME = "walls"

# ============================================================

# ------------------------------------------------------------
# Hyperparameters

EPISODES = 500
STEPS_PER_EPISODE = 100

N_PREDATORS = 3
N_PREYS = 10  # same as Li et al. during training

STATE_DIM = 40
ACTION_DIM = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exploration noise
NOISE_PRED = 0.15
NOISE_PREY = 0.15

# Update schedule
WARMUP_STEPS = 2000
UPDATE_EVERY = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "train_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Initialization

env = PredatorPreyEnv(
    n_prey=N_PREYS,
    n_predators=N_PREDATORS,
    world_size=2.0,
    prey_mode="rl"  # "couzin" or "rl"
)

agent_pred = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=N_PREDATORS,
    device=DEVICE
)

# Create and train prey agent only if prey is controlled by RL
agent_prey = None
if getattr(env, "prey_mode", "rl") == "rl":
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
# Helper: robust environment reset

def reset_env():
    """
    Reset the environment with compatibility for different reset signatures.

    Supports:
    - reset() -> (pred_obs, prey_obs)
    - reset() -> pred_obs (legacy), in which case prey_obs is retrieved separately
    """
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    else:
        pred_obs = out
        if hasattr(env, "_get_prey_obs"):
            prey_obs = env._get_prey_obs()
        else:
            raise RuntimeError(
                "Env reset() does not return prey_obs and env._get_prey_obs() is missing."
            )
        return pred_obs, prey_obs

# ------------------------------------------------------------
# Training loop

for episode in range(EPISODES):
    pred_obs, prey_obs = reset_env()

    ep_pred_reward = 0.0
    ep_prey_reward = 0.0
    last_metrics = {}

    for step in range(STEPS_PER_EPISODE):
        # --- Predator actions ---
        pred_actions = np.zeros((N_PREDATORS, ACTION_DIM))
        for i in range(N_PREDATORS):
            pred_actions[i] = agent_pred.select_action(
                pred_obs[i], noise_scale=NOISE_PRED
            )

        if env.prey_mode == "rl":
            # --- Prey actions (RL) ---
            prey_actions = np.zeros((N_PREYS, ACTION_DIM))
            for j in range(N_PREYS):
                prey_actions[j] = agent_prey.select_action(
                    prey_obs[j], noise_scale=NOISE_PREY
                )

            # --- Environment step (coevolution) ---
            (pred_next_obs, prey_next_obs), (pred_rewards, prey_rewards), done, metrics = env.step(
                pred_actions, prey_actions
            )

            # --- Store transitions ---
            agent_pred.store_transition(
                pred_obs, pred_actions, pred_rewards, pred_next_obs
            )
            agent_prey.store_transition(
                prey_obs, prey_actions, prey_rewards, prey_next_obs
            )

            pred_obs = pred_next_obs
            prey_obs = prey_next_obs
            ep_prey_reward += float(np.mean(prey_rewards))

        else:
            # --- Environment step (Couzin prey) ---
            pred_next_obs, pred_rewards, done, metrics = env.step(pred_actions)

            # --- Store predator transitions only ---
            agent_pred.store_transition(
                pred_obs, pred_actions, pred_rewards, pred_next_obs
            )

            pred_obs = pred_next_obs
            if hasattr(env, "_get_prey_obs"):
                prey_obs = env._get_prey_obs()

        total_steps += 1
        last_metrics = metrics
        ep_pred_reward += float(np.mean(pred_rewards))

        # --- Network updates ---
        if total_steps > WARMUP_STEPS and total_steps % UPDATE_EVERY == 0:
            agent_pred.update()
            if env.prey_mode == "rl":
                agent_prey.update()

        if done:
            break

    rewards_pred_history.append(ep_pred_reward)
    rewards_prey_history.append(ep_prey_reward if env.prey_mode == "rl" else np.nan)
    DoS_values.append(last_metrics.get("DoS", np.nan))
    DoA_values.append(last_metrics.get("DoA", np.nan))

    if env.prey_mode == "rl":
        print(
            f"Episode {episode:04d} | "
            f"R_pred={ep_pred_reward:.3f} | R_prey={ep_prey_reward:.3f} | "
            f"DoS={DoS_values[-1]:.3f}, DoA={DoA_values[-1]:.3f}"
        )
    else:
        print(
            f"Episode {episode:04d} | "
            f"R_pred={ep_pred_reward:.3f} | "
            f"DoS={DoS_values[-1]:.3f}, DoA={DoA_values[-1]:.3f}"
        )

# ------------------------------------------------------------
# Save trained models

os.makedirs("models", exist_ok=True)

torch.save(
    agent_pred.actor.state_dict(),
    f"models/actor_pred_{EXP_NAME}_{env.prey_mode}.pth"
)

if env.prey_mode == "rl":
    torch.save(
        agent_prey.actor.state_dict(),
        f"models/actor_prey_{EXP_NAME}_{env.prey_mode}.pth"
    )

print("Saved:")
print(f" - models/actor_pred_{EXP_NAME}_{env.prey_mode}.pth")
if env.prey_mode == "rl":
    print(f" - models/actor_prey_{EXP_NAME}_{env.prey_mode}.pth")

# ------------------------------------------------------------
# Plotting utilities

def rolling_mean(x, w=50):
    """Compute rolling mean over a window of size w."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")

def rolling_std(x, w=50):
    """Compute rolling standard deviation over a window of size w."""
    x = np.asarray(x, dtype=float)
    out = []
    for i in range(w - 1, len(x)):
        out.append(np.std(x[i - w + 1:i + 1]))
    return np.array(out)

W = 50  # smoothing window

# ------------------------------------------------------------
# Plot predator rewards

r = np.array(rewards_pred_history, dtype=float)
rm = rolling_mean(r, W)
rs = rolling_std(r, W)
x = np.arange(len(rm))

plt.figure(figsize=(10, 4))
plt.plot(x, rm)
plt.fill_between(x, rm - rs, rm + rs, alpha=0.2)
plt.title("Predators reward (rolling mean ± std)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        f"predator_reward_{EXP_NAME}_{env.prey_mode}.png"
    )
)
plt.show()

# ------------------------------------------------------------
# Plot prey rewards (coevolution only)

if env.prey_mode == "rl":
    r = np.array(rewards_prey_history, dtype=float)
    rm = rolling_mean(r, W)
    rs = rolling_std(r, W)
    x = np.arange(len(rm))

    plt.figure(figsize=(10, 4))
    plt.plot(x, rm)
    plt.fill_between(x, rm - rs, rm + rs, alpha=0.2)
    plt.title("Prey reward (rolling mean ± std)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"prey_reward_{EXP_NAME}_{env.prey_mode}.png"
        )
    )
    plt.show()


