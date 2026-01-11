"""
compare.py

Evaluation and comparison script for predator policies trained in two environments:
- Toroidal world (periodic boundaries)
- Bounded world with strong reflective walls

This script:
- Loads trained MADDPG predator (and optionally prey) policies
- Runs multiple evaluation episodes without exploration noise
- Computes average predator reward, Degree of Sparsity (DoS), and Degree of Alignment (DoA)
- Prints summary statistics and differences between environments

Model naming is consistent with train.py / visualize.py:
- models/actor_pred_{MODE}_{PREY_MODE}.pth
- models/actor_prey_{MODE}_{PREY_MODE}.pth (only if PREY_MODE == "rl")
"""

import numpy as np
import torch

from maddpg import MADDPGAgent
from env import PredatorPreyEnv as EnvTorus
from env_border_strong import PredatorPreyEnvReflect as EnvWalls  # strong walls version

# ------------------------------------------------------------
# Dimensions / configuration

STATE_DIM = 40
ACTION_DIM = 2

N_PREDATORS = 3
N_PREYS = 20
WORLD_SIZE = 7.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Evaluation parameters

K_EPISODES = 100      # number of episodes for statistics
T_STEPS = 200         # steps per episode
NOISE = 0.0           # no exploration noise during evaluation
ACTION_SCALE = 0.3    # same scaling as visualize.py
SEED0 = 123

# ------------------------------------------------------------
# Model selection (CONSISTENT WITH train.py / visualize.py)

PREY_MODE = "rl"      # "rl" or "couzin"

ACTOR_PRED_TORUS_PATH = f"models/actor_pred_torus_{PREY_MODE}.pth"
ACTOR_PRED_WALLS_PATH = f"models/actor_pred_walls_{PREY_MODE}.pth"

# Only used if PREY_MODE == "rl"
ACTOR_PREY_TORUS_PATH = f"models/actor_prey_torus_{PREY_MODE}.pth"
ACTOR_PREY_WALLS_PATH = f"models/actor_prey_walls_{PREY_MODE}.pth"


# ------------------------------------------------------------
# Utility functions

def set_seed(seed: int):
    """Set random seeds for NumPy and PyTorch (CPU and GPU) for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_reset(env):
    """
    Reset environment with compatibility for different reset signatures.

    Supports:
    - reset() -> pred_obs
    - reset() -> (pred_obs, prey_obs)

    Returns:
        pred_obs, prey_obs (prey_obs can be None)
    """
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, None


def safe_step(env, pred_actions, prey_actions=None):
    """
    Step environment with compatibility for predator-only or predator+prey modes.

    Supports:
    - step(pred_actions)
    - step(pred_actions, prey_actions)

    Returns:
        pred_next_obs, prey_next_obs, pred_rewards, prey_rewards, done, info
    """
    if prey_actions is None:
        pred_next_obs, pred_rewards, done, info = env.step(pred_actions)
        return pred_next_obs, None, pred_rewards, None, done, info
    else:
        (pred_next_obs, prey_next_obs), (pred_rewards, prey_rewards), done, info = env.step(
            pred_actions, prey_actions
        )
        return pred_next_obs, prey_next_obs, pred_rewards, prey_rewards, done, info


def load_agent(actor_path, n_agents):
    """
    Load a MADDPG agent and its actor network from disk.
    """
    agent = MADDPGAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_agents=n_agents,
        device=DEVICE
    )
    agent.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE, weights_only=True))
    agent.actor.eval()
    return agent


def run_eval(env_class, actor_pred_path, use_prey_policy=False, actor_prey_path=None):
    """
    Evaluate predator (and optionally prey) policies in a given environment.
    """
    env = env_class(
        n_prey=N_PREYS,
        n_predators=N_PREDATORS,
        world_size=WORLD_SIZE,
        prey_mode=("rl" if use_prey_policy else "couzin"),
    )

    agent_pred = load_agent(actor_pred_path, n_agents=N_PREDATORS)

    agent_prey = None
    if use_prey_policy:
        if actor_prey_path is None:
            raise ValueError("use_prey_policy=True but actor_prey_path=None")
        agent_prey = load_agent(actor_prey_path, n_agents=N_PREYS)

    ep_pred_reward = []
    ep_prey_reward = []
    ep_dos = []
    ep_doa = []

    for ep in range(K_EPISODES):
        set_seed(SEED0 + ep)

        pred_obs, prey_obs = safe_reset(env)

        pred_r_list = []
        prey_r_list = []
        dos_list = []
        doa_list = []

        for _ in range(T_STEPS):
            pred_actions = np.zeros((N_PREDATORS, ACTION_DIM))
            for i in range(N_PREDATORS):
                a = agent_pred.select_action(pred_obs[i], noise_scale=NOISE)
                pred_actions[i] = ACTION_SCALE * a

            prey_actions = None
            if use_prey_policy:
                if prey_obs is None:
                    raise RuntimeError("use_prey_policy=True but prey_obs is None")
                prey_actions = np.zeros((N_PREYS, ACTION_DIM))
                for j in range(N_PREYS):
                    a = agent_prey.select_action(prey_obs[j], noise_scale=NOISE)
                    prey_actions[j] = ACTION_SCALE * a

            pred_next_obs, prey_next_obs, pred_rewards, prey_rewards, done, info = safe_step(
                env, pred_actions, prey_actions
            )

            pred_r_list.append(float(np.mean(pred_rewards)))
            if prey_rewards is not None:
                prey_r_list.append(float(np.mean(prey_rewards)))

            dos_list.append(float(info.get("DoS", np.nan)))
            doa_list.append(float(info.get("DoA", np.nan)))

            pred_obs = pred_next_obs
            prey_obs = prey_next_obs

            if done:
                break

        ep_pred_reward.append(np.nanmean(pred_r_list))
        ep_dos.append(np.nanmean(dos_list))
        ep_doa.append(np.nanmean(doa_list))

        if use_prey_policy:
            ep_prey_reward.append(np.nanmean(prey_r_list) if len(prey_r_list) else np.nan)

    out = {
        "pred_reward": np.array(ep_pred_reward),
        "dos": np.array(ep_dos),
        "doa": np.array(ep_doa),
    }
    if use_prey_policy:
        out["prey_reward"] = np.array(ep_prey_reward)

    return out


def summarize(arr):
    """Compute mean and standard deviation, ignoring NaNs."""
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def print_block(title, dct, has_prey=False):
    """Print a formatted summary block of evaluation statistics."""
    print(f"\n=== {title} ===")
    m, s = summarize(dct["pred_reward"])
    print(f"Pred reward : mean={m:.4f}  std={s:.4f}")
    m, s = summarize(dct["dos"])
    print(f"DoS         : mean={m:.4f}  std={s:.4f}")
    m, s = summarize(dct["doa"])
    print(f"DoA         : mean={m:.4f}  std={s:.4f}")
    if has_prey:
        m, s = summarize(dct["prey_reward"])
        print(f"Prey reward : mean={m:.4f}  std={s:.4f}")


# ------------------------------------------------------------
# Main script

if __name__ == "__main__":
    # Base comparison: predator policies only, unless PREY_MODE == "rl"
    USE_PREY_POLICY = (PREY_MODE == "rl")

    print("PREY_MODE =", PREY_MODE)
    print("Using TORUS model:", ACTOR_PRED_TORUS_PATH)
    print("Using WALLS model:", ACTOR_PRED_WALLS_PATH)
    if USE_PREY_POLICY:
        print("Using TORUS prey model:", ACTOR_PREY_TORUS_PATH)
        print("Using WALLS prey model:", ACTOR_PREY_WALLS_PATH)

    tor = run_eval(
        EnvTorus,
        actor_pred_path=ACTOR_PRED_TORUS_PATH,
        use_prey_policy=USE_PREY_POLICY,
        actor_prey_path=(ACTOR_PREY_TORUS_PATH if USE_PREY_POLICY else None),
    )

    wal = run_eval(
        EnvWalls,
        actor_pred_path=ACTOR_PRED_WALLS_PATH,
        use_prey_policy=USE_PREY_POLICY,
        actor_prey_path=(ACTOR_PREY_WALLS_PATH if USE_PREY_POLICY else None),
    )

    print_block("TORUS", tor, has_prey=USE_PREY_POLICY)
    print_block("WALLS", wal, has_prey=USE_PREY_POLICY)

    print("\n=== DIFF (WALLS - TORUS) ===")
    m, s = summarize(wal["pred_reward"] - tor["pred_reward"])
    print(f"Pred reward : mean={m:.4f}  std={s:.4f}")
    m, s = summarize(wal["dos"] - tor["dos"])
    print(f"DoS         : mean={m:.4f}  std={s:.4f}")
    m, s = summarize(wal["doa"] - tor["doa"])
    print(f"DoA         : mean={m:.4f}  std={s:.4f}")
    if USE_PREY_POLICY:
        m, s = summarize(wal["prey_reward"] - tor["prey_reward"])
        print(f"Prey reward : mean={m:.4f}  std={s:.4f}")
