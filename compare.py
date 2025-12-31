import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from maddpg import MADDPGAgent
from env import PredatorPreyEnv as EnvTorus
from env_border_strong import PredatorPreyEnvReflect as EnvReflect


# -----------------------------
# Config
# -----------------------------
N_SEEDS = 10
EPISODES = 50
STEPS_PER_EPISODE = 80

N_PREDATORS = 3
N_PREYS = 20
WORLD_SIZE = 7.0

STATE_DIM = 40
ACTION_DIM = 2

ACTOR_PATH = "models/actor_predator_shared.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utils
# -----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_agent(device):
    agent = MADDPGAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_agents=N_PREDATORS,
        device=device
    )
    if not os.path.exists(ACTOR_PATH):
        raise FileNotFoundError(
            f"Actor introuvable: {ACTOR_PATH}. Lance train.py avant."
        )
    agent.actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    agent.actor.eval()

    # Important : ton MADDPGAgent normalise avec state_mean/state_std.
    # Comme tu ne les sauvegardes pas dans train.py, on garde les valeurs par défaut
    # (mean=0, std=1). Ça permet AU MOINS une comparaison "à pression identique"
    # entre environnements (même acteur, même pipeline).
    agent.state_mean = torch.zeros(STATE_DIM, device=device)
    agent.state_std = torch.ones(STATE_DIM, device=device)
    return agent


def rollout(env, agent, seed: int):
    """
    Retourne:
    - dos: (EPISODES, STEPS_PER_EPISODE)
    - doa: (EPISODES, STEPS_PER_EPISODE)
    - rew: (EPISODES,) reward moyen par épisode (moyenne des prédateurs)
    """
    dos = np.zeros((EPISODES, STEPS_PER_EPISODE), dtype=np.float64)
    doa = np.zeros((EPISODES, STEPS_PER_EPISODE), dtype=np.float64)
    rew = np.zeros((EPISODES,), dtype=np.float64)

    for ep in range(EPISODES):
        # seed fixée -> mêmes tirages reset si on fait pareil dans les 2 env
        set_all_seeds(seed * 10_000 + ep)  # seed différente par épisode, reproductible
        obs = env.reset()

        ep_rew = 0.0
        for t in range(STEPS_PER_EPISODE):
            actions = np.zeros((N_PREDATORS, ACTION_DIM), dtype=np.float32)
            for i in range(N_PREDATORS):
                actions[i] = agent.select_action(obs[i], i, noise_scale=0.0)

            next_obs, rewards, done, info = env.step(actions)
            ep_rew += float(np.mean(rewards))

            dos[ep, t] = float(info.get("DoS", np.nan))
            doa[ep, t] = float(info.get("DoA", np.nan))

            obs = next_obs
            if done:
                # si un jour tu introduis un done=True, on remplit le reste avec NaN
                dos[ep, t+1:] = np.nan
                doa[ep, t+1:] = np.nan
                break

        rew[ep] = ep_rew

    return dos, doa, rew


def mean_std_over_runs(x_list):
    """
    x_list: liste de tableaux (EPISODES, STEPS) pour plusieurs seeds
    On moyenne d'abord sur EPISODES (pour chaque seed), puis sur seeds.
    """
    # -> (n_seeds, EPISODES, STEPS)
    X = np.stack(x_list, axis=0)

    # moyenne par seed (sur EPISODES) => (n_seeds, STEPS)
    per_seed = np.nanmean(X, axis=1)

    mean = np.nanmean(per_seed, axis=0)
    std = np.nanstd(per_seed, axis=0)
    return mean, std


# -----------------------------
# Main
# -----------------------------
def main():
    agent = make_agent(DEVICE)

    torus_dos_runs, torus_doa_runs, torus_rew_runs = [], [], []
    refl_dos_runs,  refl_doa_runs,  refl_rew_runs  = [], [], []

    for s in range(N_SEEDS):
        seed = 12345 + s

        # --- Torus ---
        set_all_seeds(seed)
        env_t = EnvTorus(
            n_prey=N_PREYS,
            n_predators=N_PREDATORS,
            world_size=WORLD_SIZE
        )
        dos_t, doa_t, rew_t = rollout(env_t, agent, seed=seed)
        torus_dos_runs.append(dos_t)
        torus_doa_runs.append(doa_t)
        torus_rew_runs.append(rew_t)

        # --- Reflect (même seed + même protocole) ---
        set_all_seeds(seed)
        env_r = EnvReflect(
            n_prey=N_PREYS,
            n_predators=N_PREDATORS,
            world_size=WORLD_SIZE
        )
        dos_r, doa_r, rew_r = rollout(env_r, agent, seed=seed)
        refl_dos_runs.append(dos_r)
        refl_doa_runs.append(doa_r)
        refl_rew_runs.append(rew_r)

        print(f"[seed {seed}] done")

    # Courbes DoS(t), DoA(t)
    torus_dos_mean, torus_dos_std = mean_std_over_runs(torus_dos_runs)
    torus_doa_mean, torus_doa_std = mean_std_over_runs(torus_doa_runs)
    refl_dos_mean,  refl_dos_std  = mean_std_over_runs(refl_dos_runs)
    refl_doa_mean,  refl_doa_std  = mean_std_over_runs(refl_doa_runs)

    t = np.arange(STEPS_PER_EPISODE)

    # --- Plot DoS ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, torus_dos_mean, label="Torus (mean)")
    plt.fill_between(t, torus_dos_mean - torus_dos_std, torus_dos_mean + torus_dos_std, alpha=0.2)
    plt.plot(t, refl_dos_mean, label="Reflect walls (mean)")
    plt.fill_between(t, refl_dos_mean - refl_dos_std, refl_dos_mean + refl_dos_std, alpha=0.2)
    plt.title("DoS(t) – torus vs reflective walls (same predator policy)")
    plt.xlabel("Step")
    plt.ylabel("Degree of Sparsity (DoS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_DoS_torus_vs_reflect.png", dpi=200)

    # --- Plot DoA ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, torus_doa_mean, label="Torus (mean)")
    plt.fill_between(t, torus_doa_mean - torus_doa_std, torus_doa_mean + torus_doa_std, alpha=0.2)
    plt.plot(t, refl_doa_mean, label="Reflect walls (mean)")
    plt.fill_between(t, refl_doa_mean - refl_doa_std, refl_doa_mean + refl_doa_std, alpha=0.2)
    plt.title("DoA(t) – torus vs reflective walls (same predator policy)")
    plt.xlabel("Step")
    plt.ylabel("Degree of Alignment (DoA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_DoA_torus_vs_reflect.png", dpi=200)

    # --- Résumés scalaires ---
    # Moyenne temporelle par épisode, puis moyenne sur épisodes, puis sur seeds
    def summarize_runs(dos_runs, doa_runs, rew_runs):
        # dos_runs: list of (EPISODES, STEPS)
        dos_seed = []
        doa_seed = []
        rew_seed = []
        for dos, doa, rew in zip(dos_runs, doa_runs, rew_runs):
            dos_seed.append(np.nanmean(dos))  # moyenne sur tout (ep, t)
            doa_seed.append(np.nanmean(doa))
            rew_seed.append(np.mean(rew))     # reward moyen sur EPISODES
        return (np.mean(dos_seed), np.std(dos_seed),
                np.mean(doa_seed), np.std(doa_seed),
                np.mean(rew_seed), np.std(rew_seed))

    td_m, td_s, ta_m, ta_s, tr_m, tr_s = summarize_runs(torus_dos_runs, torus_doa_runs, torus_rew_runs)
    rd_m, rd_s, ra_m, ra_s, rr_m, rr_s = summarize_runs(refl_dos_runs,  refl_doa_runs,  refl_rew_runs)

    print("\n=== Summary (mean ± std across seeds) ===")
    print(f"Torus   : DoS={td_m:.3f} ± {td_s:.3f} | DoA={ta_m:.3f} ± {ta_s:.3f} | Reward={tr_m:.2f} ± {tr_s:.2f}")
    print(f"Reflect : DoS={rd_m:.3f} ± {rd_s:.3f} | DoA={ra_m:.3f} ± {ra_s:.3f} | Reward={rr_m:.2f} ± {rr_s:.2f}")

    plt.show()
    print("\nSaved: compare_DoS_torus_vs_reflect.png, compare_DoA_torus_vs_reflect.png")


if __name__ == "__main__":
    main()
