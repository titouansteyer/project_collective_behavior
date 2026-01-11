import pygame
import numpy as np
import imageio.v2 as imageio
import torch
import matplotlib.pyplot as plt

from maddpg import MADDPGAgent

# ============================================================
# CHOIX ICI (MINIMAL)
MODE = "walls"        # "torus" ou "walls"
PREY_MODE = "rl"  # "couzin" ou "rl"
# ============================================================

# --- Import env selon MODE ---
if MODE == "torus":
    from env import PredatorPreyEnv as PredatorPreyEnv  # bords infinis (tore)
elif MODE == "walls":
    from env_border_strong import PredatorPreyEnvReflect as PredatorPreyEnv  # bords solides (strong)
else:
    raise ValueError("MODE must be 'torus' or 'walls'")

# ============================================================
# Paths (attention: suffixe _couzin / _rl)
pred_actor_path = f"models/actor_pred_{MODE}_{PREY_MODE}.pth"
prey_actor_path = f"models/actor_prey_{MODE}_{PREY_MODE}.pth"

# Enregistrer dans un fichier qui s'appelle "simulation"
OUTPUT_GIF = f"simulation_{MODE}_{PREY_MODE}.gif"
# ============================================================

WIDTH, HEIGHT = 800, 800
FPS = 30
N_STEPS = 600

STATE_DIM = 40
ACTION_DIM = 2

ACTION_SCALE_PRED = 0.3
ACTION_SCALE_PREY = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


#print
dos_hist = []
doa_hist = []
steps_hist = []



# IMPORTANT: mêmes paramètres que le training
env = PredatorPreyEnv(world_size=7.0, n_prey=40, n_predators=3, prey_mode=PREY_MODE)
print(f"MODE={MODE} | PREY_MODE={PREY_MODE}")
print("env.prey_mode =", getattr(env, "prey_mode", None))

# reset renvoie parfois pred_obs seul OU (pred_obs, prey_obs)
out = env.reset()
if isinstance(out, tuple) and len(out) == 2:
    pred_obs, prey_obs = out
else:
    pred_obs = out
    if hasattr(env, "_get_prey_obs"):
        prey_obs = env._get_prey_obs()
    else:
        raise RuntimeError("Env reset() ne renvoie pas prey_obs et env._get_prey_obs() absent.")

# --- Agents ---
agent_pred = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=env.n_predators,
    device=device
)

agent_prey = None
if PREY_MODE == "rl":
    agent_prey = MADDPGAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        n_agents=env.n_prey,
        device=device
    )

# --- Load trained actors ---
agent_pred.actor.load_state_dict(torch.load(pred_actor_path, map_location=device))
agent_pred.actor.eval()
print(f"[{MODE}] Loaded predator actor from {pred_actor_path}")

if PREY_MODE == "rl":
    agent_prey.actor.load_state_dict(torch.load(prey_actor_path, map_location=device))
    agent_prey.actor.eval()
    print(f"[{MODE}] Loaded prey actor from {prey_actor_path}")
else:
    print(f"[{MODE}] PREY_MODE=couzin -> pas de chargement actor proie (normal).")

frames = []

def w2p(pos):
    x, y = pos
    scale = WIDTH / env.world_size
    px = int(x * scale)
    py = int(HEIGHT - y * scale)
    px = max(0, min(WIDTH - 1, px))
    py = max(0, min(HEIGHT - 1, py))
    return px, py

print(f"Running simulation for visualization ({MODE}, {PREY_MODE})...")

for step in range(N_STEPS):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # --- Predator actions (no noise) ---
    pred_actions = np.zeros((env.n_predators, ACTION_DIM))
    for i in range(env.n_predators):
        a = agent_pred.select_action(pred_obs[i], noise_scale=0.0)
        pred_actions[i] = ACTION_SCALE_PRED * a

    # --- Step env ---
    if PREY_MODE == "rl":
        # --- Prey actions (no noise) ---
        prey_actions = np.zeros((env.n_prey, ACTION_DIM))
        for j in range(env.n_prey):
            a = agent_prey.select_action(prey_obs[j], noise_scale=0.0)
            prey_actions[j] = ACTION_SCALE_PREY * a

        (pred_obs, prey_obs), (rew_pred, rew_prey), done, info = env.step(pred_actions, prey_actions)
        dos = info.get("DoS", np.nan)
        doa = info.get("DoA", np.nan)

        steps_hist.append(step)
        dos_hist.append(dos)
        doa_hist.append(doa)

    else:
        pred_obs, rew_pred, done, info = env.step(pred_actions)
        dos = info.get("DoS", np.nan)
        doa = info.get("DoA", np.nan)

        steps_hist.append(step)
        dos_hist.append(dos)
        doa_hist.append(doa)
        rew_prey = None
        if hasattr(env, "_get_prey_obs"):
            prey_obs = env._get_prey_obs()

    # --- Render ---
    screen.fill((255, 255, 255))

    # prey
    for i, (x, y) in enumerate(env.prey_pos):
        px, py = w2p((x, y))
        pygame.draw.circle(screen, (0, 200, 0), (px, py), 4)

        v = env.prey_vel[i]
        if np.linalg.norm(v) > 1e-6:
            heading = np.arctan2(v[1], v[0])
            L = 14
            end_x = int(px + L * np.cos(heading))
            end_y = int(py - L * np.sin(heading))
            pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 1)

    # predators
    for i, (x, y) in enumerate(env.pred_pos):
        px, py = w2p((x, y))
        pygame.draw.circle(screen, (220, 0, 0), (px, py), 7)

        v = env.pred_vel[i]
        if np.linalg.norm(v) > 1e-6:
            heading = np.arctan2(v[1], v[0])
            L = 18
            end_x = int(px + L * np.cos(heading))
            end_y = int(py - L * np.sin(heading))
            pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 2)

    if step % 50 == 0:
        print(f"Step {step:04d} | DoS={info.get('DoS', np.nan):.3f}, DoA={info.get('DoA', np.nan):.3f}")

    pygame.display.flip()


    # capture frame
    arr = pygame.surfarray.array3d(screen)
    frame = np.transpose(arr, (1, 0, 2))
    frames.append(frame.astype(np.uint8))

    clock.tick(FPS)

pygame.quit()
print("Saving GIF...")
imageio.mimsave(OUTPUT_GIF, frames, fps=FPS)
print(f"GIF saved as {OUTPUT_GIF}")

print("Plotting DoS / DoA...")
plt.figure()
plt.plot(steps_hist, dos_hist, label="DoS")
plt.plot(steps_hist, doa_hist, label="DoA")
plt.xlabel("Step")
plt.ylabel("Metric value")
plt.title(f"Collective metrics (MODE={MODE}, PREY_MODE={PREY_MODE})")
plt.legend()
plt.tight_layout()

plot_name = f"metrics_{MODE}_{PREY_MODE}.png"
plt.savefig(plot_name, dpi=200)
plt.show()
print(f"Saved plot as {plot_name}")

