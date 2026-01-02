import pygame
import numpy as np
import imageio.v2 as imageio
import torch

# from env import PredatorPreyEnv  # bords infinis
from env_border_strong import PredatorPreyEnvReflect as PredatorPreyEnv  # bords solides

from maddpg import MADDPGAgent

WIDTH, HEIGHT = 800, 800
FPS = 30
N_STEPS = 600
OUTPUT_GIF = "simulation.gif"

STATE_DIM = 40
ACTION_DIM = 2

# Tu peux mettre deux échelles différentes si besoin
ACTION_SCALE_PRED = 0.3
ACTION_SCALE_PREY = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# IMPORTANT: mets les mêmes paramètres que le training (world_size, n_prey, n_predators)
env = PredatorPreyEnv(world_size=7.0, n_prey=40, n_predators=3)

# reset renvoie les obs des deux espèces
pred_obs, prey_obs = env.reset()

# --- Agents ---
agent_pred = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=env.n_predators,
    device=device
)

agent_prey = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=env.n_prey,
    device=device
)

# --- Load trained actors ---
pred_actor_path = "models/actor_predator_shared.pth"
prey_actor_path = "models/actor_prey_shared.pth"

agent_pred.actor.load_state_dict(torch.load(pred_actor_path, map_location=device))
agent_pred.actor.eval()
print(f"Loaded predator actor from {pred_actor_path}")

agent_prey.actor.load_state_dict(torch.load(prey_actor_path, map_location=device))
agent_prey.actor.eval()
print(f"Loaded prey actor from {prey_actor_path}")

frames = []

def w2p(pos):
    x, y = pos
    scale = WIDTH / env.world_size
    px = int(x * scale)
    py = int(HEIGHT - y * scale)

    # clamp to screen (avoid disappearing on borders)
    px = max(0, min(WIDTH - 1, px))
    py = max(0, min(HEIGHT - 1, py))

    return px, py


print("Running simulation for visualization...")

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

    # --- Prey actions (no noise) ---
    prey_actions = np.zeros((env.n_prey, ACTION_DIM))
    for j in range(env.n_prey):
        a = agent_prey.select_action(prey_obs[j], noise_scale=0.0)
        prey_actions[j] = ACTION_SCALE_PREY * a

    # --- Step env (coevolution) ---
    (pred_obs, prey_obs), (rew_pred, rew_prey), done, info = env.step(pred_actions, prey_actions)

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

    # optional: afficher DoS/DoA en console toutes les 50 steps
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
