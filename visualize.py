import pygame
import numpy as np
import imageio.v2 as imageio
import torch

from env import PredatorPreyEnv
from maddpg import MADDPGAgent   # nouvelle version (acteur partagé)

WIDTH, HEIGHT = 800, 800
FPS = 30
N_STEPS = 300
OUTPUT_GIF = "simulation.gif"

STATE_DIM = 40   # doit matcher env._get_predator_obs()
ACTION_DIM = 2
ACTION_SCALE = 0.3  # pour calmer un peu les actions en visu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# === ENVIRONNEMENT ===
env = PredatorPreyEnv(world_size=7.0)
obs = env.reset()     # shape (n_predators, STATE_DIM)

# === AGENT ===
agent = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=env.n_predators,
    device=device
)

# Charger l’acteur partagé entraîné
actor_path = "models/actor_predator_shared.pth"
agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
agent.actor.eval()
print(f"Loaded shared actor from {actor_path}")

frames = []

def w2p(pos):
    """coordonnées monde -> pixels (y inversé)."""
    x, y = pos
    return int(x * 110), int(HEIGHT - y * 110)

print("Running simulation...")

for step in range(N_STEPS):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # === ACTIONS RL (une pour chaque prédateur) ===
    actions = []
    for i in range(env.n_predators):
        a = agent.select_action(obs[i], noise_scale=0.0)   # pas de bruit en eval
        actions.append(ACTION_SCALE * a)
    actions = np.array(actions)

    # step environnement
    obs, rew, done, info = env.step(actions)

    # === DRAW ===
    screen.fill((255, 255, 255))

    # Proies
    for i, (x, y) in enumerate(env.prey_pos):
        px, py = w2p((x, y))
        pygame.draw.circle(screen, (0, 200, 0), (px, py), 4)

        v = env.prey_vel[i]
        if np.linalg.norm(v) > 1e-6:
            heading = np.arctan2(v[1], v[0])
            L = 12
            end_x = int(px + L * np.cos(heading))
            end_y = int(py - L * np.sin(heading))
            pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 1)

    # Prédateurs
    for i, (x, y) in enumerate(env.pred_pos):
        px, py = w2p((x, y))
        pygame.draw.circle(screen, (220, 0, 0), (px, py), 7)

        v = env.pred_vel[i]
        if np.linalg.norm(v) > 1e-6:
            heading = np.arctan2(v[1], v[0])
            L = 16
            end_x = int(px + L * np.cos(heading))
            end_y = int(py - L * np.sin(heading))
            pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 2)

    pygame.display.flip()

    # === CAPTURE GIF ===
    arr = pygame.surfarray.array3d(screen)   # (W,H,3)
    frame = np.transpose(arr, (1, 0, 2))     # (H,W,3)
    frames.append(frame.astype(np.uint8))

    clock.tick(FPS)

pygame.quit()

print("Saving GIF...")
imageio.mimsave(OUTPUT_GIF, frames, fps=30)
print("GIF saved as:", OUTPUT_GIF)
