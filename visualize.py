import pygame
import numpy as np
import imageio.v2 as imageio
import torch

from env import PredatorPreyEnv
from maddpg import MADDPGAgent

WIDTH, HEIGHT = 800, 800
FPS = 30
N_STEPS = 300
OUTPUT_GIF = "simulation.gif"

STATE_DIM = 6
ACTION_DIM = 2
ACTION_SCALE = 0.3

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# === ENVIRONNEMENT ===
env = PredatorPreyEnv(world_size=7.0)
obs = env.reset()

# === AGENT ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = MADDPGAgent(STATE_DIM, ACTION_DIM, env.n_predators, device=device)

for i in range(env.n_predators):
    agent.actors[i].load_state_dict(
        torch.load(f"models/actor_predator_{i}.pth", map_location=device)
    )
    agent.actors[i].eval()

frames = []

def w2p(pos):
    x, y = pos
    return int(x * 110), int(800 - y * 110)

print("Running simulation...")

for step in range(N_STEPS):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # === ACTIONS ===
    acts = []
    for i in range(env.n_predators):
        t = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            a = agent.actors[i](t).cpu().numpy()[0]
        acts.append(ACTION_SCALE * a)
    acts = np.array(acts)

    obs, rew, done, info = env.step(acts)

    # === DRAW ===
    screen.fill((255, 255, 255))

    # ----- Proies -----
    for i, (x, y) in enumerate(env.prey_pos):
        px, py = w2p((x, y))
        pygame.draw.circle(screen, (0, 200, 0), (px, py), 4)

        # "nez" = direction de la vitesse
        v = env.prey_vel[i]
        if np.linalg.norm(v) > 1e-6:
            heading = np.arctan2(v[1], v[0])
            L = 12  # longueur du nez en pixels
            end_x = int(px + L * np.cos(heading))
            # attention : y écran inversé
            end_y = int(py - L * np.sin(heading))
            pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 1)

    # ----- Prédateurs -----
    for i, (x, y) in enumerate(env.pred_pos):
        px, py = w2p((x, y))
        pygame.draw.circle(screen, (220, 0, 0), (px, py), 7)

        v = env.pred_vel[i]
        if np.linalg.norm(v) > 1e-6:
            heading = np.arctan2(v[1], v[0])
            L = 16  # nez un peu plus long pour les prédateurs
            end_x = int(px + L * np.cos(heading))
            end_y = int(py - L * np.sin(heading))
            pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 2)

    pygame.display.flip()

    # === CAPTURE ===
    arr = pygame.surfarray.array3d(screen)       # (W,H,3)
    frame = np.transpose(arr, (1, 0, 2))         # (H,W,3)
    frames.append(frame.astype(np.uint8))        # copie propre

    clock.tick(FPS)

pygame.quit()

print("Saving GIF...")
imageio.mimsave(OUTPUT_GIF, frames, fps=30)

print("GIF saved as:", OUTPUT_GIF)
