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

    # Proies
    for x, y in env.prey_pos:
        pygame.draw.circle(screen, (0, 200, 0), w2p((x, y)), 4)

    # Prédateurs
    for x, y in env.pred_pos:
        pygame.draw.circle(screen, (220, 0, 0), w2p((x, y)), 7)

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
