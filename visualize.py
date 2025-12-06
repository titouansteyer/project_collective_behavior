import pygame
import numpy as np
import imageio.v2 as imageio
import torch

#from env import PredatorPreyEnv #pour bord infini
from env_border_strong import PredatorPreyEnvReflect as PredatorPreyEnv#pour bord solide
from maddpg import MADDPGAgent

WIDTH, HEIGHT = 800, 800
FPS = 30
N_STEPS = 500
OUTPUT_GIF = "simulation.gif"

STATE_DIM = 40
ACTION_DIM = 2
ACTION_SCALE = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

env = PredatorPreyEnv(world_size=7.0)
obs = env.reset()

agent = MADDPGAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    n_agents=env.n_predators,
    device=device
)

# load trained actor
actor_path = "models/actor_predator_shared.pth"
agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
agent.actor.eval()
print(f"Loaded shared actor from {actor_path}")

frames = []

def w2p(pos):
    x, y = pos
    # stretch to fill the window
    scale = WIDTH / env.world_size
    px = int(x * scale)
    py = int(HEIGHT - y * scale)
    return px, py

print("Running simulation for visualization...")

for step in range(N_STEPS):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # predator actions (no noise)
    actions = []
    for i in range(env.n_predators):
        a = agent.select_action(obs[i], noise_scale=0.0)
        actions.append(ACTION_SCALE * a)
    actions = np.array(actions)

    obs, rew, done, info = env.step(actions)

    # rendering
    screen.fill((255, 255, 255))

    # prey
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

    # predators
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

    # capture
    arr = pygame.surfarray.array3d(screen)
    frame = np.transpose(arr, (1, 0, 2))
    frames.append(frame.astype(np.uint8))

    clock.tick(FPS)

pygame.quit()
print("Saving GIF...")
imageio.mimsave(OUTPUT_GIF, frames, fps=30)
print(f"GIF saved as {OUTPUT_GIF}")
