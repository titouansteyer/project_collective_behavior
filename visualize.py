import pygame
import numpy as np
import imageio.v2 as imageio
import torch

from env import PredatorPreyEnv
from maddpg import MADDPGAgent

WIDTH, HEIGHT = 800, 800
FPS = 60
N_STEPS = 600
OUTPUT_VIDEO = "simulation_fixed.mp4"

STATE_DIM = 6
ACTION_DIM = 2
ACTION_SCALE = 0.3

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), depth=32)
clock = pygame.time.Clock()

env = PredatorPreyEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = MADDPGAgent(STATE_DIM, ACTION_DIM, env.n_predators, device=device)

# Chargement des modèles entraînés
for i in range(env.n_predators):
    agent.actors[i].load_state_dict(
        torch.load(f"models/actor_predator_{i}.pth", map_location=device)
    )
    agent.actors[i].eval()

def world_to_px(pos):
    x, y = pos
    px = int(x * (WIDTH / env.world_size))
    py = int(HEIGHT - y * (HEIGHT / env.world_size))
    return px, py

def draw_all():
    screen.fill((255, 255, 255))

    # proies
    for i,(x,y) in enumerate(env.prey_pos):
        px,py = world_to_px((x,y))
        color = (0,200,0) if env.prey_alive[i] else (150,150,150)
        pygame.draw.circle(screen, color, (px,py), 4)

    # prédateurs
    for (x,y) in env.pred_pos:
        px,py = world_to_px((x,y))
        pygame.draw.circle(screen, (220,0,0), (px,py), 6)

obs = env.reset()

with imageio.get_writer(OUTPUT_VIDEO, fps=30, codec="libx264") as writer:
    for step in range(N_STEPS):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # actions RL
        actions = np.zeros((env.n_predators, 2))
        for i in range(env.n_predators):
            state_t = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                a = agent.actors[i](state_t).cpu().numpy()[0]
            actions[i] = ACTION_SCALE * a

        obs, rewards, done, metrics = env.step(actions)

        draw_all()
        pygame.display.flip()

        # -------------------------------
        #  LECTURE D’ÉCRAN **SANS BUG**
        # -------------------------------
        arr = pygame.surfarray.pixels3d(screen)   # (W, H, 3)
        frame = np.transpose(arr, (1, 0, 2))      # (H, W, 3)

        frame = np.array(frame, dtype=np.uint8, copy=True)  # COPIE CONTIGUË OK

        writer.append_data(frame)

        clock.tick(FPS)

pygame.quit()
print("Vidéo enregistrée :", OUTPUT_VIDEO)
