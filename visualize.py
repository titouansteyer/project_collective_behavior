import pygame
import numpy as np
import imageio
import os
from env import PredatorPreyEnv
import cv2
import imageio.v3 as iio

# --- PARAMÈTRES GÉNÉRAUX ---
WIDTH, HEIGHT = 800, 800
WORLD_SIZE = 7
SCALE = WIDTH / WORLD_SIZE
FPS = 60
N_STEPS = 1500
OUTPUT_VIDEO = "simulation_couzin.mp4"

# --- CRÉER DOSSIER TEMPORAIRE ---
os.makedirs("frames", exist_ok=True)

# --- INITIALISATION PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator–Prey Simulation (Couzin Model)")
clock = pygame.time.Clock()

# --- COULEURS ---
WHITE = (255, 255, 255)
LIGHT_GRID = (220, 220, 220)
DARK_GRID = (180, 180, 180)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# --- ENVIRONNEMENT ---
env = PredatorPreyEnv()

# --- FONCTIONS ---
def draw_grid():
    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, LIGHT_GRID, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, LIGHT_GRID, (0, y), (WIDTH, y))
    for x in range(0, WIDTH, 100):
        pygame.draw.line(screen, DARK_GRID, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 100):
        pygame.draw.line(screen, DARK_GRID, (0, y), (WIDTH, y))

def draw_agents(prey_pos, pred_pos):
    # Draw prey with heading (nose)
    for i, (x, y) in enumerate(prey_pos):
        px = int(x * SCALE)
        py = int(y * SCALE)
        pygame.draw.circle(screen, GREEN, (px, py), 5)
        # Draw heading line if velocity available
        try:
            v = env.prey_velocities[i]
            heading = np.arctan2(v[1], v[0])
            line_length = 12  # pixels
            end_x = int(px + line_length * np.cos(heading))
            end_y = int(py + line_length * np.sin(heading))
            pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 2)
        except Exception:
            # If velocities aren't present or indexing fails, skip heading
            pass

    # Draw predators with heading (nose)
    for i, (x, y) in enumerate(pred_pos):
        px = int(x * SCALE)
        py = int(y * SCALE)
        pygame.draw.circle(screen, RED, (px, py), 7)
        try:
            v = env.predator_velocities[i]
            heading = np.arctan2(v[1], v[0])
            line_length = 15  # predators slightly longer
            end_x = int(px + line_length * np.cos(heading))
            end_y = int(py + line_length * np.sin(heading))
            pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 2)
        except Exception:
            pass

def draw_info(step):
    font = pygame.font.Font(None, 30)
    text = font.render(f"Step: {step}/{N_STEPS}", True, BLACK)
    screen.blit(text, (10, 10))

print(f"Recording {N_STEPS} steps...")

for step in range(N_STEPS):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Mise à jour de la simulation
    (prey_pos, pred_pos), _ = env.step()

    # Rendu visuel
    screen.fill(WHITE)
    draw_grid()
    draw_agents(prey_pos, pred_pos)
    draw_info(step)
    pygame.display.flip()

    # Sauvegarde image
    frame_path = f"frames/frame_{step:04d}.png"
    pygame.image.save(screen, frame_path)

    clock.tick(FPS)

# --- CONSTRUCTION VIDÉO ---
print("Assembling frames into video...")
with imageio.get_writer(OUTPUT_VIDEO, fps=30, codec='libx264') as writer:
    for step in range(N_STEPS):
        frame_path = f"frames/frame_{step:04d}.png"
        # Lecture avec cv2 pour éviter les erreurs de format
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        writer.append_data(image)


# Nettoyage (optionnel)
for file in os.listdir("frames"):
    os.remove(os.path.join("frames", file))
os.rmdir("frames")

pygame.quit()
print(f"Video saved as {OUTPUT_VIDEO}")
