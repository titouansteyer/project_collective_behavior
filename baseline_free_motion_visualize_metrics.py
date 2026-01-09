
"""
baseline_free_motion_visualize_metrics.py

Same as baseline_free_motion_visualize.py (visualize-style pygame rendering + GIF),
PLUS logs and plots the time evolution of DoS and DoA.

Metrics implementation uses your existing metrics.py:
- degree_of_sparsity(positions, world_size=world_size)
- degree_of_alignment(velocities)

By default we compute DoS/DoA on PREY only (same spirit as env.py + Li paper).
Option --all_agents_metrics computes them on all agents (prey + predators).

Run:
    python baseline_free_motion_visualize_metrics.py --mode torus --steps 1500 --n_prey 40 --gif out.gif --plot_metrics

Also save CSV:
    python baseline_free_motion_visualize_metrics.py --gif out.gif --csv metrics.csv --plot_metrics
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from metrics import degree_of_sparsity, degree_of_alignment


@dataclass
class Config:
    n_pred: int = 3
    n_prey: int = 10
    world_size: float = 7.0
    dt: float = 0.1
    steps: int = 1500

    pred_speed: float = 0.8
    prey_speed: float = 0.6
    turn_sigma: float = 0.9

    mode: str = "torus"  # torus or walls
    seed: int = 0

    # render like visualize.py
    width: int = 800
    height: int = 800
    fps: int = 25


def wrap_torus(pos: np.ndarray, L: float) -> np.ndarray:
    return np.mod(pos, L)


def reflect_walls(pos: np.ndarray, vel: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray]:
    for axis in (0, 1):
        over = pos[:, axis] > L
        if np.any(over):
            pos[over, axis] = 2 * L - pos[over, axis]
            vel[over, axis] *= -1.0
        under = pos[:, axis] < 0.0
        if np.any(under):
            pos[under, axis] = -pos[under, axis]
            vel[under, axis] *= -1.0
    return pos, vel


def simulate(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    L = float(cfg.world_size)
    T = int(cfg.steps) + 1
    dt = float(cfg.dt)

    pred_pos = rng.random((cfg.n_pred, 2)) * L
    prey_pos = rng.random((cfg.n_prey, 2)) * L

    pred_theta = rng.uniform(0, 2 * math.pi, size=(cfg.n_pred,))
    prey_theta = rng.uniform(0, 2 * math.pi, size=(cfg.n_prey,))

    pred_pos_traj = np.zeros((T, cfg.n_pred, 2), dtype=float)
    prey_pos_traj = np.zeros((T, cfg.n_prey, 2), dtype=float)
    pred_vel_traj = np.zeros((T, cfg.n_pred, 2), dtype=float)
    prey_vel_traj = np.zeros((T, cfg.n_prey, 2), dtype=float)

    pred_vel = np.stack([np.cos(pred_theta), np.sin(pred_theta)], axis=1) * float(cfg.pred_speed)
    prey_vel = np.stack([np.cos(prey_theta), np.sin(prey_theta)], axis=1) * float(cfg.prey_speed)

    pred_pos_traj[0] = pred_pos
    prey_pos_traj[0] = prey_pos
    pred_vel_traj[0] = pred_vel
    prey_vel_traj[0] = prey_vel

    turn_scale = float(cfg.turn_sigma) * math.sqrt(dt)

    for t in range(1, T):
        pred_theta = pred_theta + rng.normal(0.0, turn_scale, size=pred_theta.shape)
        prey_theta = prey_theta + rng.normal(0.0, turn_scale, size=prey_theta.shape)

        pred_vel = np.stack([np.cos(pred_theta), np.sin(pred_theta)], axis=1) * float(cfg.pred_speed)
        prey_vel = np.stack([np.cos(prey_theta), np.sin(prey_theta)], axis=1) * float(cfg.prey_speed)

        pred_pos = pred_pos + pred_vel * dt
        prey_pos = prey_pos + prey_vel * dt

        if cfg.mode == "torus":
            pred_pos = wrap_torus(pred_pos, L)
            prey_pos = wrap_torus(prey_pos, L)
        else:
            pred_pos, pred_vel = reflect_walls(pred_pos, pred_vel, L)
            prey_pos, prey_vel = reflect_walls(prey_pos, prey_vel, L)
            pred_theta = np.arctan2(pred_vel[:, 1], pred_vel[:, 0])
            prey_theta = np.arctan2(prey_vel[:, 1], prey_vel[:, 0])

        pred_pos_traj[t] = pred_pos
        prey_pos_traj[t] = prey_pos
        pred_vel_traj[t] = pred_vel
        prey_vel_traj[t] = prey_vel

    return pred_pos_traj, prey_pos_traj, pred_vel_traj, prey_vel_traj


def compute_metrics_series(
    prey_pos_traj: np.ndarray,
    prey_vel_traj: np.ndarray,
    pred_pos_traj: np.ndarray,
    pred_vel_traj: np.ndarray,
    world_size: float,
    all_agents: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    T = prey_pos_traj.shape[0]
    dos = np.zeros(T, dtype=float)
    doa = np.zeros(T, dtype=float)

    for t in range(T):
        if all_agents:
            pos = np.vstack([prey_pos_traj[t], pred_pos_traj[t]])
            vel = np.vstack([prey_vel_traj[t], pred_vel_traj[t]])
        else:
            pos = prey_pos_traj[t]
            vel = prey_vel_traj[t]

        dos[t] = degree_of_sparsity(pos, world_size=world_size)
        doa[t] = degree_of_alignment(vel)

    return dos, doa


def make_gif_and_metrics(
    cfg: Config,
    gif_path: str,
    gif_stride: int,
    no_window: bool,
    plot_metrics: bool,
    csv_path: Optional[str],
    all_agents_metrics: bool,
) -> None:
    import pygame
    import imageio.v2 as imageio

    pred_pos_traj, prey_pos_traj, pred_vel_traj, prey_vel_traj = simulate(cfg)
    dos, doa = compute_metrics_series(
        prey_pos_traj, prey_vel_traj, pred_pos_traj, pred_vel_traj,
        world_size=float(cfg.world_size),
        all_agents=all_agents_metrics,
    )

    if csv_path:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestep", "DoS", "DoA"])
            for t in range(len(dos)):
                w.writerow([t, float(dos[t]), float(doa[t])])
        print(f"Saved CSV: {csv_path}")

    WIDTH, HEIGHT = cfg.width, cfg.height
    L = float(cfg.world_size)

    pygame.init()
    flags = pygame.HIDDEN if no_window else 0
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=flags)
    pygame.display.set_caption("Baseline free motion (visualize-style) + DoS/DoA")
    clock = pygame.time.Clock()

    def w2p(pos):
        x, y = pos
        scale = WIDTH / L
        px = int(x * scale)
        py = int(HEIGHT - y * scale)
        px = max(0, min(WIDTH - 1, px))
        py = max(0, min(HEIGHT - 1, py))
        return px, py

    frames = []
    stride = max(1, int(gif_stride))

    for step in range(0, cfg.steps + 1, stride):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        screen.fill((255, 255, 255))

        # prey
        prey_pos = prey_pos_traj[step]
        prey_vel = prey_vel_traj[step]
        for i, (x, y) in enumerate(prey_pos):
            px, py = w2p((x, y))
            pygame.draw.circle(screen, (0, 200, 0), (px, py), 4)
            v = prey_vel[i]
            if np.linalg.norm(v) > 1e-6:
                heading = math.atan2(v[1], v[0])
                Lh = 14
                end_x = int(px + Lh * math.cos(heading))
                end_y = int(py - Lh * math.sin(heading))
                pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 1)

        # predators
        pred_pos = pred_pos_traj[step]
        pred_vel = pred_vel_traj[step]
        for i, (x, y) in enumerate(pred_pos):
            px, py = w2p((x, y))
            pygame.draw.circle(screen, (220, 0, 0), (px, py), 7)
            v = pred_vel[i]
            if np.linalg.norm(v) > 1e-6:
                heading = math.atan2(v[1], v[0])
                Lh = 18
                end_x = int(px + Lh * math.cos(heading))
                end_y = int(py - Lh * math.sin(heading))
                pygame.draw.line(screen, (0, 0, 0), (px, py), (end_x, end_y), 2)

        # console metric every ~50 original steps
        if step % 50 == 0:
            tag = "ALL" if all_agents_metrics else "PREY"
            print(f"Step {step:04d} [{tag}] | DoS={dos[step]:.3f}, DoA={doa[step]:.3f}")

        pygame.display.flip()

        arr = pygame.surfarray.array3d(screen)
        frame = np.transpose(arr, (1, 0, 2))
        frames.append(frame.astype(np.uint8))

        clock.tick(cfg.fps)

    pygame.quit()
    print("Saving GIF...")
    imageio.mimsave(gif_path, frames, fps=cfg.fps)
    print(f"Saved GIF: {gif_path}")

    if plot_metrics:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(dos, label="DoS")
        plt.plot(doa, label="DoA")
        plt.ylim(0, 1)
        plt.xlabel("Timestep")
        plt.ylabel("Metric value")
        title_tag = "all agents" if all_agents_metrics else "prey only"
        plt.title(f"DoS/DoA evolution (baseline free motion) — {title_tag}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline free motion (visualize-style GIF) + DoS/DoA time series.")
    p.add_argument("--mode", type=str, default="torus", choices=["torus", "walls"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--world_size", type=float, default=7.0)

    p.add_argument("--n_pred", type=int, default=3)
    p.add_argument("--n_prey", type=int, default=10)
    p.add_argument("--pred_speed", type=float, default=0.8)
    p.add_argument("--prey_speed", type=float, default=0.6)
    p.add_argument("--turn_sigma", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--gif", type=str, required=True)
    p.add_argument("--gif_stride", type=int, default=1)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--no_window", action="store_true")

    p.add_argument("--plot_metrics", action="store_true", help="Show DoS/DoA curves after rendering.")
    p.add_argument("--csv", type=str, default=None, help="Save DoS/DoA timeseries to CSV.")
    p.add_argument("--all_agents_metrics", action="store_true", help="Compute DoS/DoA on prey+predators.")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = Config(
        n_pred=a.n_pred,
        n_prey=a.n_prey,
        world_size=a.world_size,
        dt=a.dt,
        steps=a.steps,
        pred_speed=a.pred_speed,
        prey_speed=a.prey_speed,
        turn_sigma=a.turn_sigma,
        mode=a.mode,
        seed=a.seed,
        fps=a.fps,
    )
    make_gif_and_metrics(
        cfg=cfg,
        gif_path=a.gif,
        gif_stride=a.gif_stride,
        no_window=a.no_window,
        plot_metrics=a.plot_metrics,
        csv_path=a.csv,
        all_agents_metrics=a.all_agents_metrics,
    )


if __name__ == "__main__":
    main()
