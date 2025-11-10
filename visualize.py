import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from env import PredatorPreyEnv

def animate_environment(n_steps=200, world_size=10, n_prey=30, n_predators=3):
    env = PredatorPreyEnv(n_prey=n_prey, n_predators=n_predators, world_size=world_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_title("Predator–Prey Environment (Random baseline)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    prey_scatter, = ax.plot([], [], 'go', markersize=6, label='Prey')
    pred_scatter, = ax.plot([], [], 'ro', markersize=6, label='Predators')
    ax.legend(loc="upper right")

    def init():
        prey_scatter.set_data([], [])
        pred_scatter.set_data([], [])
        return prey_scatter, pred_scatter

    def update(frame):
        # random movements
        predator_actions = np.random.uniform(-1, 1, (env.n_predators, 2))
        prey_actions = np.random.uniform(-1, 1, (env.n_prey, 2))
        (prey_pos, pred_pos), _ = env.step(predator_actions, prey_actions)

        prey_scatter.set_data(prey_pos[:, 0], prey_pos[:, 1])
        pred_scatter.set_data(pred_pos[:, 0], pred_pos[:, 1])
        return prey_scatter, pred_scatter

    ani = FuncAnimation(fig, update, frames=n_steps, init_func=init,
                        blit=True, interval=80, repeat=False)
    plt.show()

if __name__ == "__main__":
    animate_environment()