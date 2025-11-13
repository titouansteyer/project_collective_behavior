import numpy as np

class PredatorPreyEnv:
    def __init__(self, n_prey=30, n_predators=3, world_size=10):
        self.n_prey = n_prey
        self.n_predators = n_predators
        self.world_size = world_size
        self.reset()

    def reset(self):
        self.prey_pos = np.random.rand(self.n_prey, 2) * self.world_size
        self.pred_pos = np.random.rand(self.n_predators, 2) * self.world_size
        self.prey_vel = np.random.uniform(-1, 1, (self.n_prey, 2))
        self.pred_vel = np.random.uniform(-1, 1, (self.n_predators, 2))
        return self.get_obs()

    def get_obs(self):
        # Chaque agent voit sa position et celles des autres
        obs_prey = np.concatenate([self.prey_pos.flatten(), self.pred_pos.flatten()])
        obs_pred = np.concatenate([self.pred_pos.flatten(), self.prey_pos.flatten()])
        return {"prey": obs_prey, "predators": obs_pred}

    def step(self, actions):
        # actions = dict { "prey": (N,2), "predators": (M,2) }
        prey_actions = actions["prey"]
        pred_actions = actions["predators"]

        self.prey_pos += 0.05 * prey_actions
        self.pred_pos += 0.08 * pred_actions

        # Periodic boundaries
        self.prey_pos %= self.world_size
        self.pred_pos %= self.world_size

        rewards_pred = np.zeros(self.n_predators)
        rewards_prey = np.zeros(self.n_prey)
        done = False

        for i, pred in enumerate(self.pred_pos):
            dists = np.linalg.norm(self.prey_pos - pred, axis=1)
            caught = np.where(dists < 0.2)[0]
            if len(caught) > 0:
                rewards_pred[i] += len(caught)
                rewards_prey[caught] = -1
                # Respawn caught prey
                self.prey_pos[caught] = np.random.rand(len(caught), 2) * self.world_size

        return self.get_obs(), {"predators": rewards_pred, "prey": rewards_prey}, done, {}
