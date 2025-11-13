import numpy as np
from metrics import degree_of_sparsity, degree_of_alignment


class PredatorPreyEnv:
    def __init__(
        self,
        n_prey=20,
        n_predators=3,
        world_size=10.0,
        dt=0.1,
        prey_speed_limit=0.8,
        pred_speed_limit=1.0,
        friction=0.3,
        catch_radius=0.25,
        prey_noise_std=0.4
    ):
        self.n_prey = n_prey
        self.n_predators = n_predators
        self.world_size = world_size
        self.dt = dt
        self.prey_speed_limit = prey_speed_limit
        self.pred_speed_limit = pred_speed_limit
        self.friction = friction
        self.catch_radius = catch_radius
        self.prey_noise_std = prey_noise_std

        self.reset()

    # --------------------------------------------------------
    def reset(self):
        self.prey_pos = np.random.rand(self.n_prey, 2) * self.world_size
        self.pred_pos = np.random.rand(self.n_predators, 2) * self.world_size

        self.prey_vel = np.random.uniform(-0.5, 0.5, (self.n_prey, 2))
        self.pred_vel = np.random.uniform(-0.5, 0.5, (self.n_predators, 2))

        return self._get_predator_obs()

    # --------------------------------------------------------
    def _get_predator_obs(self):
        obs = []
        for i in range(self.n_predators):
            px, py = self.pred_pos[i]
            vx, vy = self.pred_vel[i]

            # proie la plus proche
            diffs = self.prey_pos - self.pred_pos[i]
            dists = np.linalg.norm(diffs, axis=1)
            j = np.argmin(dists)
            dx, dy = diffs[j]

            obs.append([px, py, vx, vy, dx, dy])
        return np.array(obs, dtype=float)

    # --------------------------------------------------------
    def step(self, predator_actions):

        predator_actions = np.asarray(predator_actions)
        prey_actions = self.prey_noise_std * np.random.randn(self.n_prey, 2)

        # Update vitesses
        self.pred_vel += self.dt * predator_actions
        self.prey_vel += self.dt * prey_actions

        self.pred_vel *= (1 - self.friction * self.dt)
        self.prey_vel *= (1 - self.friction * self.dt)

        # clamp
        pred_s = np.linalg.norm(self.pred_vel, axis=1, keepdims=True)
        prey_s = np.linalg.norm(self.prey_vel, axis=1, keepdims=True)

        self.pred_vel = np.where(pred_s > self.pred_speed_limit,
                                 self.pred_vel * self.pred_speed_limit / pred_s,
                                 self.pred_vel)
        self.prey_vel = np.where(prey_s > self.prey_speed_limit,
                                 self.prey_vel * self.prey_speed_limit / prey_s,
                                 self.prey_vel)

        # déplacement
        self.pred_pos = (self.pred_pos + self.dt * self.pred_vel) % self.world_size
        self.prey_pos = (self.prey_pos + self.dt * self.prey_vel) % self.world_size

        # reward
        rewards = np.zeros(self.n_predators)
        for i, pred in enumerate(self.pred_pos):
            dists = np.linalg.norm(self.prey_pos - pred, axis=1)
            rewards[i] += np.sum(dists < self.catch_radius)   # +1 par proie touchée
            rewards[i] -= 0.01 * np.linalg.norm(predator_actions[i])  # énergie

        # metrics
        dos = degree_of_sparsity(self.prey_pos)
        doa = degree_of_alignment(self.prey_vel)

        return self._get_predator_obs(), rewards, False, {"DoS": dos, "DoA": doa}
