import numpy as np
from metrics import degree_of_sparsity, degree_of_alignment


class PredatorPreyEnv:
    def __init__(
        self,
        n_prey=20,
        n_predators=3,
        world_size=10.0,
        dt=0.1,
        prey_speed_limit=1.0,
        pred_speed_limit=1.2,
        friction=0.2,
        catch_radius=0.2,
        prey_noise_std=0.5
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

    # -------------------------------------------------------
    def reset(self):
        self.prey_pos = np.random.rand(self.n_prey, 2) * self.world_size
        self.pred_pos = np.random.rand(self.n_predators, 2) * self.world_size

        self.prey_vel = np.random.uniform(-0.5, 0.5, (self.n_prey, 2))
        self.pred_vel = np.random.uniform(-0.5, 0.5, (self.n_predators, 2))

        # <<< TRUE = encore vivante, FALSE = attrapée
        self.prey_alive = np.ones(self.n_prey, dtype=bool)

        return self._get_predator_obs()

    # -------------------------------------------------------
    # obs = [x, y, vx, vy, dx_closest_prey, dy_closest_prey]
    # -------------------------------------------------------
    def _get_predator_obs(self):
        obs_list = []
        for i in range(self.n_predators):
            px, py = self.pred_pos[i]
            vx, vy = self.pred_vel[i]

            # proie la plus proche parmi les vivantes
            alive_idx = np.where(self.prey_alive)[0]
            if len(alive_idx) > 0:
                diffs = self.prey_pos[alive_idx] - self.pred_pos[i]
                dists = np.linalg.norm(diffs, axis=1)
                idx_local = np.argmin(dists)
                dx, dy = diffs[idx_local]
            else:
                dx, dy = 0.0, 0.0

            obs_list.append([px, py, vx, vy, dx, dy])

        return np.array(obs_list, dtype=float)

    # -------------------------------------------------------
    def step(self, predator_actions):
        predator_actions = np.asarray(predator_actions, dtype=float)
        if predator_actions.shape != (self.n_predators, 2):
            raise ValueError(
                f"predator_actions doit avoir la shape "
                f"({self.n_predators}, 2), mais a {predator_actions.shape}"
            )

        # --- actions des proies (vivantes seulement) ---
        prey_actions = np.zeros_like(self.prey_pos)
        alive_idx = np.where(self.prey_alive)[0]
        if len(alive_idx) > 0:
            prey_actions[alive_idx] = (
                self.prey_noise_std * np.random.randn(len(alive_idx), 2)
            )

        # ========== VITESSES ==========
        self.pred_vel += self.dt * predator_actions
        self.prey_vel += self.dt * prey_actions

        self.pred_vel *= (1.0 - self.friction * self.dt)
        self.prey_vel *= (1.0 - self.friction * self.dt)

        pred_speed = np.linalg.norm(self.pred_vel, axis=1, keepdims=True) + 1e-8
        prey_speed = np.linalg.norm(self.prey_vel, axis=1, keepdims=True) + 1e-8

        self.pred_vel = np.where(
            pred_speed > self.pred_speed_limit,
            self.pred_vel * (self.pred_speed_limit / pred_speed),
            self.pred_vel
        )
        self.prey_vel = np.where(
            prey_speed > self.prey_speed_limit,
            self.prey_vel * (self.prey_speed_limit / prey_speed),
            self.prey_vel
        )

        # les proies mortes ne bougent plus
        self.prey_vel[~self.prey_alive] = 0.0

        # ========== POSITIONS ==========
        self.pred_pos += self.dt * self.pred_vel
        self.prey_pos += self.dt * self.prey_vel

        self.pred_pos %= self.world_size
        self.prey_pos %= self.world_size

        # ========== RÉCOMPENSES ==========
        rewards_pred = np.zeros(self.n_predators, dtype=float)

        # collisions uniquement avec les proies vivantes
        alive_idx = np.where(self.prey_alive)[0]
        for i, pred in enumerate(self.pred_pos):
            if len(alive_idx) == 0:
                break
            dists = np.linalg.norm(self.prey_pos[alive_idx] - pred, axis=1)
            caught_local = np.where(dists < self.catch_radius)[0]
            if len(caught_local) > 0:
                caught_global = alive_idx[caught_local]
                rewards_pred[i] += float(len(caught_global))

                # marquer ces proies comme mortes, elles restent visibles
                self.prey_alive[caught_global] = False
                self.prey_vel[caught_global] = 0.0

        rewards_pred -= 0.01 * np.linalg.norm(predator_actions, axis=1)

        # ========== METRICS (sur les proies VIVANTES) ==========
        alive_idx = np.where(self.prey_alive)[0]
        if len(alive_idx) >= 2:
            dos = degree_of_sparsity(self.prey_pos[alive_idx])
            doa = degree_of_alignment(self.prey_vel[alive_idx])
        else:
            dos, doa = 0.0, 0.0

        metrics = {"DoS": dos, "DoA": doa}
        done = False
        next_obs = self._get_predator_obs()
        return next_obs, rewards_pred, done, metrics
