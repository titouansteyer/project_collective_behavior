import numpy as np
from metrics import degree_of_sparsity, degree_of_alignment

def reflect_positions_and_velocities(pos, vel, L):
    """
    Réflexion propre dans le carré [0,L]x[0,L] via la technique du repli 2L.
    Supporte plusieurs rebonds en un seul grand pas (dt large).
    pos, vel: (N, 2)
    """
    period = 2.0 * L
    tmp = np.mod(pos, period)  # dans [0, 2L)
    pos_ref = np.where(tmp <= L, tmp, 2.0 * L - tmp)

    flip_x = (tmp[:, 0] > L).astype(np.float64)
    flip_y = (tmp[:, 1] > L).astype(np.float64)

    vel_ref = vel.copy()
    vel_ref[:, 0] = np.where(flip_x == 1.0, -vel_ref[:, 0], vel_ref[:, 0])
    vel_ref[:, 1] = np.where(flip_y == 1.0, -vel_ref[:, 1], vel_ref[:, 1])

    return pos_ref, vel_ref


class PredatorPreyEnvReflect:
    """
    2D predator–prey avec MURS RÉFLÉCHISSANTS (pas de tore).

    - Proies: règles type Couzin (répulsion / alignement / attraction) + évitement des prédateurs.
    - Prédateurs: contrôlés par actions continues (ax, ay).
    - Observations: uniquement pour les prédateurs (un vecteur par prédateur).
    - API compatible avec ton PredatorPreyEnv d'origine (state_dim=40).
    """

    def __init__(
        self,
        n_prey: int = 20,
        n_predators: int = 3,
        world_size: float = 10.0,
        dt: float = 0.1,
        prey_speed_limit: float = 0.6,
        pred_speed_limit: float = 0.8,
        friction: float = 0.4,
        catch_radius: float = 0.35,
        prey_noise_std: float = 0.02,
        # Couzin interaction radii
        r_rep: float = 0.5,
        r_align: float = 2.0,
        r_attr: float = 6.0,
        k_neighbors: int = 6,
        # predator avoidance
        predator_influence_radius: float = 2.0,
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

        self.r_rep = r_rep
        self.r_align = r_align
        self.r_attr = r_attr
        self.k_neighbors = k_neighbors
        self.predator_influence_radius = predator_influence_radius

        # perception globale (inchangé)
        self.perception_radius = world_size

        self.reset()

    # --------------------------------------------------------
    def reset(self):
        """Random init positions & velocities (dans [0, L])."""
        self.prey_pos = np.random.rand(self.n_prey, 2) * self.world_size
        self.pred_pos = np.random.rand(self.n_predators, 2) * self.world_size

        rand_dirs_prey = np.random.uniform(-1.0, 1.0, (self.n_prey, 2))
        norms = np.linalg.norm(rand_dirs_prey, axis=1, keepdims=True) + 1e-8
        self.prey_vel = self.prey_speed_limit * rand_dirs_prey / norms

        rand_dirs_pred = np.random.uniform(-1.0, 1.0, (self.n_predators, 2))
        norms_p = np.linalg.norm(rand_dirs_pred, axis=1, keepdims=True) + 1e-8
        self.pred_vel = self.pred_speed_limit * rand_dirs_pred / norms_p

        return self._get_predator_obs()

    # --------------------------------------------------------
    def _get_predator_obs(self) -> np.ndarray:
        """
        Observation par prédateur (40 dims) — mêmes features que l’original.
        ATTENTION: plus de correction torique des différences.
        """
        obs_all = []

        for i in range(self.n_predators):
            px, py = self.pred_pos[i]
            vx, vy = self.pred_vel[i]
            speed = float(np.linalg.norm([vx, vy]) + 1e-8)
            heading = float(np.arctan2(vy, vx + 1e-8))
            own_state = [px, py, speed, heading]

            # --- nearest prey ---
            diffs_prey = self.prey_pos - self.pred_pos[i]  # pas de tore
            dists_prey = np.linalg.norm(diffs_prey, axis=1)

            idx_sorted = np.argsort(dists_prey)
            k = min(self.k_neighbors, len(idx_sorted))
            prey_feat = []
            for j in range(k):
                idx = idx_sorted[j]
                dx, dy = diffs_prey[idx]
                vxj, vyj = self.prey_vel[idx]
                heading_j = float(np.arctan2(vyj, vxj + 1e-8))
                prey_feat.extend([dx, dy, heading_j])

            while len(prey_feat) < 3 * self.k_neighbors:
                prey_feat.extend([0.0, 0.0, 0.0])

            # --- nearest predators (excluding self) ---
            diffs_pred = self.pred_pos - self.pred_pos[i]  # pas de tore
            dists_pred = np.linalg.norm(diffs_pred, axis=1)
            dists_pred[i] = np.inf

            idx_sorted = np.argsort(dists_pred)
            k = min(self.k_neighbors, len(idx_sorted))
            pred_feat = []
            for j in range(k):
                idx = idx_sorted[j]
                dx, dy = diffs_pred[idx]
                vxj, vyj = self.pred_vel[idx]
                heading_j = float(np.arctan2(vyj, vxj + 1e-8))
                pred_feat.extend([dx, dy, heading_j])

            while len(pred_feat) < 3 * self.k_neighbors:
                pred_feat.extend([0.0, 0.0, 0.0])

            obs_i = np.array(own_state + prey_feat + pred_feat, dtype=float)
            obs_all.append(obs_i)

        return np.vstack(obs_all)

    # --------------------------------------------------------
    def _compute_prey_directions(self) -> np.ndarray:
        """
        Règles Couzin + avoidance sans géométrie torique.
        """
        N = self.n_prey
        directions = np.zeros_like(self.prey_pos)

        for i in range(N):
            pos_i = self.prey_pos[i]
            vel_i = self.prey_vel[i]

            # relative positions to other prey (EUCLIDIEN simple)
            diffs = self.prey_pos - pos_i
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf

            # 1) Repulsion
            mask_rep = dists < self.r_rep
            force_rep = np.zeros(2)
            if np.any(mask_rep):
                vecs = -diffs[mask_rep] / (dists[mask_rep][:, None] + 1e-8)
                force_rep = np.sum(vecs, axis=0)

            # 2) Alignment
            mask_align = (dists >= self.r_rep) & (dists < self.r_align)
            force_align = np.zeros(2)
            if np.any(mask_align):
                v_neighbors = self.prey_vel[mask_align]
                norms = np.linalg.norm(v_neighbors, axis=1, keepdims=True) + 1e-8
                force_align = np.sum(v_neighbors / norms, axis=0)

            # 3) Attraction
            mask_attr = (dists >= self.r_align) & (dists < self.r_attr)
            force_attr = np.zeros(2)
            if np.any(mask_attr):
                vecs = diffs[mask_attr] / (dists[mask_attr][:, None] + 1e-8)
                force_attr = np.sum(vecs, axis=0)

            # 4) Predator avoidance
            p_diffs = self.pred_pos - pos_i  # pas de tore
            p_dists = np.linalg.norm(p_diffs, axis=1)
            mask_pred = p_dists < self.predator_influence_radius
            force_pred = np.zeros(2)
            if np.any(mask_pred):
                vecs = -p_diffs[mask_pred] / (p_dists[mask_pred][:, None] + 1e-8)
                force_pred = np.sum(vecs, axis=0)

            # Combinaison
            force = (
                1.2 * force_rep +
                5.0 * force_align +
                4.0 * force_attr +
                1.5 * force_pred
            )

            if np.linalg.norm(force) < 1e-6:
                force = vel_i

            dir_i = force / (np.linalg.norm(force) + 1e-8)
            directions[i] = dir_i

        # Noise optionnelle
        if self.prey_noise_std > 0.0:
            noise = self.prey_noise_std * np.random.randn(self.n_prey, 2)
            directions = directions + noise
            norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
            directions = directions / norms

        return directions

    # --------------------------------------------------------
    def step(self, predator_actions: np.ndarray):
        """
        Un pas de simulation avec MURS RÉFLÉCHISSANTS.
        """
        predator_actions = np.asarray(predator_actions, dtype=float)

        # --- Proies ---
        new_dirs = self._compute_prey_directions()
        self.prey_vel = self.prey_speed_limit * new_dirs
        prey_pos_raw = self.prey_pos + self.dt * self.prey_vel
        self.prey_pos, self.prey_vel = reflect_positions_and_velocities(
            prey_pos_raw, self.prey_vel, self.world_size
        )

        # --- Prédateurs ---
        self.pred_vel += self.dt * predator_actions
        self.pred_vel *= (1.0 - self.friction * self.dt)

        pred_s = np.linalg.norm(self.pred_vel, axis=1, keepdims=True) + 1e-8
        self.pred_vel = np.where(
            pred_s > self.pred_speed_limit,
            self.pred_vel * (self.pred_speed_limit / pred_s),
            self.pred_vel,
        )

        pred_pos_raw = self.pred_pos + self.dt * self.pred_vel
        self.pred_pos, self.pred_vel = reflect_positions_and_velocities(
            pred_pos_raw, self.pred_vel, self.world_size
        )

        # --- Rewards + respawn proies attrapées ---
        rewards = np.zeros(self.n_predators)
        for i, pred in enumerate(self.pred_pos):
            dists = np.linalg.norm(self.prey_pos - pred, axis=1)
            caught_mask = dists < self.catch_radius
            n_caught = int(np.sum(caught_mask))

            rewards[i] += n_caught
            rewards[i] -= 0.01 * np.linalg.norm(predator_actions[i])

            if n_caught > 0:
                caught_idx = np.where(caught_mask)[0]
                self.prey_pos[caught_idx] = np.random.rand(len(caught_idx), 2) * self.world_size
                rand_dirs = np.random.uniform(-1.0, 1.0, (len(caught_idx), 2))
                norms = np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-8
                self.prey_vel[caught_idx] = self.prey_speed_limit * rand_dirs / norms

        dos = degree_of_sparsity(self.prey_pos)
        doa = degree_of_alignment(self.prey_vel)

        done = False
        obs = self._get_predator_obs()
        info = {"DoS": dos, "DoA": doa}
        return obs, rewards, done, info
