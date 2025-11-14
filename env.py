import numpy as np
from metrics import degree_of_sparsity, degree_of_alignment


class PredatorPreyEnv:
    def __init__(
        self,
        n_prey: int = 20,
        n_predators: int = 3,
        world_size: float = 10.0,
        dt: float = 0.1,
        prey_speed_limit: float = 0.3,
        pred_speed_limit: float = 0.35,
        friction: float = 0.4,
        catch_radius: float = 0.25,
        prey_noise_std: float = 0.0,
        r_rep: float = 0.4,
        r_align: float = 2.5,
        r_attr: float = 4.0,
        k_neighbors: int = 6,
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

        # rayon de perception "global" (utile pour la métrique, si besoin)
        self.perception_radius = world_size

        self.reset()

    # --------------------------------------------------------
    def reset(self):
        # positions proies & prédateurs aléatoires
        self.prey_pos = np.random.rand(self.n_prey, 2) * self.world_size
        self.pred_pos = np.random.rand(self.n_predators, 2) * self.world_size

        # vitesses initiales
        self.prey_vel = np.random.uniform(-0.5, 0.5, (self.n_prey, 2))
        self.pred_vel = np.random.uniform(-0.5, 0.5, (self.n_predators, 2))

        return self._get_predator_obs()

    # --------------------------------------------------------
    def _get_predator_obs(self) -> np.ndarray:
        """
        Observation pour chaque prédateur (dim 40) :

        [x, y, speed, heading]                   -> 4
        + 6 proies * (dx, dy, heading_j)        -> 18
        + 6 prédateurs * (dx, dy, heading_j)    -> 18
        = 40 dimensions par prédateur
        """
        obs_all = []
        for i in range(self.n_predators):
            # état propre
            px, py = self.pred_pos[i]
            vx, vy = self.pred_vel[i]
            speed = float(np.linalg.norm([vx, vy]))
            heading = float(np.arctan2(vy, vx))
            own_state = [px, py, speed, heading]

            # ----- voisins proies -----
            diffs_prey = self.prey_pos - self.pred_pos[i]
            diffs_prey = diffs_prey - np.round(diffs_prey / self.world_size) * self.world_size
            dists_prey = np.linalg.norm(diffs_prey, axis=1)

            idx_sorted = np.argsort(dists_prey)
            k = min(self.k_neighbors, len(idx_sorted))
            prey_feat = []
            for j in range(k):
                idx = idx_sorted[j]
                dx, dy = diffs_prey[idx]
                vxj, vyj = self.prey_vel[idx]
                heading_j = float(np.arctan2(vyj, vxj))
                prey_feat.extend([dx, dy, heading_j])

            while len(prey_feat) < 3 * self.k_neighbors:
                prey_feat.extend([0.0, 0.0, 0.0])

            # ----- voisins prédateurs -----
            diffs_pred = self.pred_pos - self.pred_pos[i]
            diffs_pred = diffs_pred - np.round(diffs_pred / self.world_size) * self.world_size
            dists_pred = np.linalg.norm(diffs_pred, axis=1)
            dists_pred[i] = np.inf

            idx_sorted = np.argsort(dists_pred)
            k = min(self.k_neighbors, len(idx_sorted))
            pred_feat = []
            for j in range(k):
                idx = idx_sorted[j]
                dx, dy = diffs_pred[idx]
                vxj, vyj = self.pred_vel[idx]
                heading_j = float(np.arctan2(vyj, vxj))
                pred_feat.extend([dx, dy, heading_j])

            while len(pred_feat) < 3 * self.k_neighbors:
                pred_feat.extend([0.0, 0.0, 0.0])

            obs_i = np.array(own_state + prey_feat + pred_feat, dtype=float)
            obs_all.append(obs_i)

        return np.vstack(obs_all)

    # --------------------------------------------------------
    def _couzin_forces(self) -> np.ndarray:
        """
        Forces sociales (répulsion / alignement / attraction) pour les proies.
        Retourne un array (n_prey, 2) de "pseudo-accélérations".
        """
        forces = np.zeros_like(self.prey_pos)

        for i in range(self.n_prey):
            diffs = self.prey_pos - self.prey_pos[i]
            diffs = diffs - np.round(diffs / self.world_size) * self.world_size
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf

            # répulsion
            mask_rep = dists < self.r_rep
            rep = np.zeros(2)
            if np.any(mask_rep):
                vecs = -diffs[mask_rep] / (dists[mask_rep][:, None] + 1e-8)
                rep = np.sum(vecs, axis=0)

            # alignement
            mask_align = (dists >= self.r_rep) & (dists < self.r_align)
            align = np.zeros(2)
            if np.any(mask_align):
                v_neighbors = self.prey_vel[mask_align]
                norms = np.linalg.norm(v_neighbors, axis=1, keepdims=True) + 1e-8
                align = np.sum(v_neighbors / norms, axis=0)

            # attraction
            mask_attr = (dists >= self.r_align) & (dists < self.r_attr)
            attr = np.zeros(2)
            if np.any(mask_attr):
                vecs = diffs[mask_attr] / (dists[mask_attr][:, None] + 1e-8)
                attr = np.sum(vecs, axis=0)

            force = 2.8 * rep + 2.0 * align + 1.5 * attr
            forces[i] = force

        # normalisation pour limiter l'accélération
        norms = np.linalg.norm(forces, axis=1, keepdims=True) + 1e-8
        forces = forces / np.maximum(norms, 1.0)
        return forces

    # --------------------------------------------------------
    def step(self, predator_actions):
        """
        predator_actions : np.array (n_predators, 2) -> vecteurs d'accélération (ax, ay)
        """
        predator_actions = np.asarray(predator_actions)

        # --- Proies : forces Couzin + bruit éventuel ---
        prey_social = self._couzin_forces()
        if self.prey_noise_std > 0.0:
            noise = self.prey_noise_std * np.random.randn(self.n_prey, 2)
        else:
            noise = 0.0

        prey_actions = prey_social + noise

        # cap sur l'accélération
        acc_norms = np.linalg.norm(prey_actions, axis=1, keepdims=True) + 1e-8
        prey_actions = prey_actions / np.maximum(acc_norms, 1.0)
        prey_actions *= 0.5

        # --- update vitesses ---
        self.pred_vel += self.dt * predator_actions
        self.prey_vel += self.dt * prey_actions

        self.pred_vel *= (1 - self.friction * self.dt)
        self.prey_vel *= (1 - self.friction * self.dt)

        # clamp vitesses
        pred_s = np.linalg.norm(self.pred_vel, axis=1, keepdims=True) + 1e-8
        prey_s = np.linalg.norm(self.prey_vel, axis=1, keepdims=True) + 1e-8

        self.pred_vel = np.where(
            pred_s > self.pred_speed_limit,
            self.pred_vel * (self.pred_speed_limit / pred_s),
            self.pred_vel
        )
        self.prey_vel = np.where(
            prey_s > self.prey_speed_limit,
            self.prey_vel * (self.prey_speed_limit / prey_s),
            self.prey_vel
        )

        # --- update positions (tore) ---
        self.pred_pos = (self.pred_pos + self.dt * self.pred_vel) % self.world_size
        self.prey_pos = (self.prey_pos + self.dt * self.prey_vel) % self.world_size

        # --- récompenses prédateurs ---
        rewards = np.zeros(self.n_predators)
        for i, pred in enumerate(self.pred_pos):
            dists = np.linalg.norm(self.prey_pos - pred, axis=1)
            rewards[i] += np.sum(dists < self.catch_radius)   # +1 par proie touchée
            rewards[i] -= 0.01 * np.linalg.norm(predator_actions[i])  # coût énergétique

        # --- metrics sur les proies ---
        dos = degree_of_sparsity(self.prey_pos)
        doa = degree_of_alignment(self.prey_vel)

        done = False  # pas de condition de fin interne, on coupe dans train.py

        return self._get_predator_obs(), rewards, done, {"DoS": dos, "DoA": doa}
