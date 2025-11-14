import numpy as np
from metrics import degree_of_sparsity, degree_of_alignment


class PredatorPreyEnv:
    def __init__(
        self,
        n_prey=20,
        n_predators=3,
        world_size=10.0,
        dt=0.1,
        prey_speed_limit=0.3,
        pred_speed_limit=0.35,
        friction=0.4,
        catch_radius=0.25,
        prey_noise_std=0.0,      # on peut le laisser pour un peu de bruit
        r_rep=0.4,
        r_align=2.5,
        r_attr=4.0
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
    def _couzin_forces(self):
        """
        Calcule les "forces sociales" sur chaque proie :
        répulsion / alignement / attraction (modèle type Couzin).
        Retourne un array (n_prey, 2) = pseudo-accélérations.
        """
        forces = np.zeros_like(self.prey_pos)

        for i in range(self.n_prey):
            # vecteurs vers les autres proies
            diffs = self.prey_pos - self.prey_pos[i]

            # distances en tore (distance minimale modulo world_size)
            diffs = diffs - np.round(diffs / self.world_size) * self.world_size
          # (n_prey, 2)
            dists = np.linalg.norm(diffs, axis=1)
            # on ignore soi-même
            dists[i] = np.inf

            # --- répulsion ---
            mask_rep = dists < self.r_rep
            rep = np.zeros(2)
            if np.any(mask_rep):
                # s'éloigner des trop proches
                vecs = -diffs[mask_rep] / (dists[mask_rep][:, None] + 1e-8)
                rep = np.sum(vecs, axis=0)

            # --- alignement ---
            mask_align = (dists >= self.r_rep) & (dists < self.r_align)
            align = np.zeros(2)
            if np.any(mask_align):
                v_neighbors = self.prey_vel[mask_align]
                norms = np.linalg.norm(v_neighbors, axis=1, keepdims=True) + 1e-8
                align = np.sum(v_neighbors / norms, axis=0)

            # --- attraction ---
            mask_attr = (dists >= self.r_align) & (dists < self.r_attr)
            attr = np.zeros(2)
            if np.any(mask_attr):
                vecs = diffs[mask_attr] / (dists[mask_attr][:, None] + 1e-8)
                attr = np.sum(vecs, axis=0)

            # pondérations simples (à ajuster si besoin)
            force = 2.8 * rep + 2.0 * align + 1.5 * attr

            forces[i] = force

        # on normalise un peu pour éviter des accélérations gigantesques
        norms = np.linalg.norm(forces, axis=1, keepdims=True) + 1e-8
        forces = forces / np.maximum(norms, 1.0)  # cap à 1

        return forces


    # --------------------------------------------------------
    def step(self, predator_actions):

        predator_actions = np.asarray(predator_actions)

        # forces de type Couzin pour les proies
        prey_social = self._couzin_forces()

        # éventuellement un léger bruit pour casser la symétrie
        if self.prey_noise_std > 0.0:
            noise = self.prey_noise_std * np.random.randn(self.n_prey, 2)
        else:
            noise = 0.0

        prey_actions = prey_social + noise

        # Cap sur l’accélération (pour éviter de sprinter partout)
        acc_norms = np.linalg.norm(prey_actions, axis=1, keepdims=True) + 1e-8
        prey_actions = prey_actions / np.maximum(acc_norms, 1.0)
        prey_actions *= 0.5  # scale global


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
