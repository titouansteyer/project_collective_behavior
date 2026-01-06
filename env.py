import numpy as np
from metrics import degree_of_sparsity, degree_of_alignment

class PredatorPreyEnv:
    """
    2D toroidal predator–prey environment.

    - Prey follow Couzin-like flocking rules (repulsion / alignment / attraction)
      + avoidance of nearby predators.
    - Predators are controlled via continuous acceleration actions (ax, ay).
    - If prey are learning (co-evolution), they are also controlled via actions; 
      otherwise they move with fixed flocking rules.
    - Observations: by default returned for predators (each predator gets a 40-dim vector). 
      In co-evolution mode, observations for prey can also be retrieved.
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
        # Couzin interaction radii:
        r_rep: float = 0.5,
        r_align: float = 2.0,
        r_attr: float = 6.0,
        k_neighbors: int = 6,
        # predator avoidance radius for prey:
        predator_influence_radius: float = 2.0,
        prey_mode: str = "couzin",  # "couzin" or "rl"
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

        # (unused) global perception radius set equal to world_size
        self.perception_radius = world_size

        assert prey_mode in ["couzin", "rl"], "prey_mode must be 'couzin' or 'rl'"
        self.prey_mode = prey_mode

        self.reset()

    def reset(self):
        """Randomly initialize positions and velocities of all agents."""
        # positions (uniform in [0, world_size])
        self.prey_pos = np.random.rand(self.n_prey, 2) * self.world_size
        self.pred_pos = np.random.rand(self.n_predators, 2) * self.world_size

        # initial velocities: random directions at max speed
        rand_dirs_prey = np.random.uniform(-1.0, 1.0, (self.n_prey, 2))
        norms = np.linalg.norm(rand_dirs_prey, axis=1, keepdims=True) + 1e-8
        self.prey_vel = self.prey_speed_limit * rand_dirs_prey / norms

        rand_dirs_pred = np.random.uniform(-1.0, 1.0, (self.n_predators, 2))
        norms_p = np.linalg.norm(rand_dirs_pred, axis=1, keepdims=True) + 1e-8
        self.pred_vel = self.pred_speed_limit * rand_dirs_pred / norms_p

        # Return initial observations (predators and prey)
        pred_obs = self._get_predator_obs()
        prey_obs = self._get_prey_obs()
        return (pred_obs, prey_obs)

    def _get_predator_obs(self) -> np.ndarray:
        """
        Observation for each predator (shape: n_predators x 40).
        For each predator i, features are:
        - [x_i, y_i, speed_i, heading_i]                  (4 values)
        - 6 closest prey: (dx, dy, heading_j) * 6         (18 values)
        - 6 closest other predators: (dx, dy, heading_k) * 6  (18 values)
        = 40-dimensional observation.
        All position differences are computed on the torus (wrap-around distances).
        """
        obs_all = []
        for i in range(self.n_predators):
            px, py = self.pred_pos[i]
            vx, vy = self.pred_vel[i]
            speed = float(np.linalg.norm([vx, vy]) + 1e-8)
            heading = float(np.arctan2(vy, vx + 1e-8))
            own_state = [px, py, speed, heading]

            # nearest prey (distance on torus)
            diffs_prey = self.prey_pos - self.pred_pos[i]
            diffs_prey -= np.round(diffs_prey / self.world_size) * self.world_size
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
            # zero-pad if fewer than k_neighbors prey
            while len(prey_feat) < 3 * self.k_neighbors:
                prey_feat.extend([0.0, 0.0, 0.0])

            # nearest predators (excluding self)
            diffs_pred = self.pred_pos - self.pred_pos[i]
            diffs_pred -= np.round(diffs_pred / self.world_size) * self.world_size
            dists_pred = np.linalg.norm(diffs_pred, axis=1)
            dists_pred[i] = np.inf  # ignore self
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

    def _get_prey_obs(self) -> np.ndarray:
        """
        Observation for each prey (shape: n_prey x 40).
        For each prey i:
        - [x_i, y_i, speed_i, heading_i]                    (4)
        - 6 closest other prey: (dx, dy, heading_j) * 6     (18)
        - 6 closest predators: (dx, dy, heading_k) * 6      (18)
        = 40-dimensional observation.
        (All distances computed on torus, similar to predator observations.)
        """
        obs_all = []
        for i in range(self.n_prey):
            px, py = self.prey_pos[i]
            vx, vy = self.prey_vel[i]
            speed = float(np.linalg.norm([vx, vy]) + 1e-8)
            heading = float(np.arctan2(vy, vx + 1e-8))
            own_state = [px, py, speed, heading]

            # nearest other prey
            diffs_prey = self.prey_pos - self.prey_pos[i]
            diffs_prey -= np.round(diffs_prey / self.world_size) * self.world_size
            dists_prey = np.linalg.norm(diffs_prey, axis=1)
            dists_prey[i] = np.inf  # exclude itself
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

            # nearest predators
            diffs_pred = self.pred_pos - self.prey_pos[i]
            diffs_pred -= np.round(diffs_pred / self.world_size) * self.world_size
            dists_pred = np.linalg.norm(diffs_pred, axis=1)
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

    def _compute_prey_directions(self) -> np.ndarray:
        """
        Compute new heading unit vectors for each prey based on Couzin rules + predator avoidance.
        (Used when prey are not learning, i.e., following fixed swarm behavior.)
        """
        N = self.n_prey
        directions = np.zeros_like(self.prey_pos)
        for i in range(N):
            pos_i = self.prey_pos[i]
            vel_i = self.prey_vel[i]
            # relative positions to all other prey (on torus)
            diffs = self.prey_pos - pos_i
            diffs -= np.round(diffs / self.world_size) * self.world_size
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf  # ignore self

            # 1) Repulsion zone
            mask_rep = dists < self.r_rep
            force_rep = np.zeros(2)
            if np.any(mask_rep):
                vecs = -diffs[mask_rep] / (dists[mask_rep][:, None] + 1e-8)
                force_rep = np.sum(vecs, axis=0)

            # 2) Alignment zone
            mask_align = (dists >= self.r_rep) & (dists < self.r_align)
            force_align = np.zeros(2)
            if np.any(mask_align):
                v_neighbors = self.prey_vel[mask_align]
                norms = np.linalg.norm(v_neighbors, axis=1, keepdims=True) + 1e-8
                force_align = np.sum(v_neighbors / norms, axis=0)

            # 3) Attraction zone
            mask_attr = (dists >= self.r_align) & (dists < self.r_attr)
            force_attr = np.zeros(2)
            if np.any(mask_attr):
                vecs = diffs[mask_attr] / (dists[mask_attr][:, None] + 1e-8)
                force_attr = np.sum(vecs, axis=0)

            # 4) Predator avoidance zone
            p_diffs = self.pred_pos - pos_i
            p_diffs -= np.round(p_diffs / self.world_size) * self.world_size
            p_dists = np.linalg.norm(p_diffs, axis=1)
            mask_pred = p_dists < self.predator_influence_radius
            force_pred = np.zeros(2)
            if np.any(mask_pred):
                vecs = -p_diffs[mask_pred] / (p_dists[mask_pred][:, None] + 1e-8)
                force_pred = np.sum(vecs, axis=0)

            # weighted sum of forces
            force = (1.2 * force_rep    # avoid collisions
                     + 5.0 * force_align   # alignment (smooth flocking)
                     + 4.0 * force_attr    # attraction (cohesion)
                     + 1.5 * force_pred)   # flee predators

            if np.linalg.norm(force) < 1e-6:
                force = vel_i  # if no social forces, keep current direction
            dir_i = force / (np.linalg.norm(force) + 1e-8)
            directions[i] = dir_i

        # add small random noise to directions (if configured)
        if self.prey_noise_std > 0.0:
            noise = self.prey_noise_std * np.random.randn(self.n_prey, 2)
            directions += noise
            norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
            directions = directions / norms
        return directions

    def step(self, predator_actions: np.ndarray, prey_actions: np.ndarray = None):
        """
        Run one simulation timestep.

        Args:
            predator_actions (np.ndarray): shape (n_predators, 2) acceleration actions for predators.
            prey_actions (np.ndarray, optional): shape (n_prey, 2) actions for prey (if learning). 
                                                 If None, prey move according to Couzin rules.
        Returns:
            obs: If prey_actions is None, returns predator observations (n_predators x obs_dim). 
                 If prey_actions provided (co-evolution), returns tuple (pred_obs, prey_obs).
            rewards: If prey_actions is None, returns predator rewards array (n_predators,). 
                     If co-evolution, returns tuple (pred_rewards, prey_rewards).
            done: bool (always False here, episodes are time-limited externally).
            info: dict with collective metrics {"DoS": float, "DoA": float} for prey.
        """
        predator_actions = np.asarray(predator_actions, dtype=float)
        if prey_actions is not None:
            prey_actions = np.asarray(prey_actions, dtype=float)

         #--- Prey update ---
        if self.prey_mode == "couzin":
            new_dirs = self._compute_prey_directions()
            self.prey_vel = self.prey_speed_limit * new_dirs

        elif self.prey_mode == "rl":
            if prey_actions is None:
                raise RuntimeError("prey_mode='rl' mais prey_actions=None")
            self.prey_vel += self.dt * prey_actions
            self.prey_vel *= (1.0 - self.friction * self.dt)
            prey_speed = np.linalg.norm(self.prey_vel, axis=1, keepdims=True) + 1e-8
            self.prey_vel = np.where(
                prey_speed > self.prey_speed_limit,
                self.prey_vel * (self.prey_speed_limit / prey_speed),
                self.prey_vel
            )

        # move prey (wrap around world edges)
        self.prey_pos = (self.prey_pos + self.dt * self.prey_vel) % self.world_size

        # --- Predator update ---
        self.pred_vel += self.dt * predator_actions
        self.pred_vel *= (1.0 - self.friction * self.dt)
        pred_speed = np.linalg.norm(self.pred_vel, axis=1, keepdims=True) + 1e-8
        # clamp predator speed to max limit
        self.pred_vel = np.where(pred_speed > self.pred_speed_limit,
                                  self.pred_vel * (self.pred_speed_limit / pred_speed),
                                  self.pred_vel)
        self.pred_pos = (self.pred_pos + self.dt * self.pred_vel) % self.world_size

        # --- Rewards for predators and prey ---
        pred_rewards = np.zeros(self.n_predators)
        prey_rewards = np.zeros(self.n_prey)
        # check predator-prey contacts for capture
        dists = np.linalg.norm(self.pred_pos[:, None, :] - self.prey_pos[None, :, :], axis=2)
        contacts = (dists < self.catch_radius)
        pred_rewards += contacts.sum(axis=1)        # each predator gets +1 per caught prey
        prey_rewards -= contacts.sum(axis=0)        # each prey gets -1 if caught

        # small energy cost for actions (to discourage excessive acceleration)
        for i in range(self.n_predators):
            pred_rewards[i] -= 0.01 * np.linalg.norm(predator_actions[i])
        if prey_actions is not None:
            for j in range(self.n_prey):
                prey_rewards[j] -= 0.01 * np.linalg.norm(prey_actions[j])

        # If prey not learning, respawn caught prey elsewhere (to keep prey count constant)
        if prey_actions is None:
            caught_any = contacts.sum(axis=0) > 0
            if np.any(caught_any):
                caught_idx = np.where(caught_any)[0]
                # respawn these prey at random new positions and headings
                self.prey_pos[caught_idx] = np.random.rand(len(caught_idx), 2) * self.world_size
                rand_dirs = np.random.uniform(-1.0, 1.0, (len(caught_idx), 2))
                norms = np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-8
                self.prey_vel[caught_idx] = self.prey_speed_limit * rand_dirs / norms

        # --- Compute collective metrics ---
        dos = degree_of_sparsity(self.prey_pos, world_size=self.world_size)
        doa = degree_of_alignment(self.prey_vel)
        info = {"DoS": dos, "DoA": doa}
        done = False

        # observations for next state
        pred_obs = self._get_predator_obs()
        prey_obs = self._get_prey_obs()

        if self.prey_mode == "couzin":
            return pred_obs, pred_rewards, done, info
        else:
            return (pred_obs, prey_obs), (pred_rewards, prey_rewards), done, info
