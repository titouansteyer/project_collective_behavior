import numpy as np
from metrics import degree_of_sparsity, degree_of_alignment




def reflect_positions_and_velocities(pos, vel, L):
    """
    Perfect reflection in a square [0, L]x[0, L] using a 2L wrapping technique.
    Supports multiple bounces in one step.
    pos, vel: (N, 2) arrays.
    Returns: (pos_ref, vel_ref) reflecting any out-of-bound positions.
    """
    period = 2.0 * L
    tmp = np.mod(pos, period)  # positions in [0, 2L)
    pos_ref = np.where(tmp <= L, tmp, 2.0 * L - tmp)
    flip_x = (tmp[:, 0] > L).astype(np.float64)
    flip_y = (tmp[:, 1] > L).astype(np.float64)
    vel_ref = vel.copy()
    vel_ref[:, 0] = np.where(flip_x == 1.0, -vel_ref[:, 0], vel_ref[:, 0])
    vel_ref[:, 1] = np.where(flip_y == 1.0, -vel_ref[:, 1], vel_ref[:, 1])
    return pos_ref, vel_ref




def wall_force_hooke(pos, vel, L, k=50.0, c=2.0):
    """
    Physical wall force (Hooke's law + damping):
    If an agent penetrates the wall, apply a spring force proportional to penetration (stiffness k)
    and a damping force on the normal component (coefficient c).
    pos, vel: (N, 2) arrays for positions and velocities.
    Returns: fb (N, 2) array of bounce forces.
    """
    fb = np.zeros_like(pos)
    # left wall (x < 0)
    pen = -pos[:, 0]
    m = pen > 0
    if np.any(m):
        fb[m, 0] += k * pen[m]
        fb[m, 0] += -c * vel[m, 0]
    # right wall (x > L)
    pen = pos[:, 0] - L
    m = pen > 0
    if np.any(m):
        fb[m, 0] += -k * pen[m]
        fb[m, 0] += -c * vel[m, 0]
    # bottom wall (y < 0)
    pen = -pos[:, 1]
    m = pen > 0
    if np.any(m):
        fb[m, 1] += k * pen[m]
        fb[m, 1] += -c * vel[m, 1]
    # top wall (y > L)
    pen = pos[:, 1] - L
    m = pen > 0
    if np.any(m):
        fb[m, 1] += -k * pen[m]
        fb[m, 1] += -c * vel[m, 1]
    return fb

class PredatorPreyEnvReflect:
    """
    2D predator–prey environment with REFLECTIVE WALLS (bounded square).

    - Prey: follow Couzin-type rules (repulsion/alignment/attraction) + predator avoidance when not learning.
    - Predators: controlled by continuous acceleration actions.
    - Supports co-evolution: prey actions can be provided for learning prey.
    - Observations: similar 40-dim vectors as toroidal env (for predators by default, and for prey if learning).
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
        # Couzin radii:
        r_rep: float = 0.5,
        r_align: float = 2.0,
        r_attr: float = 6.0,
        k_neighbors: int = 6,
        # predator avoidance radius:
        predator_influence_radius: float = 2.0,
        # wall parameters:
        wall_stiffness: float = 50.0,
        wall_damping: float = 2.0,
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

        self.wall_stiffness = wall_stiffness
        self.wall_damping = wall_damping

        self.reset()






    def reset(self):
        """Random initialize positions & velocities within [0, L] square."""
        self.prey_pos = np.random.rand(self.n_prey, 2) * self.world_size
        self.pred_pos = np.random.rand(self.n_predators, 2) * self.world_size

        rand_dirs_prey = np.random.uniform(-1.0, 1.0, (self.n_prey, 2))
        norms = np.linalg.norm(rand_dirs_prey, axis=1, keepdims=True) + 1e-8
        self.prey_vel = self.prey_speed_limit * rand_dirs_prey / norms

        rand_dirs_pred = np.random.uniform(-1.0, 1.0, (self.n_predators, 2))
        norms_p = np.linalg.norm(rand_dirs_pred, axis=1, keepdims=True) + 1e-8
        self.pred_vel = self.pred_speed_limit * rand_dirs_pred / norms_p

        pred_obs = self._get_predator_obs()
        prey_obs = self._get_prey_obs()



        return (pred_obs, prey_obs)
    




    def _get_predator_obs(self) -> np.ndarray:
        """
        Predator observation (40 dims) – same structure as toroidal env (no torus wrapping here).
        """
        obs_all = []
        for i in range(self.n_predators):
            px, py = self.pred_pos[i]
            vx, vy = self.pred_vel[i]
            speed = float(np.linalg.norm([vx, vy]) + 1e-8)
            heading = float(np.arctan2(vy, vx + 1e-8))
            own_state = [px, py, speed, heading]
            # 6 nearest prey (Euclidean distance in box)
            diffs_prey = self.prey_pos - self.pred_pos[i]
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
            # 6 nearest predators (excluding self)
            diffs_pred = self.pred_pos - self.pred_pos[i]
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
            obs_all.append(np.array(own_state + prey_feat + pred_feat, dtype=float))
        return np.vstack(obs_all)




    def _get_prey_obs(self) -> np.ndarray:
        """
        Prey observation (40 dims) – similar structure for each prey.
        """
        obs_all = []
        for i in range(self.n_prey):
            px, py = self.prey_pos[i]
            vx, vy = self.prey_vel[i]
            speed = float(np.linalg.norm([vx, vy]) + 1e-8)
            heading = float(np.arctan2(vy, vx + 1e-8))
            own_state = [px, py, speed, heading]
            # 6 nearest other prey
            diffs_prey = self.prey_pos - self.prey_pos[i]
            dists_prey = np.linalg.norm(diffs_prey, axis=1)
            dists_prey[i] = np.inf
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
            # 6 nearest predators
            diffs_pred = self.pred_pos - self.prey_pos[i]
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
            obs_all.append(np.array(own_state + prey_feat + pred_feat, dtype=float))
        return np.vstack(obs_all)




    def _compute_prey_directions(self) -> np.ndarray:
        """
        Compute Couzin-rule directions for prey (no torus wrapping, since walls).
        """
        N = self.n_prey
        directions = np.zeros_like(self.prey_pos)
        for i in range(N):
            pos_i = self.prey_pos[i]
            vel_i = self.prey_vel[i]
            # distances to other prey (Euclidean)
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
            p_diffs = self.pred_pos - pos_i
            p_dists = np.linalg.norm(p_diffs, axis=1)
            mask_pred = p_dists < self.predator_influence_radius
            force_pred = np.zeros(2)
            if np.any(mask_pred):
                vecs = -p_diffs[mask_pred] / (p_dists[mask_pred][:, None] + 1e-8)
                force_pred = np.sum(vecs, axis=0)
            # combine forces
            force = (1.2 * force_rep 
                     + 5.0 * force_align 
                     + 4.0 * force_attr 
                     + 1.5 * force_pred)
            if np.linalg.norm(force) < 1e-6:
                force = vel_i
            directions[i] = force / (np.linalg.norm(force) + 1e-8)
        if self.prey_noise_std > 0.0:
            noise = self.prey_noise_std * np.random.randn(self.n_prey, 2)
            directions += noise
            norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
            directions = directions / norms
        return directions





    def step(self, predator_actions: np.ndarray, prey_actions: np.ndarray = None):
        """
        One simulation step with reflective wall boundaries.
        """
        predator_actions = np.asarray(predator_actions, dtype=float)
        if prey_actions is not None:
            prey_actions = np.asarray(prey_actions, dtype=float)

        # --- Prey update ---
        if prey_actions is None:
            # apply Couzin rules for prey movement
            new_dirs = self._compute_prey_directions()
            self.prey_vel = self.prey_speed_limit * new_dirs
        else:
            # prey RL control
            self.prey_vel += self.dt * prey_actions
            self.prey_vel *= (1.0 - self.friction * self.dt)
            prey_speed = np.linalg.norm(self.prey_vel, axis=1, keepdims=True) + 1e-8
            self.prey_vel = np.where(prey_speed > self.prey_speed_limit,
                                      self.prey_vel * (self.prey_speed_limit / prey_speed),
                                      self.prey_vel)
        # compute new prey position (raw) and apply wall bounce forces
        prey_pos_raw = self.prey_pos + self.dt * self.prey_vel
        fb_prey = wall_force_hooke(prey_pos_raw, self.prey_vel, self.world_size,
                                   k=self.wall_stiffness, c=self.wall_damping)
        # update prey velocity and position after wall interactions
        self.prey_vel = self.prey_vel + self.dt * fb_prey
        self.prey_pos = self.prey_pos + self.dt * self.prey_vel
        self.prey_pos, self.prey_vel = reflect_positions_and_velocities(
        self.prey_pos, self.prey_vel, self.world_size
    )


        # --- Predators update ---
        # acceleration & friction
        self.pred_vel += self.dt * predator_actions
        self.pred_vel *= (1.0 - self.friction * self.dt)
        # clamp predator speed
        pred_speed = np.linalg.norm(self.pred_vel, axis=1, keepdims=True) + 1e-8
        self.pred_vel = np.where(pred_speed > self.pred_speed_limit,
                                  self.pred_vel * (self.pred_speed_limit / pred_speed),
                                  self.pred_vel)
        # compute new predator position and apply wall forces
        pred_pos_raw = self.pred_pos + self.dt * self.pred_vel
        fb_pred = wall_force_hooke(pred_pos_raw, self.pred_vel, self.world_size,
                                   k=self.wall_stiffness, c=self.wall_damping)
        self.pred_vel = self.pred_vel + self.dt * fb_pred
        self.pred_pos = self.pred_pos + self.dt * self.pred_vel
        self.pred_pos, self.pred_vel = reflect_positions_and_velocities(
        self.pred_pos, self.pred_vel, self.world_size
    )


        # --- Rewards for predators and prey ---
        pred_rewards = np.zeros(self.n_predators)
        prey_rewards = np.zeros(self.n_prey)

        # predator-prey capture events
        dists = np.linalg.norm(self.pred_pos[:, None, :] - self.prey_pos[None, :, :], axis=2)
        contacts = (dists < self.catch_radius)

        pred_rewards += contacts.sum(axis=1)      # +1 per caught prey
        prey_rewards -= contacts.sum(axis=0)      # -1 if caught

        # ------------------------------------------------------------
        # Wall proximity penalty (prey): penalize being close to walls
        # This prevents "camping" near walls without actually touching them.
        if prey_actions is not None:
            d = np.minimum.reduce([
                self.prey_pos[:, 0],
                self.world_size - self.prey_pos[:, 0],
                self.prey_pos[:, 1],
                self.world_size - self.prey_pos[:, 1],
            ])

            d0 = 1.0
            wall_penalty = 0.2 * np.clip((d0 - d) / d0, 0.0, 1.0)
            prey_rewards -= wall_penalty


        # small action energy costs
        for i in range(self.n_predators):
            pred_rewards[i] -= 0.01 * np.linalg.norm(predator_actions[i])
        if prey_actions is not None:
            for j in range(self.n_prey):
                prey_rewards[j] -= 0.01 * np.linalg.norm(prey_actions[j])

        # apply wall-touch penalty: any agent hitting a wall gets -0.1
        if self.n_predators > 0:
            # check predator raw positions beyond boundaries
            out_left = pred_pos_raw[:, 0] < 0
            out_right = pred_pos_raw[:, 0] > self.world_size
            out_bottom = pred_pos_raw[:, 1] < 0
            out_top = pred_pos_raw[:, 1] > self.world_size
            pred_out = out_left | out_right | out_bottom | out_top
            for i in range(self.n_predators):
                if pred_out[i]:
                    pred_rewards[i] -= 0.1
        if prey_actions is not None and self.n_prey > 0:
            # check prey raw positions beyond boundaries
            out_left = prey_pos_raw[:, 0] < 0
            out_right = prey_pos_raw[:, 0] > self.world_size
            out_bottom = prey_pos_raw[:, 1] < 0
            out_top = prey_pos_raw[:, 1] > self.world_size
            prey_out = out_left | out_right | out_bottom | out_top
            for j in range(self.n_prey):
                if prey_out[j]:
                    prey_rewards[j] -= 0.1

        
        # collective metrics
        dos = degree_of_sparsity(self.prey_pos, world_size=self.world_size)
        doa = degree_of_alignment(self.prey_vel)
        info = {"DoS": dos, "DoA": doa}
        done = False

        # observations for next state
        pred_obs = self._get_predator_obs()
        prey_obs = self._get_prey_obs()
        if prey_actions is None:
            return pred_obs, pred_rewards, done, info
        else:
            return (pred_obs, prey_obs), (pred_rewards, prey_rewards), done, info
