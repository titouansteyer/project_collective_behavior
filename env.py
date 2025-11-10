import numpy as np

class PredatorPreyEnv:
    def __init__(self, n_prey=30, n_predators=3, world_size=7, speed_prey=0.02, speed_pred=0.05):
        self.n_prey = n_prey
        self.n_predators = n_predators
        self.world_size = world_size
        self.speed_prey = speed_prey
        self.speed_pred = speed_pred

        # initialize positions and velocities
        self.prey_positions = np.random.rand(n_prey, 2) * world_size
        self.predator_positions = np.random.rand(n_predators, 2) * world_size
        self.prey_velocities = np.random.uniform(-1, 1, (n_prey, 2))
        self.predator_velocities = np.random.uniform(-1, 1, (n_predators, 2))

    def reset(self):
        self.prey_positions = np.random.rand(self.n_prey, 2) * self.world_size
        self.predator_positions = np.random.rand(self.n_predators, 2) * self.world_size
        return self.prey_positions, self.predator_positions

    # --- COUZIN MODEL FOR PREY ---
    def couzin_update(self, positions, velocities, radius_rep=0.3, radius_align=0.8, radius_attr=2.0,
        w_rep=2.0, w_align=1.0, w_attr=0.8):

        new_velocities = np.zeros_like(velocities)

        for i, pos in enumerate(positions):
            dists = np.linalg.norm(positions - pos, axis=1)
            neighbors = (dists > 0) & (dists < radius_attr)

            # Si aucun voisin, continue aléatoirement
            if not np.any(neighbors):
                rnd = np.random.uniform(-1, 1, 2)
                new_velocities[i] = rnd / (np.linalg.norm(rnd) + 1e-8)
                continue

            force = np.zeros(2)
        # Répulsion (forte)
            close = dists < radius_rep
            if np.any(close):
                force += w_rep * np.sum(pos - positions[close], axis=0)

        # Alignement
            align = (dists >= radius_rep) & (dists < radius_align)
            if np.any(align):
                avg_dir = np.sum(velocities[align], axis=0)
                force += w_align * avg_dir

        # Attraction
            far = (dists >= radius_align) & (dists < radius_attr)
            if np.any(far):
                force += w_attr * np.sum(positions[far] - pos, axis=0)

        # Normalisation
            norm = np.linalg.norm(force)
            if norm > 0:
                force /= norm
            else:
                force = velocities[i] / (np.linalg.norm(velocities[i]) + 1e-8)

            new_velocities[i] = force

        return new_velocities



    # --- COOPERATIVE STRATEGY FOR PREDATORS ---
    def predator_strategy(self, predator_positions, prey_positions):
        new_vel = np.zeros_like(predator_positions)
        for i, pred in enumerate(predator_positions):
            # Each predator targets the closest prey
            target = prey_positions[np.argmin(np.linalg.norm(prey_positions - pred, axis=1))]
            direction = target - pred
            new_vel[i] = direction / (np.linalg.norm(direction) + 1e-8)
        return new_vel

    # --- STEP FUNCTION ---
    def step(self):
        # Update prey (Couzin rules)
        self.prey_velocities = self.couzin_update(self.prey_positions, self.prey_velocities)
        self.prey_positions += self.speed_prey * self.prey_velocities

        # Update predators (deterministic cooperative pursuit)
        self.predator_velocities = self.predator_strategy(self.predator_positions, self.prey_positions)
        self.predator_positions += self.speed_pred * self.predator_velocities

        # Apply periodic boundary conditions
        self.prey_positions %= self.world_size
        self.predator_positions %= self.world_size

        # Compute rewards (predation events)
        rewards_pred = np.zeros(self.n_predators)
        rewards_prey = np.zeros(self.n_prey)

        for i, pred in enumerate(self.predator_positions):
            distances = np.linalg.norm(self.prey_positions - pred, axis=1)
            caught = np.where(distances < 0.2)[0]
            if len(caught) > 0:
                rewards_pred[i] += len(caught)
                rewards_prey[caught] = -1
                # respawn caught prey at random positions
                self.prey_positions[caught] = np.random.rand(len(caught), 2) * self.world_size

        return (self.prey_positions, self.predator_positions), (rewards_pred, rewards_prey)
