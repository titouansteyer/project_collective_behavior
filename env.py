import numpy as np


class PredatorPreyEnv:
    def __init__(
        self,
        n_prey=20,          # moins d'agents = entraînement plus léger
        n_predators=3,
        world_size=10.0,
        dt=0.1,             # pas de temps
        prey_speed_limit=1.0,
        pred_speed_limit=1.2,
        friction=0.2,
        catch_radius=0.2
    ):
        self.n_prey = n_prey
        self.n_predators = n_predators
        self.world_size = world_size
        self.dt = dt
        self.prey_speed_limit = prey_speed_limit
        self.pred_speed_limit = pred_speed_limit
        self.friction = friction
        self.catch_radius = catch_radius

        self.reset()

    def reset(self):
        # Positions aléatoires
        self.prey_pos = np.random.rand(self.n_prey, 2) * self.world_size
        self.pred_pos = np.random.rand(self.n_predators, 2) * self.world_size

        # Vitesses initiales plus petites pour éviter les sauts brutaux
        self.prey_vel = np.random.uniform(-0.5, 0.5, (self.n_prey, 2))
        self.pred_vel = np.random.uniform(-0.5, 0.5, (self.n_predators, 2))

        return self.get_obs()

    def get_obs(self):
        """
        Observation très simple :
        - pour les proies : [positions proies, positions prédateurs]
        - pour les prédateurs : [positions prédateurs, positions proies]

        On peut plus tard ajouter les vitesses dans l'obs si besoin.
        """
        obs_prey = np.concatenate([self.prey_pos.flatten(), self.pred_pos.flatten()])
        obs_pred = np.concatenate([self.pred_pos.flatten(), self.prey_pos.flatten()])
        return {"prey": obs_prey, "predators": obs_pred}

    def step(self, actions):
        """
        actions = dict {
            "prey": (n_prey, 2)  -> accélérations
            "predators": (n_predators, 2)
        }
        """
        prey_actions = actions["prey"]
        pred_actions = actions["predators"]

        # Mise à jour des vitesses avec friction
        self.prey_vel += self.dt * prey_actions
        self.pred_vel += self.dt * pred_actions

        # Friction (comme un drag très simple)
        self.prey_vel *= (1.0 - self.friction * self.dt)
        self.pred_vel *= (1.0 - self.friction * self.dt)

        # Clamp des vitesses
        prey_speed = np.linalg.norm(self.prey_vel, axis=1, keepdims=True) + 1e-8
        pred_speed = np.linalg.norm(self.pred_vel, axis=1, keepdims=True) + 1e-8

        self.prey_vel = np.where(
            prey_speed > self.prey_speed_limit,
            self.prey_vel * (self.prey_speed_limit / prey_speed),
            self.prey_vel
        )
        self.pred_vel = np.where(
            pred_speed > self.pred_speed_limit,
            self.pred_vel * (self.pred_speed_limit / pred_speed),
            self.pred_vel
        )

        # Mise à jour des positions
        self.prey_pos += self.dt * self.prey_vel
        self.pred_pos += self.dt * self.pred_vel

        # Bords périodiques (tore)
        self.prey_pos %= self.world_size
        self.pred_pos %= self.world_size

        # Récompenses
        rewards_pred = np.zeros(self.n_predators)
        rewards_prey = np.zeros(self.n_prey)
        done = False

        # Collisions prédateurs / proies
        for i, pred in enumerate(self.pred_pos):
            dists = np.linalg.norm(self.prey_pos - pred, axis=1)
            caught = np.where(dists < self.catch_radius)[0]
            if len(caught) > 0:
                # Chaque proie attrapée donne +1 au prédateur i
                rewards_pred[i] += float(len(caught))
                rewards_prey[caught] = -1.0

                # Respawn des proies attrapées
                self.prey_pos[caught] = np.random.rand(len(caught), 2) * self.world_size
                self.prey_vel[caught] = np.random.uniform(-0.5, 0.5, (len(caught), 2))

        # Petite pénalité d'énergie sur les actions pour éviter les gros mouvements
        rewards_prey -= 0.01 * np.linalg.norm(prey_actions, axis=1)
        rewards_pred -= 0.01 * np.linalg.norm(pred_actions, axis=1)

        info = {}
        return self.get_obs(), {"predators": rewards_pred, "prey": rewards_prey}, done, info
