import numpy as np

class Agent:
    def __init__(self, pos, vel, is_predator=False):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.is_predator = is_predator
        self.alive = True

class PredatorPreyEnv:
    def __init__(self, n_prey=30, n_predators=3, world_size=10):
        self.world_size = world_size
        self.n_prey = n_prey
        self.n_predators = n_predators
        self.reset()

    def reset(self):
        self.prey = [Agent(np.random.rand(2)*self.world_size, np.zeros(2)) for _ in range(self.n_prey)]
        self.predators = [Agent(np.random.rand(2)*self.world_size, np.zeros(2), True) for _ in range(self.n_predators)]
        return self.get_state()

    def get_state(self):
        prey_pos = np.array([a.pos for a in self.prey])
        pred_pos = np.array([a.pos for a in self.predators])
        return prey_pos, pred_pos

    def step(self, predator_actions, prey_actions, dt=0.1):
        # actions = velocity vectors (normalized)
        for a, action in zip(self.predators, predator_actions):
            a.pos += np.clip(action, -1, 1) * dt * 5
        for a, action in zip(self.prey, prey_actions):
            a.pos += np.clip(action, -1, 1) * dt * 4

        # boundary conditions (periodic)
        for a in self.predators + self.prey:
            a.pos = np.mod(a.pos, self.world_size)

        # check catches
        rewards_pred = np.zeros(self.n_predators)
        rewards_prey = np.zeros(self.n_prey)
        for i, p in enumerate(self.predators):
            for j, q in enumerate(self.prey):
                if np.linalg.norm(p.pos - q.pos) < 0.3:
                    q.alive = False
                    rewards_pred[i] += 1
                    rewards_prey[j] -= 1

        return self.get_state(), (rewards_pred, rewards_prey)
