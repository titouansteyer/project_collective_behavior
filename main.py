from env import PredatorPreyEnv
import numpy as np

env = PredatorPreyEnv()
for t in range(10):
    predator_actions = np.random.uniform(-1, 1, (env.n_predators, 2))
    prey_actions = np.random.uniform(-1, 1, (env.n_prey, 2))
    (prey, predators), (r_pred, r_prey) = env.step(predator_actions, prey_actions)
    print(f"Step {t}: predator reward {r_pred.sum():.2f}, prey reward {r_prey.sum():.2f}")
