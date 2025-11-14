import numpy as np

import numpy as np

def degree_of_sparsity(positions):
    """
    Degree of Sparsity (DoS) in [0, 1]

    1  → agents très espacés (sparse)
    0  → agents très compacts

    On prend :
      - d_nn_i : distance au plus proche voisin de l'agent i
      - d_mean : distance moyenne entre toutes les paires d'agents

    DoS = mean(d_nn_i / d_mean)   puis on clip dans [0, 1]
    """
    n = len(positions)
    if n < 2:
        return 0.0

    # distances complètes
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1
    )

    # distances entre paires distinctes
    mask = ~np.eye(n, dtype=bool)
    pairwise = distances[mask]
    d_mean = pairwise.mean() + 1e-8  # éviter la division par zéro

    # nearest neighbour pour chaque agent
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)

    # normalisation
    dos = np.mean(nearest / d_mean)

    # on force dans [0, 1]
    return float(np.clip(dos, 0.0, 1.0))



def degree_of_alignment(velocities):
    """
    Compute Degree of Alignment (DoA)
    → Mean cosine similarity of velocity directions between all agents.
    """
    n = len(velocities)
    if n < 2:
        return 0

    norms = np.linalg.norm(velocities, axis=1)
    valid = norms > 0
    if np.sum(valid) < 2:
        return 0

    v = velocities[valid] / norms[valid][:, None]
    dot_products = np.dot(v, v.T)
    doa = (np.sum(dot_products) - len(v)) / (len(v)*(len(v)-1))
    return doa