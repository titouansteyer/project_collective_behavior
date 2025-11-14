import numpy as np


def degree_of_sparsity(positions: np.ndarray) -> float:
    """
    Degree of Sparsity (DoS) ∈ [0, 1]

    Ici :
      - 1  → groupe très compact (agents proches)
      - 0  → groupe très dispersé

    On utilise :
      d_mean = distance moyenne entre toutes les paires d'agents
      d_nn_i = distance au plus proche voisin de l'agent i

    On définit:
      s = mean_i(d_nn_i / d_mean)
      DoS = 1 - clip(s, 0, 1)
    """
    positions = np.asarray(positions)
    n = len(positions)
    if n < 2:
        return 0.0

    # matrice des distances
    dists = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1
    )

    # distances entre paires distinctes
    mask = ~np.eye(n, dtype=bool)
    pairwise = dists[mask]
    d_mean = pairwise.mean() + 1e-8

    # plus proche voisin pour chaque agent
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)

    s = np.mean(nearest / d_mean)
    dos = 1.0 - np.clip(s, 0.0, 1.0)
    return float(dos)


def degree_of_alignment(velocities: np.ndarray) -> float:
    """
    Degree of Alignment (DoA) ∈ [-1, 1]

    DoA = moyenne des cosinus d'angle entre toutes les paires
          de vecteurs vitesse (normalisés).
    """
    velocities = np.asarray(velocities)
    n = len(velocities)
    if n < 2:
        return 0.0

    norms = np.linalg.norm(velocities, axis=1)
    valid = norms > 1e-8
    if np.sum(valid) < 2:
        return 0.0

    v = velocities[valid] / norms[valid][:, None]
    dot_products = np.dot(v, v.T)

    # on enlève la diagonale (self-self)
    np.fill_diagonal(dot_products, 0.0)
    m = len(v)
    doa = dot_products.sum() / (m * (m - 1))
    return float(doa)
