import numpy as np

def degree_of_sparsity(positions):
    """
    Compute Degree of Sparsity (DoS)
    → Mean nearest-neighbor distance normalized by mean inter-agent distance.
    """
    n = len(positions)
    if n < 2:
        return 0

    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    dos = np.mean(nearest)
    return dos


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