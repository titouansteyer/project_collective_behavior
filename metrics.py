import numpy as np

# DoS between 0 and 1
def degree_of_sparsity(positions, R=50):
    n = len(positions)
    if n < 2:
        return 1

    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)

    nearest = np.min(dists, axis=1)
    max_dist = R
    dos = np.clip(np.mean(nearest) / max_dist, 0, 1)
    return dos

# DoA between 0 and 1
def degree_of_alignment(velocities):
    n = len(velocities)
    if n < 2:
        return 0

    norms = np.linalg.norm(velocities, axis=1)
    valid = norms > 1e-6
    if np.sum(valid) < 2:
        return 0

    v = velocities[valid] / norms[valid][:, None]
    dot = np.dot(v, v.T)

    off_diag = (np.sum(dot) - len(v)) / (len(v)*(len(v)-1))
    return np.clip(off_diag, 0, 1)
