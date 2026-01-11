"""
metrics.py

Collective behavior metrics for multi-agent systems.

This module implements two scalar metrics commonly used in collective motion studies:
- Degree of Sparsity (DoS): measures how spatially spread agents are.
- Degree of Alignment (DoA): measures how aligned agent velocities are.

Both metrics are normalized to lie in [0, 1].
"""

import numpy as np


def degree_of_sparsity(positions, R=None, world_size=None):
    """
    Compute the Degree of Sparsity (DoS) of a set of agent positions.

    DoS is defined as the mean nearest-neighbor distance normalized by R,
    where R is a characteristic length scale of the domain.

    Args:
        positions: (N, 2) array of agent positions.
        R: Normalization distance. If None, it is set to sqrt(2) * world_size.
        world_size: Size of the square domain (required if R is None).

    Returns:
        DoS value in [0, 1].
    """
    n = len(positions)
    if n < 2:
        return 1.0

    if R is None:
        assert world_size is not None, "Provide world_size if R is None"
        R = np.sqrt(2) * world_size  # diagonal of the square domain

    dists = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1
    )
    np.fill_diagonal(dists, np.inf)

    nearest = np.min(dists, axis=1)
    dos = np.clip(np.mean(nearest) / R, 0.0, 1.0)
    return float(dos)


def degree_of_alignment(velocities):
    """
    Compute the Degree of Alignment (DoA) of a set of agent velocities.

    DoA is defined as the average cosine similarity between all pairs
    of normalized velocity vectors (excluding self-pairs).

    Args:
        velocities: (N, 2) array of agent velocity vectors.

    Returns:
        DoA value in [0, 1].
    """
    n = len(velocities)
    if n < 2:
        return 0

    norms = np.linalg.norm(velocities, axis=1)
    valid = norms > 1e-6
    if np.sum(valid) < 2:
        return 0

    v = velocities[valid] / norms[valid][:, None]
    dot = np.dot(v, v.T)

    off_diag = (np.sum(dot) - len(v)) / (len(v) * (len(v) - 1))
    return np.clip(off_diag, 0, 1)
