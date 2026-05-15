"""
Utility functions for distribution comparison metrics.

Metrics:
    - Jensen-Shannon Divergence (JSD)
    - KL Divergence
    - Shannon Entropy
    - Wasserstein Distance
    - Distribution normalization helpers
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon


def normalize_distribution(dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a distribution to sum to 1, with smoothing."""
    dist = np.asarray(dist, dtype=np.float64)
    dist = np.maximum(dist, 0)  # ensure non-negative
    total = dist.sum()
    if total == 0:
        # Uniform distribution if all zeros
        return np.ones_like(dist) / len(dist)
    return dist / total


def shannon_entropy(dist: np.ndarray, base: float = 2.0) -> float:
    """Compute Shannon entropy of a distribution.

    Args:
        dist: Probability distribution (will be normalized).
        base: Logarithm base (default: 2 for bits).

    Returns:
        Entropy in the specified base.
    """
    dist = normalize_distribution(dist)
    return float(scipy_entropy(dist, base=base))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: True distribution.
        q: Approximate distribution.
        eps: Smoothing constant to avoid log(0).

    Returns:
        KL divergence (non-negative, asymmetric).
    """
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    # Add smoothing
    q = q + eps
    q = q / q.sum()
    return float(scipy_entropy(p, q, base=2))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence between two distributions.

    JSD is symmetric and bounded in [0, 1] (when using base-2 log).

    Args:
        p: First distribution.
        q: Second distribution.

    Returns:
        JSD value in [0, 1].
    """
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return float(jensenshannon(p, q, base=2) ** 2)  # scipy returns sqrt(JSD)


def wasserstein_dist(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Wasserstein (Earth Mover's) distance between distributions.

    Args:
        p: First distribution.
        q: Second distribution.

    Returns:
        Wasserstein distance.
    """
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    return float(wasserstein_distance(
        range(len(p)), range(len(q)), p, q
    ))


def batch_jsd(
    distributions_p: np.ndarray,
    distributions_q: np.ndarray,
) -> np.ndarray:
    """Compute JSD for each pair of distributions in two matrices.

    Args:
        distributions_p: (N, K) array of N distributions over K categories.
        distributions_q: (N, K) array of N distributions over K categories.

    Returns:
        (N,) array of JSD values.
    """
    assert distributions_p.shape == distributions_q.shape
    n = distributions_p.shape[0]
    jsds = np.zeros(n)
    for i in range(n):
        jsds[i] = jensen_shannon_divergence(distributions_p[i], distributions_q[i])
    return jsds
