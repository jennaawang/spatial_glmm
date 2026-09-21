"""
Simulate data from the biophysical model.

Usage:
    from spatial_glmm.simulate import simulate_srt

    Y, X, coords, true_params = simulate_srt(
        true_beta=np.log([40, 20, 55]),
        a=-1.47,
        sigma2=0.2,
        rho=1.0,
        n=1000
    )
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.stats import multivariate_normal

from .covariance import spatial_covariance, compute_distance_matrix


def simulate_srt(true_beta, a=-1.47, sigma2=0.2, rho=1.0,
                 n=1000, coord_range=(0, 10),
                 correlation='exponential', seed=None):
    """
    Parameters
    -----------
    true_beta : array
        True fixed effects. Length determines number of cell types.
    a : float or array
        Baseline logit capture probability.
    sigma2 : float
        Spatial variance.
    rho : float
        Spatial range.
    n : int
        Number of observations.
    coord_range : tuple
        (min, max) for uniform spatial coordinates.
    correlation : str
        Spatial correlation function.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Y : array, shape (n,)
        Observed counts.
    X : array, shape (n, K)
        One-hot design matrix.
    coords : array, shape (n, 2)
        Spatial coordinates.
    true_params : dict
        Dictionary with all true parameters:
        'beta', 'b', 'a', 'sigma2', 'rho', 'mu', 'p', 'group'
    """
    if seed is not None:
        np.random.seed(seed)

    true_beta = np.asarray(true_beta)
    K = len(true_beta)

    # Assign groups
    group = np.random.choice(K, size=n)
    X = np.eye(K)[group]

    # Spatial coordinates
    lo, hi = coord_range
    coords = np.random.uniform(lo, hi, size=(n, 2))

    # Spatial random effects
    dist_mat = compute_distance_matrix(coords)
    G = spatial_covariance(dist_mat, sigma2, rho, correlation)
    G += 1e-6 * np.eye(n)
    b = multivariate_normal.rvs(mean=np.zeros(n), cov=G)
    b = b - b.mean()

    # Capture probability
    if np.isscalar(a):
        a_vec = np.full(n, a)
    else:
        a_vec = np.asarray(a)
    p = expit(a_vec + b)

    # Expected counts
    mu = np.exp(X @ true_beta) * p

    # Observed counts
    Y = np.random.poisson(mu)

    true_params = {
        'beta': true_beta,
        'b': b,
        'a': a_vec,
        'sigma2': sigma2,
        'rho': rho,
        'mu': mu,
        'p': p,
        'group': group
    }

    return Y, X, coords, true_params