import numpy as np
from scipy.spatial.distance import cdist

def compute_distance_matrix(coords):
    return cdist(coords, coords)

def spatial_covariance(dist_mat, sigma2, rho, correlation='exponential'):
    """Compute spatial covariance matrix"""
    if correlation == 'exponential':
        G = sigma2 * np.exp(-dist_mat / rho)
    elif correlation == 'gaussian':
        G = sigma2 * np.exp(-(dist_mat / rho)**2)
    elif correlation == 'matern':
        scaled_dist = np.sqrt(3) * dist_mat / rho
        G = sigma2 * (1 + scaled_dist) * np.exp(-scaled_dist)
    else:
        raise ValueError(f"Unknown correlation: {correlation}")
    return G

def spatial_cross_covariance(dist_cross, sigma2, rho, correlation='exponential'):
    if correlation == 'exponential':
        return sigma2 * np.exp(-dist_cross / rho)
    elif correlation == 'gaussian':
        return sigma2 * np.exp(-(dist_cross / rho) ** 2)
    elif correlation == 'matern':
        scaled_dist = np.sqrt(3) * dist_cross / rho
        return sigma2 * (1 + scaled_dist) * np.exp(-scaled_dist)
    else:
        raise ValueError(f"Unknown correlation: {correlation}")