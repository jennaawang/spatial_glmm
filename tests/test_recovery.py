import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal

from spatial_glmm_pkg import SpatialCustomLinkPQL, Poisson
from spatial_glmm_pkg.covariance import spatial_covariance, compute_distance_matrix


def _simulate(true_beta, a, n=500, sigma2=0.2, rho=1.0, seed=42):
    """Simulate data from the biophysical model."""
    np.random.seed(seed)

    K = len(true_beta)
    group = np.random.choice(K, size=n)
    X = np.eye(K)[group]

    coords = np.random.uniform(0, 10, size=(n, 2))
    dist_mat = compute_distance_matrix(coords)

    G = spatial_covariance(dist_mat, sigma2, rho)
    b = multivariate_normal.rvs(mean=np.zeros(n), cov=G)
    b = b - b.mean()

    a_vec = np.full(n, a)
    eta = X @ true_beta + np.log(expit(a_vec + b))
    mu = np.exp(eta)
    Y = np.random.poisson(mu)

    return Y, X, coords, b


def _fit_and_check(Y, X, coords, a, true_beta, true_b=None, atol=0.5):
    """Fit model with frozen variance components and check beta recovery."""
    model = SpatialCustomLinkPQL(Y, X, a=a, coords=coords, verbose=False)
    model.fit(
        init_sigma2=0.2,
        init_rho=1.0,
        init_random=true_b,
        update_var_components_every=9999,
        damping=0.5,
        max_iter=150,
        tol=1e-4
    )
    assert model.converged
    np.testing.assert_allclose(model.beta, true_beta, atol=atol)
    return model


# --- Vary number of cell types ---

def test_recovery_2_cell_types():
    true_beta = np.log([30, 50])
    Y, X, coords, b = _simulate(true_beta, a=-1.47)
    _fit_and_check(Y, X, coords, -1.47, true_beta, true_b=b)


def test_recovery_3_cell_types():
    true_beta = np.log([40, 20, 55])
    Y, X, coords, b = _simulate(true_beta, a=-1.47)
    _fit_and_check(Y, X, coords, -1.47, true_beta, true_b=b)


def test_recovery_5_cell_types():
    true_beta = np.log([10, 25, 40, 60, 80])
    Y, X, coords, b = _simulate(true_beta, a=-1.47)
    _fit_and_check(Y, X, coords, -1.47, true_beta, true_b=b)


# --- Vary capture rates ---

def test_recovery_capture_rate_10pct():
    a = np.log(0.10 / 0.90)  # logit(0.10)
    true_beta = np.log([40, 20, 55])
    Y, X, coords, b = _simulate(true_beta, a=a)
    _fit_and_check(Y, X, coords, a, true_beta, true_b=b)


def test_recovery_capture_rate_25pct():
    a = np.log(0.25 / 0.75)  # logit(0.25)
    true_beta = np.log([40, 20, 55])
    Y, X, coords, b = _simulate(true_beta, a=a)
    _fit_and_check(Y, X, coords, a, true_beta, true_b=b)


def test_recovery_capture_rate_40pct():
    a = np.log(0.40 / 0.60)  # logit(0.40)
    true_beta = np.log([40, 20, 55])
    Y, X, coords, b = _simulate(true_beta, a=a)
    _fit_and_check(Y, X, coords, a, true_beta, true_b=b)


# --- Vary expression levels ---

def test_recovery_low_expression():
    true_beta = np.log([5, 8, 12])
    Y, X, coords, b = _simulate(true_beta, a=-1.47)
    _fit_and_check(Y, X, coords, -1.47, true_beta, true_b=b, atol=0.8)


def test_recovery_medium_expression():
    true_beta = np.log([30, 50, 70])
    Y, X, coords, b = _simulate(true_beta, a=-1.47)
    _fit_and_check(Y, X, coords, -1.47, true_beta, true_b=b)


def test_recovery_high_expression():
    true_beta = np.log([100, 200, 500])
    Y, X, coords, b = _simulate(true_beta, a=-1.47)
    _fit_and_check(Y, X, coords, -1.47, true_beta, true_b=b)


# --- Vary spatial parameters ---

def test_recovery_weak_spatial():
    true_beta = np.log([40, 20, 55])
    Y, X, coords, b = _simulate(true_beta, a=-1.47, sigma2=0.05, rho=0.5)
    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    model.fit(init_sigma2=0.05, init_rho=0.5, init_random=b,
              update_var_components_every=9999, damping=0.5,
              max_iter=150, tol=1e-4)
    assert model.converged
    np.testing.assert_allclose(model.beta, true_beta, atol=0.5)


def test_recovery_strong_spatial():
    true_beta = np.log([40, 20, 55])
    Y, X, coords, b = _simulate(true_beta, a=-1.47, sigma2=1.0, rho=5.0)
    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    model.fit(init_sigma2=1.0, init_rho=5.0, init_random=b,
              update_var_components_every=9999, damping=0.5,
              max_iter=150, tol=1e-4)
    assert model.converged
    np.testing.assert_allclose(model.beta, true_beta, atol=0.5)


# --- Unbalanced groups ---

def test_recovery_unbalanced():
    np.random.seed(42)
    true_beta = np.log([40, 20, 55])
    n = 500
    # Highly unbalanced: 50, 200, 250
    group = np.concatenate([np.full(50, 0), np.full(200, 1), np.full(250, 2)])
    np.random.shuffle(group)
    X = np.eye(3)[group.astype(int)]

    coords = np.random.uniform(0, 10, size=(n, 2))
    dist_mat = compute_distance_matrix(coords)
    G = spatial_covariance(dist_mat, 0.2, 1.0)
    b = multivariate_normal.rvs(mean=np.zeros(n), cov=G)
    b = b - b.mean()

    a = -1.47
    eta = X @ true_beta + np.log(expit(a + b))
    Y = np.random.poisson(np.exp(eta))

    _fit_and_check(Y, X, coords, a, true_beta, true_b=b, atol=0.8)


# --- Bad initialization ---

def test_recovery_bad_init():
    true_beta = np.log([40, 20, 55])
    Y, X, coords, b = _simulate(true_beta, a=-1.47)

    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    model.fit(
        init_sigma2=0.2,
        init_rho=1.0,
        init_beta=np.zeros(3),  # far from truth
        update_var_components_every=9999,
        damping=0.5,
        max_iter=200,
        tol=1e-4
    )
    assert model.converged
    np.testing.assert_allclose(model.beta, true_beta, atol=0.5)


# --- Multiple seeds for consistency ---

def test_recovery_across_seeds():
    """Check beta bias is small across multiple simulations."""
    true_beta = np.log([40, 20, 55])
    biases = []

    for seed in range(10):
        Y, X, coords, b = _simulate(true_beta, a=-1.47, seed=seed)
        model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords,
                                     verbose=False)
        model.fit(init_sigma2=0.2, init_rho=1.0, init_random=b,
                  update_var_components_every=9999, damping=0.5,
                  max_iter=150, tol=1e-4)
        biases.append(model.beta - true_beta)

    mean_bias = np.mean(biases, axis=0)
    assert np.all(np.abs(mean_bias) < 0.3)
