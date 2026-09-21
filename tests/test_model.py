import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal

from spatial_glmm_pkg import SpatialCustomLinkPQL, Poisson
from spatial_glmm_pkg.covariance import spatial_covariance, compute_distance_matrix


def _make_test_data(n=200, seed=42):
    """Generate simple test data."""
    np.random.seed(seed)

    # 3 groups
    K = 3
    group = np.random.choice(K, size=n)
    X = np.eye(K)[group]

    coords = np.random.uniform(0, 10, size=(n, 2))
    dist_mat = compute_distance_matrix(coords)

    sigma2, rho = 0.2, 1.0
    G = spatial_covariance(dist_mat, sigma2, rho)
    b = multivariate_normal.rvs(mean=np.zeros(n), cov=G)
    b = b - b.mean()

    a = -1.47
    true_beta = np.log([40, 20, 55])
    eta = X @ true_beta + np.log(expit(a + b))
    mu = np.exp(eta)
    Y = np.random.poisson(mu)

    return Y, X, coords, b, true_beta


def test_mu_computation():
    """mu = exp(X @ beta) * expit(a + b)"""
    Y, X, coords, _, _ = _make_test_data()

    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    beta = np.array([3.7, 3.0, 4.0])
    b = np.zeros(model.n)
    mu, eta = model._compute_current_mu(beta, b)

    p = expit(-1.47)
    for k in range(3):
        mask = X[:, k] == 1
        expected = np.exp(beta[k]) * p
        np.testing.assert_allclose(mu[mask], expected, atol=1e-6)


def test_b_sums_to_zero():
    """Lagrange constraint should enforce sum(b) = 0."""
    Y, X, coords, _, _ = _make_test_data()

    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    model.fit(max_iter=50, update_var_components_every=9999)
    assert abs(np.sum(model.b)) < 1e-8


def test_convergence():
    """Model should converge within max_iter."""
    Y, X, coords, _, _ = _make_test_data()

    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    model.fit(max_iter=200, tol=1e-4, update_var_components_every=9999)
    assert model.converged


def test_mu_positive():
    """Fitted mu should always be positive."""
    Y, X, coords, _, _ = _make_test_data()

    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    model.fit(max_iter=100, update_var_components_every=9999)
    fitted = model.predict()
    assert np.all(fitted['mu'] > 0)


def test_predict_consistency():
    """mu should equal lambda * p."""
    Y, X, coords, _, _ = _make_test_data()

    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    model.fit(max_iter=100, update_var_components_every=9999)
    fitted = model.predict()
    expected_mu = fitted['lambda'] * fitted['p']
    np.testing.assert_allclose(fitted['mu'], expected_mu, atol=1e-6)


def test_x_preserved_as_one_hot():
    """X should not be reparameterized."""
    Y, X, coords, _, _ = _make_test_data()

    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    assert model.X.shape[1] == 3
    np.testing.assert_array_equal(model.X, X)


def test_invalid_x_raises():
    """Non-one-hot X should raise ValueError."""
    Y, _, coords, _, _ = _make_test_data()
    X_bad = np.random.randn(len(Y), 3)

    try:
        SpatialCustomLinkPQL(Y, X_bad, a=-1.47, coords=coords, verbose=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_p_bounded():
    """Capture probability should be in (0, 1)."""
    Y, X, coords, _, _ = _make_test_data()

    model = SpatialCustomLinkPQL(Y, X, a=-1.47, coords=coords, verbose=False)
    model.fit(max_iter=100, update_var_components_every=9999)
    fitted = model.predict()
    assert np.all(fitted['p'] > 0)
    assert np.all(fitted['p'] < 1)
