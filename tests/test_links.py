import numpy as np
from scipy.special import expit

from spatial_glmm_pkg.links import CustomLink


def test_z_values():
    """z(b) = log(expit(a + b))"""
    a = np.full(3, -1.47)
    b = np.array([0.0, 1.0, -1.0])
    z = CustomLink.evaluate_z(a, b)
    expected = np.log(expit(-1.47 + b))
    np.testing.assert_allclose(z, expected, atol=1e-10)


def test_z_derivative():
    """z'(b) = 1 - expit(a + b)"""
    a = np.full(3, -1.47)
    b = np.array([0.0, 1.0, -1.0])
    z_prime = CustomLink.evaluate_z_derivative(a, b)
    expected = 1 - expit(-1.47 + b)
    np.testing.assert_allclose(z_prime, expected, atol=1e-6)


# def test_z_second_derivative():
#     """z''(b) = -expit(a+b) * (1-expit(a+b))"""
#     a = np.full(3, -1.47)
#     b = np.array([0.0, 1.0, -1.0])
#     z_pp = CustomLink.evaluate_z_second_derivative(a, b)
#     p = expit(-1.47 + b)
#     expected = -p * (1 - p)
#     np.testing.assert_allclose(z_pp, expected, atol=1e-10)


def test_z_derivative_numerical():
    """z' should match numerical derivative of z."""
    a = np.full(4, -1.47)
    b = np.array([0.0, 0.5, -0.5, 1.0])
    eps = 1e-7
    numerical = ((CustomLink.evaluate_z(a, b + eps)
                  - CustomLink.evaluate_z(a, b - eps)) / (2 * eps))
    analytic = CustomLink.evaluate_z_derivative(a, b)
    np.testing.assert_allclose(analytic, numerical, atol=1e-5)


# def test_z_second_derivative_numerical():
#     """z'' should match numerical derivative of z'."""
#     a = np.full(4, -1.47)
#     b = np.array([0.0, 0.5, -0.5, 1.0])
#     eps = 1e-7
#     # Use unclipped version for numerical check
#     p_plus = expit(a + b + eps)
#     p_minus = expit(a + b - eps)
#     z_prime_plus = 1 - p_plus
#     z_prime_minus = 1 - p_minus
#     numerical = (z_prime_plus - z_prime_minus) / (2 * eps)
#     analytic = CustomLink.evaluate_z_second_derivative(a, b)
#     np.testing.assert_allclose(analytic, numerical, atol=1e-5)


def test_linearization_exact_at_expansion_point():
    """offset + z' * b should equal z(b) at the expansion point."""
    a = np.full(5, -1.47)
    b = np.random.randn(5) * 0.3
    z_prime, offset = CustomLink.compute_linearization_coefficients(a, b)
    z_exact = CustomLink.evaluate_z(a, b)
    z_linear = offset + z_prime * b
    np.testing.assert_allclose(z_linear, z_exact, atol=1e-10)


def test_z_always_negative():
    """z(b) = log(expit(...)) should always be negative."""
    a = np.full(100, -1.47)
    b = np.random.randn(100) * 2
    z = CustomLink.evaluate_z(a, b)
    assert np.all(z <= 0)
