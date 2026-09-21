import numpy as np
from scipy.optimize import minimize

from .base import VarianceEstimator
from ..covariance import spatial_covariance
from ..links import CustomLink


class REMLEstimator(VarianceEstimator):
    """
    PQL-REML variance component estimation.

    Known to overestimate variance components when the link function
    is nonlinear due to the linearization of z(b).
    """

    def __init__(self, bounds=None):
        self.bounds = bounds or [(-10, 3), (-5, 3)]

    def estimate(self, model):
        """
        Estimate variance components.
        """
        theta_init = np.log([
            np.clip(model.sigma2, 1e-4, 10),
            np.clip(model.rho, 0.1, 10)
        ])

        try:
            result = minimize(
                self._reml_objective,
                theta_init,
                args=(model,),       # pass model to objective
                method='L-BFGS-B',
                bounds=self.bounds,
            )

            if result.success:
                sigma2_new = np.exp(result.x[0])
                rho_new = np.exp(result.x[1])
                return sigma2_new, np.clip(rho_new, 0.1, 10)
        except Exception:
            pass

        if model.verbose:
            print("Couldn't estimate variance components")
        return model.sigma2, model.rho

    def _reml_objective(self, theta_log, model):
        """
        Proper PQL-REML objective for the linearized mixed model:

            y* = X beta + Z b + e
            b ~ N(0, G)
            e ~ N(0, W^{-1})

        REML criterion:
            0.5 [ log|V| + log|X^T V^{-1} X|
                  + (y* - X beta_hat)^T V^{-1} (y* - X beta_hat) ]
        """
        try:
            # --- Transform parameters ---
            sigma2 = np.exp(theta_log[0])
            rho = np.exp(theta_log[1])

            # --- Spatial covariance ---
            G = spatial_covariance(model.dist_mat, sigma2, rho, model.correlation)
            G += 1e-6 * np.eye(model.n)

            # --- Working quantities from current beta, b ---
            w, y_star = model._compute_working_response(model.beta, model.b)
            z_prime, offset = CustomLink.compute_linearization_coefficients(model.a, model.b)

            W_inv = np.diag(1.0 / np.clip(w, 1e-8, None))
            Z = np.diag(z_prime)

            # Adjust working response
            y_adj = y_star - offset

            # --- Marginal covariance ---
            V = Z @ G @ Z.T + W_inv
            V += 1e-6 * np.eye(model.n)

            # Cholesky for stability
            L = np.linalg.cholesky(V)
            logdet_V = 2.0 * np.sum(np.log(np.diag(L)))

            # Solve V^{-1} y and V^{-1} X using Cholesky
            Vinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y_adj))

            if model.p > 0:
                Vinv_X = np.linalg.solve(L.T, np.linalg.solve(L, model.X))

                XtVinvX = model.X.T @ Vinv_X
                XtVinvy = model.X.T @ Vinv_y

                # log |X^T V^{-1} X|
                L2 = np.linalg.cholesky(XtVinvX + 1e-8 * np.eye(model.p))
                logdet_X = 2.0 * np.sum(np.log(np.diag(L2)))

                beta_hat = np.linalg.solve(XtVinvX, XtVinvy)

                resid = y_adj - model.X @ beta_hat
                Vinv_resid = np.linalg.solve(L.T, np.linalg.solve(L, resid))
                quad = resid.T @ Vinv_resid

            else:
                logdet_X = 0.0
                quad = y_adj.T @ Vinv_y

            reml = 0.5 * (logdet_V + logdet_X + quad)

            # mild regularization (optional but stabilizing)
            # reml += 0.001 * (theta_log[0]**2 + theta_log[1]**2)

            return reml

        except np.linalg.LinAlgError:
            print("linalg error with est. var components")
            return 1e10