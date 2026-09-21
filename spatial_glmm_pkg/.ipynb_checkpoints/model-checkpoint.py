import warnings
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.linalg import solve
from .families import Poisson
from .links import CustomLink
from .covariance import spatial_covariance, spatial_cross_covariance, compute_distance_matrix
from .variance_estimation import REMLEstimator

class SpatialCustomLinkPQL:
    """
    PQL for Poisson and NB with custom link
    
    Model: log(mu_i) = X*beta + log(expit(a + b_i))
    
    where:
    - X has NO intercept column
    - a is a known constant baseline
    - b_i ~ N(0, G) with constraint sum(b_i) = 0
    """
    
    def __init__(self, Y, X, a, coords, family=None, correlation='exponential', variance_estimator=None, verbose=True):
        """
        Parameters:
        -----------
        Y : array, shape (n,)
            Observed counts
        X : array, shape (n, p)
            Design matrix WITHOUT intercept column
        a : float or array
            Known baseline logit-probability (constant or spatially varying)
        coords : array, shape (n, 2)
            Spatial coordinates
        """
        self.family = family if family is not None else Poisson()
        
        self.Y = np.asarray(Y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D one-hot encoded matrix.")
    
        n, K = X.shape
    
        # Basic OHE check (rows should sum to 1)
        row_sums = X.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of X must sum to 1 (valid one-hot encoding).")
    
        self.X = X
        
        # Handle a as scalar or array
        if np.isscalar(a):
            self.a = np.full(len(Y), a, dtype=float)
        else:
            self.a = np.asarray(a, dtype=float)
            
        self.coords = np.asarray(coords, dtype=float)

        self.correlation = correlation
        self.verbose = verbose
        self.variance_estimator = variance_estimator or REMLEstimator()
        
        self.n = len(Y)
        self.p = X.shape[1]
        
        self.dist_mat = compute_distance_matrix(coords)
        self.dist_typical = np.mean(self.dist_mat)
        
        # Initialize parameters
        self.beta = None
        self.b = None
        self.sigma2 = None
        self.rho = None
        self.converged = False
        self.n_iter = 0
    
    def _compute_current_mu(self, beta, b):
        """
        Compute mu = exp(X*beta) * expit(a + b)
        """
        z = CustomLink.evaluate_z(self.a, b)
        
        # If X is empty (no covariates), eta_fixed = 0
        if self.p > 0:
            eta_fixed = self.X @ beta
        else:
            eta_fixed = np.zeros(self.n)
        
        eta = eta_fixed + z
        eta = np.clip(eta, -20, 20)
        
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-10, 1e10)
        
        return mu, eta
    
    def _compute_working_response(self, beta, b):
        """
        Compute working response and weights
        """
        mu, eta = self._compute_current_mu(beta, b)
        w = self.family.working_weights(mu)
        y_star = self.family.working_response(self.Y, mu, eta)
        return w, y_star

    def _solve_mixed_model_no_intercept(self, w, y_star, z_prime, offset, G_inv):
        """
        Solve mixed model. The constraint sum(b_i) = 0 is enforced DURING solving in this version.
        """
        W = np.diag(w)
        y_adj = y_star - offset
        Z = np.diag(z_prime)
        
        XtWX = self.X.T @ W @ self.X + 1e-6 * np.eye(self.p) # ridge for stability
        XtWZ = self.X.T @ W @ Z
        ZtWX = Z.T @ W @ self.X
        ZtWZ = Z.T @ W @ Z
        
        ones = np.ones((self.n, 1))
        
        # Augmented system with constraint sum(b)=0 using Lagrange multiplier
        LHS = np.block([
            [XtWX,    XtWZ,          np.zeros((self.p, 1))],
            [ZtWX,    ZtWZ + G_inv,  ones],
            [np.zeros((1, self.p)),  ones.T,  np.zeros((1, 1))]
        ])
        
        RHS = np.concatenate([
            self.X.T @ W @ y_adj,
            Z.T @ W @ y_adj,
            [0.0]
        ])

        params = np.linalg.solve(LHS, RHS)
        beta_new = params[:self.p]
        b_new = params[self.p:self.p + self.n] # params[-1] is the Lagrange multiplier, so we discard it

        print("betas info:")
        print(beta_new)
        
        return beta_new, b_new
    
    def fit(self, max_iter=100, tol=1e-5, init_beta=None,
            init_random=None,
            init_sigma2=0.5, 
            init_rho=2.0, 
            update_var_components_every=1, 
            damping=0.3
        ):
        """
        Fit model without intercept
        
        Parameters:
        -----------
        init_beta : array, shape (p,), optional
            Initial values for beta (no intercept)
        damping : float
            Damping factor (use 0.3-0.5 for stability)
        """

    
        # Initialize beta 
        if init_beta is None:
            if self.p > 0:
                self.beta = np.zeros(self.p)
            else:
                self.beta = np.array([])
        else:
            self.beta = np.array(init_beta)
            if len(self.beta) != self.p:
                raise ValueError(f"init_beta should have length {self.p}")
        
        # Initialize b
        if init_random is None:
            self.b = np.zeros(self.n)
        else: 
            self.b = init_random
            if len(self.b) != self.n:
                raise ValueError(f"init_random should have length {self.n}")
        
        # Initialize variance components
        self.sigma2 = init_sigma2
        self.rho = init_rho
        
        # Compute baseline
        baseline_p = np.mean(expit(self.a))
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("Spatial Poisson Custom Link PQL")
            print(f"{'='*70}")
            print(f"Observations: {self.n}")
            print(f"Covariates: {self.p} (no intercept)")
            print(f"Baseline 'a': mean={np.mean(self.a):.3f}, "
                  f"std={np.std(self.a):.3f}")
            print(f"Baseline probability: {baseline_p:.3f}")
            print(f"Mean Y: {np.mean(self.Y):.2f}")
            print(f"Initial sigma2={self.sigma2:.4f}, rho={self.rho:.4f}")
            print(f"Damping: {damping}")
            print(f"{'='*70}\n")
        
        for iteration in range(max_iter):
            beta_old = self.beta.copy() if self.p > 0 else np.array([])
            b_old = self.b.copy()
            sigma2_old = self.sigma2
            rho_old = self.rho
            
            # Spatial covariance
            G = spatial_covariance(self.dist_mat, self.sigma2, self.rho, self.correlation)  
            jitter = max(1e-5, self.sigma2 * 1e-3)
            G_stable = G + jitter * np.eye(self.n)
            
            try:
                G_inv = np.linalg.inv(G_stable)
            except np.linalg.LinAlgError:
                warnings.warn("Singular covariance matrix")
                break
            
            # Working response
            try:
                w, y_star = self._compute_working_response(self.beta, self.b)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error at iteration {iteration}: {e}")
                break
            
            # Linearization
            z_prime, offset = CustomLink.compute_linearization_coefficients(self.a, self.b)
            
            # Solve
            try:
                beta_new, b_new = self._solve_mixed_model_no_intercept(
                    w, y_star, z_prime, offset, G_inv
                )
                
            except Exception as e:
                if self.verbose:
                    print(f"Solve error at iteration {iteration}: {e}")
                break
            
            # Apply damping
            if self.p > 0:
                self.beta = beta_old + damping * (beta_new - beta_old)
            
            self.b = b_old + damping * (b_new - b_old)

            # Update variance components
            if (iteration + 1) % update_var_components_every == 0:
                sigma2_new, rho_new = self.variance_estimator.estimate(self)
                self.sigma2, self.rho = sigma2_new, rho_new
                
                if self.verbose:
                    beta_str = np.array2string(self.beta, precision=3, suppress_small=True) if self.p > 0 else "[]"
                    k_str = f", k={self.family.k:.3f}" if hasattr(self.family, 'k') else ""
                    print(f"Iter {iteration+1:3d}: beta={beta_str}, "
                          f"sigma2={self.sigma2:.4f}, rho={self.rho:.4f}{k_str}, "
                          f"b: [{np.min(self.b):.3f}, {np.max(self.b):.3f}], "
                          f"std(b)={np.std(self.b):.3f}")

            mu_current, _ = self._compute_current_mu(self.beta, self.b)
            df_resid = self.n - self.p
            self.family.estimate_dispersion(self.Y, mu_current, df_resid)
            
            # Convergence check
            beta_change = np.max(np.abs(self.beta - beta_old)) if self.p > 0 else 0.0
            b_change = np.max(np.abs(self.b - b_old))
            sigma2_change = abs(self.sigma2 - sigma2_old) / max(abs(sigma2_old), 1e-8)
            rho_change = abs(self.rho - rho_old) / max(abs(rho_old), 1e-8)
            max_change = max(beta_change, b_change, sigma2_change, rho_change)

            if hasattr(self.family, 'k'):
                if iteration > 0 and hasattr(self, '_prev_k'):
                    k_change = abs(self.family.k - self._prev_k) / max(abs(self._prev_k), 1e-8)
                    max_change = max(max_change, k_change)
                self._prev_k = self.family.k
            
            if max_change < tol:
                self.converged = True
                self.n_iter = iteration + 1
                if self.verbose:
                    print(f"\n{'='*70}")
                    print(f"Converged in {self.n_iter} iterations")
                    beta_str = np.array2string(self.beta, precision=4) if self.p > 0 else "[]"
                    print(f"Final beta: {beta_str}")
                    print(f"Final b: mean={np.mean(self.b):.6f}, std={np.std(self.b):.4f}")
                    print(f"Final sigma2={self.sigma2:.4f}, rho={self.rho:.4f}")
                    print(f"{'='*70}\n")
                break
        
        if not self.converged:
            self.n_iter = max_iter
            warnings.warn(f"Did not converge in {max_iter} iterations")
        
        return self
    
    def predict(self, X_new=None, a_new=None, coords_new=None):
        """Predict at new locations"""
        if X_new is None:
            mu_pred, eta_pred = self._compute_current_mu(self.beta, self.b)
            
            if self.p > 0:
                lambda_pred = np.exp(self.X @ self.beta)
            else:
                lambda_pred = np.ones(self.n)
            
            p_pred = expit(self.b + self.a)
            
            return {
                'mu': mu_pred,
                'lambda': lambda_pred,
                'p': p_pred,
                'eta': eta_pred,
                'b': self.b
            }
        else:
            # Prediction at new locations
            if self.p > 0:
                eta_fixed = X_new @ self.beta
                lambda_pred = np.exp(eta_fixed)
            else:
                eta_fixed = np.zeros(len(X_new))
                lambda_pred = np.ones(len(X_new))
            
            if coords_new is not None and a_new is not None:
                # Spatial prediction
                dist_cross = cdist(coords_new, self.coords)
                
                G_cross = spatial_cross_covariance(dist_cross, self.sigma2, self.rho, self.correlation)
                G = spatial_covariance(self.dist_mat, self.sigma2, self.rho, self.correlation)
                G_stable = G + 1e-6 * np.eye(self.n)
                G_inv = np.linalg.inv(G_stable)
                
                b_pred = G_cross @ G_inv @ self.b
                z_pred = CustomLink.evaluate_z(self.a, b_pred)
                eta_pred = eta_fixed + z_pred
                p_pred = expit((a_new + b_pred))
                mu_pred = lambda_pred * p_pred
                
                return {
                    'mu': mu_pred,
                    'lambda': lambda_pred,
                    'p': p_pred,
                    'eta': eta_pred,
                    'b': b_pred
                }
            else:
                return {'lambda': lambda_pred, 'eta': eta_fixed}
  
    
    def summary(self):
        """Print summary"""
        print("\n" + "="*70)
        print("Spatial Poisson Custom Link GLMM")
        print("="*70)
        print(f"Family: {self.family}")
        print(f"Converged: {self.converged} (in {self.n_iter} iterations)")
        print(f"\nBaseline 'a': {np.mean(self.a):.4f} (mean)")
        print(f"Spatial variance: sigma2 = {self.sigma2:.4f}")
        print(f"Spatial range: rho = {self.rho:.4f}")
        
        if self.p > 0:
            print(f"\nFixed effects (no intercept):")
            for i, val in enumerate(self.beta):
                print(f"  beta[{i}]: {val:.4f}")
        else:
            print("\n(No covariates in model)")
        
        print(f"\nRandom effects:")
        print(f"  mean: {np.mean(self.b):.6f} (should be ≈0)")
        print(f"  std:  {np.std(self.b):.4f}")
        print(f"  range: [{np.min(self.b):.4f}, {np.max(self.b):.4f}]")
        
        fitted = self.predict()
        
        print(f"\nFitted values:")
        print(f"  mu: mean={np.mean(fitted['mu']):.2f}, range=[{np.min(fitted['mu']):.2f}, {np.max(fitted['mu']):.2f}]")
        print(f"  p:  mean={np.mean(fitted['p']):.4f}, range=[{np.min(fitted['p']):.4f}, {np.max(fitted['p']):.4f}]")
        print("="*70 + "\n")
        if hasattr(self.family, 'k'):
            print(f"NB dispersion (k): {self.family.k:.4f}")
        if hasattr(self.family, '_pearson_dispersion'):
            print(f"Pearson dispersion: {self.family._pearson_dispersion:.4f}")
            if self.family._pearson_dispersion > 1.5:
                print("** Overdispersion detected. Consider NegativeBinomial. **")