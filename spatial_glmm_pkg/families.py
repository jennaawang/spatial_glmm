"""
Distribution families for Spatial GLMM with custom link, fit via PQL.

Follows the statsmodels pattern. Each Family defines the distribution-specific
math (variance, weights, deviance, log-likelihood), and the fitting algorithm 
calls these methods generically.

Usage:    
    # Poisson model (default)
    model = SpatialCustomLinkPQL(Y, X, a, coords, family=Poisson())

    # Negative Binomial model
    model = SpatialCustomLinkPQL(Y, X, a, coords, family=NegativeBinomial())
"""

import numpy as np
from scipy.special import gammaln

# ======================================================================
# Base Family
# ======================================================================
class Family:
    """
    Base class for exponential family distributions.
    
    Subclasses must implement:
        variance(mu)        - V(mu), the variance as a function of the mean
        deviance(Y, mu)     - vector of deviance contributions
        log_likelihood(Y, mu) - log-likelihood contributions
    
    The base class provides working weights and working response for
    log-link GLMs, derived from the variance function.
    """
    
    def variance(self, mu):
        raise NotImplementedError
    
    def deviance(self, Y, mu):
        raise NotImplementedError
    
    def log_likelihood(self, Y, mu):
        raise NotImplementedError
    
    def working_weights(self, mu):
        """
        IRLS working weights for log link: w = mu^2 / V(mu).
        """
        mu_safe = np.clip(mu, 1e-10, None)
        v = self.variance(mu_safe)
        w = mu_safe ** 2 / np.clip(v, 1e-10, None)
        return np.clip(w, 1e-10, 1e10)
    
    def working_response(self, Y, mu, eta):
        """
        IRLS working response for log link:
            y* = eta + (Y - mu) / mu
        
        Same formula for any distribution with log link because it
        depends only on the link function, not the variance.
        """
        mu_safe = np.clip(mu, 1e-10, None)
        y_star = eta + (Y - mu_safe) / mu_safe
        
        if not np.all(np.isfinite(y_star)):
            y_star = np.nan_to_num(y_star, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return y_star
    
    def pearson_residuals(self, Y, mu):
        """(Y - mu) / sqrt(V(mu))"""
        mu_safe = np.clip(mu, 1e-10, None)
        v = self.variance(mu_safe)
        return (Y - mu_safe) / np.sqrt(np.clip(v, 1e-10, None))
    
    def deviance_residuals(self, Y, mu):
        """sign(Y - mu) * sqrt(|d_i|)"""
        d = self.deviance(Y, mu)
        d = np.clip(d, 0, None)
        return np.sign(Y - mu) * np.sqrt(d)
    
    def estimate_dispersion(self, Y, mu, df_resid):
        """
        Estimate dispersion from Pearson residuals.
        Checks for overdispersion for Poisson.
        Updates the k parameter for NB.
        
        Default: Pearson chi-squared / df_resid
        """
        r = self.pearson_residuals(Y, mu)
        return np.sum(r ** 2) / df_resid


# ======================================================================
# Poisson
# ======================================================================
class Poisson(Family):
    """
    Poisson family: Y ~ Pois(mu)
    
    Variance: V(mu) = mu
    No dispersion parameter to estimate.
    """
    
    name = 'Poisson'
    
    def __init__(self):
        self.dispersion = 1.0  # fixed for Poisson
    
    def variance(self, mu):
        return np.clip(mu, 1e-10, None)
    
    def deviance(self, Y, mu):
        """
        Deviance contribution: 2 * [Y * log(Y/mu) - (Y - mu)]
        """
        mu_safe = np.clip(mu, 1e-10, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(Y > 0, Y / mu_safe, 1.0)
            d = 2 * (Y * np.log(ratio) - (Y - mu_safe))
        return np.clip(d, 0, None) # used for QQ plots
    
    def log_likelihood(self, Y, mu):
        """
        Poisson log-likelihood: Y*log(mu) - mu - log(Y!)
        """
        mu_safe = np.clip(mu, 1e-10, None)
        return Y * np.log(mu_safe) - mu_safe - gammaln(Y + 1)
    
    def estimate_dispersion(self, Y, mu, df_resid):
        """
        Poisson dispersion is fixed at 1. But we return the Pearson
        estimate as a diagnostic because values >> 1 suggest overdispersion.
        """
        r = self.pearson_residuals(Y, mu)
        pearson_disp = np.sum(r ** 2) / df_resid
        self._pearson_dispersion = pearson_disp  # store for summary
        return 1.0  # dispersion is 1 for Poisson
    
    def __repr__(self):
        return "Poisson()"


# ======================================================================
# Negative Binomial
# ======================================================================
class NegativeBinomial(Family):
    """
    Negative Binomial family: Y ~ NB(mu, k)
    
    Parameterization:
        E[Y] = mu
        Var[Y] = mu + mu^2 / k
    
    k is the dispersion parameter (larger k = closer to Poisson).
    k is estimated iteratively during PQL fitting.
    
    Parameters
    ----------
    k_init : float
        Initial value for the dispersion parameter k. Default 1.0.
    k_bounds : tuple
        (lower, upper) bounds for k. Default (0.01, 1000).
    """
    
    name = 'NegativeBinomial'
    
    def __init__(self, k_init=1.0, k_bounds=(0.01, 1000.0)):
        self.k = k_init
        self.k_bounds = k_bounds
        self.dispersion = 1.0  # scale parameter separate from k
    
    def variance(self, mu):
        """V(mu) = mu + mu^2 / k"""
        mu_safe = np.clip(mu, 1e-10, None)
        return mu_safe + mu_safe ** 2 / self.k
    
    def deviance(self, Y, mu):
        """
        NB deviance contribution:
            2 * [Y * log(Y/mu) - (Y + k) * log((Y + k) / (mu + k))]
        """
        mu_safe = np.clip(mu, 1e-10, None)
        k = self.k
        
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = np.where(Y > 0, Y * np.log(Y / mu_safe), 0.0)
            term2 = (Y + k) * np.log((Y + k) / (mu_safe + k))
        
        d = 2 * (term1 - term2)
        return np.clip(d, 0, None)
    
    def log_likelihood(self, Y, mu):
        """
        NB log-likelihood:
            log Gamma(Y + k) - log Gamma(k) - log Gamma(Y + 1)
            + k*log(k/(mu+k)) + Y*log(mu/(mu+k))
        """
        mu_safe = np.clip(mu, 1e-10, None)
        k = self.k
        
        ll = (gammaln(Y + k) - gammaln(k) - gammaln(Y + 1)
              + k * np.log(k / (mu_safe + k))
              + Y * np.log(mu_safe / (mu_safe + k)))
        return ll
    
    def estimate_dispersion(self, Y, mu, df_resid):
        """
        Estimate k by maximizing the NB profile log-likelihood
        over k, with mu fixed at current estimates.
        
        Uses Brent's method followed by MOMs.
        """
        from scipy.optimize import minimize_scalar
        
        mu_safe = np.clip(mu, 1e-10, None)
        
        def neg_profile_ll(log_k):
            k = np.exp(log_k)
            ll = np.sum(
                gammaln(Y + k) - gammaln(k)
                + k * np.log(k / (mu_safe + k))
                + Y * np.log(mu_safe / (mu_safe + k))
            )
            return -ll # minimizing -ll is same as maximizing ll
        
        # Search over log(k) for numerical stability
        log_k_bounds = (np.log(self.k_bounds[0]), np.log(self.k_bounds[1]))
        
        try:
            result = minimize_scalar(
                neg_profile_ll,
                bounds=log_k_bounds,
                method='bounded'
            )
            if result.success:
                self.k = np.exp(result.x)
        except Exception:
            # Fall back to method-of-moments estimate
            r = self.pearson_residuals(Y, mu)
            pearson_disp = np.sum(r ** 2) / df_resid
            if pearson_disp > 1:
                # From V(Y) = mu + mu^2/k and Pearson dispersion ~ V(Y)/mu:
                # dispersion ~ 1 + mean(mu)/k => k ~ mean(mu)/(dispersion - 1)
                k_mom = np.mean(mu_safe) / (pearson_disp - 1)
                self.k = np.clip(k_mom, self.k_bounds[0], self.k_bounds[1])
        
        return self.k
    
    def __repr__(self):
        return f"NegativeBinomial(k={self.k:.4f})"