"""
Diagnostic plots for SpatialPoissonCustomLinkPQL.

Usage:
    from diagnostics import PQLDiagnostics

    model.fit(...)
    diag = PQLDiagnostics(model)
    diag.plot_diagnostics()        # 2x2 summary
    diag.plot_variogram()          # standalone variogram
    diag.plot_convergence(history) # parameter traces
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import norm

class PQLDiagnostics:
    """
    Diagnostic plots for a fitted SpatialPoissonCustomLinkPQL model.
    
    Parameters
    ----------
    model : SpatialPoissonCustomLinkPQL
        A fitted model instance (must have called .fit() first).
    """
    
    def __init__(self, model):
        if model.beta is None:
            raise ValueError("Model has not been fitted yet.")
        self.model = model
        self._compute_residuals()
    
    def _compute_residuals(self):
        """Compute residuals and fitted values from the fitted model."""
        m = self.model
        pred = m.predict()
        
        self.mu = pred['mu']
        self.p_hat = pred['p']
        self.b_hat = pred['b']
        self.lambda_hat = pred['lambda']
        
        # Pearson residuals: (Y - mu) / sqrt(mu)
        # For Poisson, Var(Y) = mu
        self.resid_pearson = (m.Y - self.mu) / np.sqrt(np.clip(self.mu, 1e-10, None))
        
        # Deviance residuals
        # d_i = 2 * [Y*log(Y/mu) - (Y - mu)]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(m.Y > 0, m.Y / self.mu, 1.0)
            dev_component = 2 * (m.Y * np.log(ratio) - (m.Y - self.mu))
            dev_component = np.clip(dev_component, 0, None)
        self.resid_deviance = np.sign(m.Y - self.mu) * np.sqrt(dev_component)
    
    # ------------------------------------------------------------------
    # Main diagnostic panel
    # ------------------------------------------------------------------
    def plot_diagnostics(self, figsize=(13, 11)):
        """
        2x2 diagnostic panel:
          [0,0] Pearson residuals vs fitted
          [0,1] QQ plot of deviance residuals
          [1,0] Spatial map of Pearson residuals
          [1,1] Observed vs fitted
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Diagnostics', fontsize=14, fontweight='bold', y=0.98)
        
        self._plot_resid_vs_fitted(axes[0, 0])
        self._plot_qq(axes[0, 1])
        self._plot_spatial_residuals(axes[1, 0])
        self._plot_observed_vs_fitted(axes[1, 1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
    
    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------
    def _plot_resid_vs_fitted(self, ax):
        """Pearson residuals vs fitted values."""
        ax.scatter(self.mu, self.resid_pearson, alpha=0.4, s=15,
                   edgecolors='none', c='#2c3e50')
        ax.axhline(0, color='#c0392b', linewidth=1, linestyle='--')
        
        # LOWESS-style smoother: binned means
        sorted_idx = np.argsort(self.mu)
        n_bins = min(30, len(self.mu) // 10)
        if n_bins > 3:
            bins = np.array_split(sorted_idx, n_bins)
            bin_x = [self.mu[b].mean() for b in bins]
            bin_y = [self.resid_pearson[b].mean() for b in bins]
            ax.plot(bin_x, bin_y, color='#e74c3c', linewidth=2, label='Binned mean')
            ax.legend(fontsize=9, frameon=False)
        
        ax.set_xlabel('Fitted values (μ̂)')
        ax.set_ylabel('Pearson residuals')
        ax.set_title('Residuals vs Fitted')
    
    def _plot_qq(self, ax):
        """QQ plot of deviance residuals against normal quantiles."""
        r = np.sort(self.resid_deviance)
        n = len(r)
        theoretical = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        
        ax.scatter(theoretical, r, alpha=0.4, s=15,
                   edgecolors='none', c='#2c3e50')
        
        # Reference line through Q1, Q3
        q1_t, q3_t = np.percentile(theoretical, [25, 75])
        q1_r, q3_r = np.percentile(r, [25, 75])
        slope = (q3_r - q1_r) / (q3_t - q1_t) if q3_t != q1_t else 1
        intercept = q1_r - slope * q1_t
        xlim = np.array([theoretical.min(), theoretical.max()])
        ax.plot(xlim, intercept + slope * xlim, color='#c0392b',
                linewidth=1, linestyle='--')
        
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Deviance residuals')
        ax.set_title('Normal Q-Q')
    
    def _plot_spatial_residuals(self, ax):
        """Map of Pearson residuals at spatial coordinates."""
        coords = self.model.coords
        
        # Symmetric color limits
        vmax = np.percentile(np.abs(self.resid_pearson), 95)
        
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=self.resid_pearson,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        s=20, edgecolors='none', alpha=0.7)
        plt.colorbar(sc, ax=ax, shrink=0.8, label='Pearson residual')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Spatial Residuals')
        ax.set_aspect('equal')
    
    def _plot_observed_vs_fitted(self, ax):
        """Observed Y vs fitted mu with 1:1 line."""
        ax.scatter(self.mu, self.model.Y, alpha=0.4, s=15,
                   edgecolors='none', c='#2c3e50')
        
        lims = [0, max(np.max(self.mu), np.max(self.model.Y)) * 1.05]
        ax.plot(lims, lims, color='#c0392b', linewidth=1, linestyle='--',
                label='1:1')
        ax.legend(fontsize=9, frameon=False)
        
        ax.set_xlabel('Fitted values (μ̂)')
        ax.set_ylabel('Observed counts (Y)')
        ax.set_title('Observed vs Fitted')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    # ------------------------------------------------------------------
    # Variogram
    # ------------------------------------------------------------------
    def plot_variogram(self, n_bins=20, max_dist=None, figsize=(8, 5)):
        """
        Empirical variogram of Pearson residuals overlaid with
        the fitted theoretical variogram from estimated sigma2, rho.
        
        The variogram plots semivariance vs distance. If the spatial model
        is adequate, the empirical points should follow the theoretical curve.
        
        Parameters
        ----------
        n_bins : int
            Number of distance bins.
        max_dist : float or None
            Maximum distance to include. Defaults to half the max pairwise distance.
        """
        m = self.model
        coords = m.coords
        
        # Pairwise distances and semivariances of residuals
        dists = pdist(coords)
        
        # Semivariance: 0.5 * (r_i - r_j)^2 for all pairs
        r = self.resid_pearson
        n = len(r)
        semiv = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                semiv.append(0.5 * (r[i] - r[j]) ** 2)
        semiv = np.array(semiv)
        
        if max_dist is None:
            max_dist = np.percentile(dists, 60)
        
        mask = dists <= max_dist
        dists_use = dists[mask]
        semiv_use = semiv[mask]
        
        # Bin
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        bin_centers = []
        bin_semiv = []
        bin_counts = []
        
        for k in range(n_bins):
            in_bin = (dists_use >= bin_edges[k]) & (dists_use < bin_edges[k + 1])
            if np.sum(in_bin) > 30:  # need enough pairs
                bin_centers.append(0.5 * (bin_edges[k] + bin_edges[k + 1]))
                bin_semiv.append(np.mean(semiv_use[in_bin]))
                bin_counts.append(np.sum(in_bin))
        
        bin_centers = np.array(bin_centers)
        bin_semiv = np.array(bin_semiv)
        bin_counts = np.array(bin_counts)
        
        # Theoretical variogram from fitted parameters
        # For exponential: gamma(h) = sigma2 * (1 - exp(-h/rho))
        # But this is for the random effects, not the residuals directly.
        # The residual variogram should flatten at the "sill" (total variance)
        # with spatial structure at short distances.
        h_theory = np.linspace(0, max_dist, 200)
        nugget = max(0, np.var(r) - m.sigma2)  # residual non-spatial variance
        
        if m.correlation == 'exponential':
            gamma_theory = nugget + m.sigma2 * (1 - np.exp(-h_theory / m.rho))
        elif m.correlation == 'gaussian':
            gamma_theory = nugget + m.sigma2 * (1 - np.exp(-(h_theory / m.rho) ** 2))
        elif m.correlation == 'matern':
            scaled = np.sqrt(3) * h_theory / m.rho
            gamma_theory = nugget + m.sigma2 * (1 - (1 + scaled) * np.exp(-scaled))
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Point sizes proportional to number of pairs in bin
        sizes = 30 + 150 * (bin_counts / bin_counts.max())
        ax.scatter(bin_centers, bin_semiv, s=sizes, c='#2c3e50',
                   edgecolors='white', linewidth=0.5, zorder=3,
                   label='Empirical')
        ax.plot(h_theory, gamma_theory, color='#c0392b', linewidth=2,
                label=f'Fitted (σ²={m.sigma2:.3f}, ρ={m.rho:.2f})')
        
        ax.axhline(np.var(r), color='grey', linewidth=1, linestyle=':',
                   label='Sample variance')
        
        ax.set_xlabel('Distance')
        ax.set_ylabel('Semivariance')
        ax.set_title('Empirical Variogram of Residuals')
        ax.legend(fontsize=9, frameon=False)
        ax.set_xlim(0, max_dist)
        ax.set_ylim(0, None)
        
        return fig
    
    # ------------------------------------------------------------------
    # Convergence traces
    # ------------------------------------------------------------------
    @staticmethod
    def plot_convergence(history, figsize=(12, 8)):
        """
        Plot parameter traces across PQL iterations.
        
        Parameters
        ----------
        history : dict
            Keys: 'beta' (list of arrays), 'sigma2' (list), 'rho' (list),
                  'b_std' (list), 'b_range' (list of [min, max] pairs).
            Populated by adding tracking to the fit() loop — see example below.
        
        Example — add this inside fit() before the convergence check::
        
            if not hasattr(self, '_history'):
                self._history = {'beta': [], 'sigma2': [], 'rho': [],
                                 'b_std': [], 'b_range': []}
            self._history['beta'].append(self.beta.copy())
            self._history['sigma2'].append(self.sigma2)
            self._history['rho'].append(self.rho)
            self._history['b_std'].append(np.std(self.b))
            self._history['b_range'].append([np.min(self.b), np.max(self.b)])
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Convergence Diagnostics', fontsize=14,
                     fontweight='bold', y=0.98)
        iters = np.arange(1, len(history['sigma2']) + 1)
        
        # Fixed effects
        ax = axes[0, 0]
        betas = np.array(history['beta'])
        if betas.ndim == 2 and betas.shape[1] > 0:
            for j in range(betas.shape[1]):
                ax.plot(iters, betas[:, j], linewidth=1.5, label=f'β[{j}]')
            ax.legend(fontsize=8, frameon=False)
        ax.set_ylabel('Value')
        ax.set_title('Fixed Effects (β)')
        ax.set_xlabel('Iteration')
        
        # Variance components
        ax = axes[0, 1]
        ax.plot(iters, history['sigma2'], linewidth=1.5, color='#2c3e50',
                label='σ²')
        ax.set_ylabel('σ²', color='#2c3e50')
        ax.set_title('Variance Components')
        ax.set_xlabel('Iteration')
        ax2 = ax.twinx()
        ax2.plot(iters, history['rho'], linewidth=1.5, color='#c0392b',
                 label='ρ')
        ax2.set_ylabel('ρ', color='#c0392b')
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False)
        
        # Random effects summary
        ax = axes[1, 0]
        ax.plot(iters, history['b_std'], linewidth=1.5, color='#2c3e50',
                label='std(b)')
        b_ranges = np.array(history['b_range'])
        ax.fill_between(iters, b_ranges[:, 0], b_ranges[:, 1],
                         alpha=0.15, color='#2c3e50', label='range(b)')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.legend(fontsize=8, frameon=False)
        ax.set_ylabel('Value')
        ax.set_title('Random Effects (b)')
        ax.set_xlabel('Iteration')
        
        # Parameter changes (convergence rate)
        ax = axes[1, 1]
        if len(iters) > 1:
            beta_changes = np.max(np.abs(np.diff(betas, axis=0)), axis=1) \
                           if betas.ndim == 2 and betas.shape[1] > 0 \
                           else np.zeros(len(iters) - 1)
            sigma2_arr = np.array(history['sigma2'])
            rho_arr = np.array(history['rho'])
            s2_changes = np.abs(np.diff(sigma2_arr)) / np.clip(sigma2_arr[:-1], 1e-8, None)
            rho_changes = np.abs(np.diff(rho_arr)) / np.clip(rho_arr[:-1], 1e-8, None)
            b_std_changes = np.abs(np.diff(history['b_std']))
            
            change_iters = iters[1:]
            ax.semilogy(change_iters, beta_changes + 1e-12, linewidth=1.2,
                        label='max|Δβ|')
            ax.semilogy(change_iters, s2_changes + 1e-12, linewidth=1.2,
                        label='|Δσ²|/σ²')
            ax.semilogy(change_iters, rho_changes + 1e-12, linewidth=1.2,
                        label='|Δρ|/ρ')
            ax.legend(fontsize=8, frameon=False)
        ax.set_ylabel('Change (log scale)')
        ax.set_title('Convergence Rate')
        ax.set_xlabel('Iteration')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
    
    # ------------------------------------------------------------------
    # Spatial effects panel
    # ------------------------------------------------------------------
    def plot_spatial(self, figsize=(13, 5)):
        """
        1x3 panel:
          [0] Estimated random effects b
          [1] Estimated capture probability p
          [2] Pearson residuals (spatial)
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Spatial Effects', fontsize=14, fontweight='bold', y=1.02)
        coords = self.model.coords
        
        # Random effects
        ax = axes[0]
        vmax = np.percentile(np.abs(self.b_hat), 95)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=self.b_hat,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        s=20, edgecolors='none', alpha=0.7)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title('Random Effects (b̂)')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Capture probability
        ax = axes[1]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=self.p_hat,
                        cmap='plasma', s=20, edgecolors='none', alpha=0.7)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title('Capture Prob (p̂)')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Residuals
        ax = axes[2]
        vmax = np.percentile(np.abs(self.resid_pearson), 95)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=self.resid_pearson,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        s=20, edgecolors='none', alpha=0.7)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title('Residuals')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        plt.tight_layout()
        return fig
