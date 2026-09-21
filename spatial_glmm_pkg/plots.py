"""
Diagnostic and spatial plots for Spatial GLMM.

Standard GLMM diagnostics:
    - QQ plot of residuals
    - Residuals vs fitted values
    - Scale-location plot

Spatial plots:
    - Random effects (b) in space
    - Fitted values (mu) in space
    - Observed counts (Y) in space
    - Residuals (Y - mu) in space
    - Capture probability (p) in space
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ======================================================================
# Standard GLMM Diagnostics
# ======================================================================

def qq_plot(model, residual_type='pearson', ax=None):
    """
    QQ plot of residuals against standard normal.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    residual_type : str
        'pearson' or 'deviance'.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    mu, _ = model._compute_current_mu(model.beta, model.b)

    if residual_type == 'pearson':
        resid = model.family.pearson_residuals(model.Y, mu)
        label = 'Pearson residuals'
    elif residual_type == 'deviance':
        resid = model.family.deviance_residuals(model.Y, mu)
        label = 'Deviance residuals'
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    sorted_resid = np.sort(resid)
    n = len(sorted_resid)
    theoretical = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    ax.scatter(theoretical, sorted_resid, alpha=0.5, s=10, color='steelblue')

    # Reference line
    q25, q75 = np.percentile(sorted_resid, [25, 75])
    t25, t75 = stats.norm.ppf([0.25, 0.75])
    slope = (q75 - q25) / (t75 - t25)
    intercept = q25 - slope * t25
    xlim = np.array([theoretical.min(), theoretical.max()])
    ax.plot(xlim, intercept + slope * xlim, 'r--', linewidth=1)

    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel(label)
    ax.set_title(f'QQ Plot ({residual_type.capitalize()} Residuals)')

    return fig, ax


def residuals_vs_fitted(model, residual_type='pearson', ax=None):
    """
    Residuals vs fitted values plot.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    residual_type : str
        'pearson' or 'deviance'.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    mu, _ = model._compute_current_mu(model.beta, model.b)

    if residual_type == 'pearson':
        resid = model.family.pearson_residuals(model.Y, mu)
        label = 'Pearson residuals'
    elif residual_type == 'deviance':
        resid = model.family.deviance_residuals(model.Y, mu)
        label = 'Deviance residuals'
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.scatter(mu, resid, alpha=0.5, s=10, color='steelblue')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)

    ax.set_xlabel('Fitted values (μ)')
    ax.set_ylabel(label)
    ax.set_title(f'{label} vs Fitted Values')

    return fig, ax


def scale_location(model, ax=None):
    """
    Scale-location plot: sqrt(|standardized residuals|) vs fitted values.
    Checks for heteroscedasticity.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    mu, _ = model._compute_current_mu(model.beta, model.b)
    resid = model.family.pearson_residuals(model.Y, mu)
    std_resid = resid / np.std(resid)
    sqrt_abs_resid = np.sqrt(np.abs(std_resid))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.scatter(mu, sqrt_abs_resid, alpha=0.5, s=10, color='steelblue')

    ax.set_xlabel('Fitted values (μ)')
    ax.set_ylabel('√|Standardized residuals|')
    ax.set_title('Scale-Location Plot')

    return fig, ax


def diagnostics(model, residual_type='pearson'):
    """
    Four-panel diagnostic plot.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    residual_type : str
        'pearson' or 'deviance'.

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    qq_plot(model, residual_type=residual_type, ax=axes[0])
    residuals_vs_fitted(model, residual_type=residual_type, ax=axes[1])
    scale_location(model, ax=axes[2])

    fig.tight_layout()
    return fig, axes


# ======================================================================
# Spatial Plots
# ======================================================================

def _spatial_scatter(coords, values, title, cmap, ax, colorbar_label, vmin=None, vmax=None):
    """Helper for spatial scatter plots."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=values, cmap=cmap,
        s=15, alpha=0.8, edgecolors='none',
        vmin=vmin, vmax=vmax
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(colorbar_label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    return fig, ax


def plot_random_effects(model, ax=None):
    """
    Plot random effects b in space.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    # Symmetric colorbar around zero
    vmax = np.max(np.abs(model.b))
    return _spatial_scatter(
        model.coords, model.b,
        title='Random Effects (b)',
        cmap='RdBu_r', ax=ax,
        colorbar_label='b',
        vmin=-vmax, vmax=vmax
    )


def plot_fitted(model, ax=None):
    """
    Plot fitted values mu in space.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    mu, _ = model._compute_current_mu(model.beta, model.b)
    return _spatial_scatter(
        model.coords, mu,
        title='Fitted Values (μ)',
        cmap='viridis', ax=ax,
        colorbar_label='μ'
    )


def plot_observed(model, ax=None):
    """
    Plot observed counts Y in space.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    return _spatial_scatter(
        model.coords, model.Y,
        title='Observed Counts (Y)',
        cmap='viridis', ax=ax,
        colorbar_label='Y'
    )


def plot_residuals_spatial(model, residual_type='raw', ax=None):
    """
    Plot residuals in space.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    residual_type : str
        'raw' for Y - mu, 'pearson', or 'deviance'.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    mu, _ = model._compute_current_mu(model.beta, model.b)

    if residual_type == 'raw':
        resid = model.Y - mu
        label = 'Y - μ'
    elif residual_type == 'pearson':
        resid = model.family.pearson_residuals(model.Y, mu)
        label = 'Pearson residuals'
    elif residual_type == 'deviance':
        resid = model.family.deviance_residuals(model.Y, mu)
        label = 'Deviance residuals'
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")

    vmax = np.max(np.abs(resid))
    return _spatial_scatter(
        model.coords, resid,
        title=f'Spatial Residuals ({label})',
        cmap='RdBu_r', ax=ax,
        colorbar_label=label,
        vmin=-vmax, vmax=vmax
    )


def plot_capture_probability(model, ax=None):
    """
    Plot capture probability p = expit(a + b) in space.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    from scipy.special import expit
    p = expit(model.a + model.b)
    return _spatial_scatter(
        model.coords, p,
        title='Capture Probability (p)',
        cmap='YlOrRd', ax=ax,
        colorbar_label='p',
        vmin=0, vmax=min(1.0, p.max() * 1.1)
    )


def spatial_summary(model):
    """
    Six-panel spatial summary: observed, fitted, residuals,
    random effects, capture probability, and QQ plot.

    Parameters
    ----------
    model : SpatialCustomLinkPQL
        Fitted model.

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_observed(model, ax=axes[0, 0])
    plot_fitted(model, ax=axes[0, 1])
    plot_residuals_spatial(model, residual_type='raw', ax=axes[0, 2])
    plot_random_effects(model, ax=axes[1, 0])
    plot_capture_probability(model, ax=axes[1, 1])
    qq_plot(model, ax=axes[1, 2])

    fig.tight_layout()
    return fig, axes