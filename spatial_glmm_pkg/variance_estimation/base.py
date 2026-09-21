from abc import ABC, abstractmethod


class VarianceEstimator(ABC):
    """Abstract base class for variance component estimation."""

    @abstractmethod
    def estimate(self, model):
        """
        Estimate variance components given the current model state.

        Parameters
        ----------
        model : SpatialCustomLinkPQL
            The model object with current beta, b, Y, X, etc.

        Returns
        -------
        sigma2 : float
            Estimated spatial variance.
        rho : float
            Estimated spatial range.
        """
        pass