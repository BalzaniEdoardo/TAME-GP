"""Observation models: how one group of observed variables depends on the latents.

Every block depends only on the latent groups listed in ``latents`` and has

* ``log_likelihood(z, y, mask)``: summed over observed entries, up to constants in ``z``;
* ``negative_hessian(z, y, mask)``: one ``K_b x K_b`` block per time point. Both
  models below are log-concave in ``z``, so every block is positive semidefinite.

``z`` has shape ``(T, K_b)``: the columns of the latent groups in ``latents``, in
that order. ``y`` has shape ``(T, D)``. ``mask`` is a boolean ``(T, D)`` array that is
``False`` for missing entries. A missing entry adds nothing to the likelihood. The
model builds ``mask`` from the NaNs in ``y`` and from trial padding.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class ObservationBlock(eqx.Module):
    """Interface of an observation model. Subclass it to add a new noise model.

    Subclasses must have a ``loadings`` array of shape ``(D, K_b)`` and a static
    ``latents`` field: the tuple of latent group names that make up the ``K_b`` columns.
    """

    @abc.abstractmethod
    def log_likelihood(
        self,
        z: Float[Array, "n_time n_lat"],
        y: Float[Array, "n_time n_obs"],
        mask: Bool[Array, "n_time n_obs"],
    ) -> Float[Array, ""]: ...

    @abc.abstractmethod
    def negative_hessian(
        self,
        z: Float[Array, "n_time n_lat"],
        y: Float[Array, "n_time n_obs"],
        mask: Bool[Array, "n_time n_obs"],
    ) -> Float[Array, "n_time n_lat n_lat"]: ...


class PoissonObservations(ObservationBlock):
    """Spike counts, ``y_ti ~ Poisson(exp(loadings_i . z_t + offset_i))``.

    Attributes
    ----------
    loadings :
        Shape ``(n_units, K_b)``.
    offset :
        Log baseline rate of each unit, shape ``(n_units,)``.
    latents :
        Names of the latent groups the units load on, in column order of ``loadings``.
    """

    loadings: Float[Array, "n_obs n_lat"]
    offset: Float[Array, " n_obs"]
    latents: tuple[str, ...] = eqx.field(static=True)

    def _log_rate(self, z):
        return z @ self.loadings.T + self.offset

    def log_likelihood(self, z, y, mask):
        log_rate = self._log_rate(z)
        y = jnp.where(mask, y, 0.0)
        return jnp.sum(jnp.where(mask, y * log_rate - jnp.exp(log_rate), 0.0))

    def negative_hessian(self, z, y, mask):
        rate = jnp.where(mask, jnp.exp(self._log_rate(z)), 0.0)  # (T, N)
        return jnp.einsum("ik,ti,il->tkl", self.loadings, rate, self.loadings)


class GaussianObservations(ObservationBlock):
    """Continuous variables, ``y_t ~ N(loadings . z_t + offset, covariance)``.

    Missing entries are handled exactly: at each time point the likelihood is the
    marginal Gaussian of the observed entries only, with precision
    ``inv(covariance[obs, obs])``.

    Attributes
    ----------
    loadings :
        Shape ``(n_vars, K_b)``.
    offset :
        Shape ``(n_vars,)``.
    covariance :
        Noise covariance, shape ``(n_vars, n_vars)``.
    latents :
        Names of the latent groups the variables load on.
    """

    loadings: Float[Array, "n_obs n_lat"]
    offset: Float[Array, " n_obs"]
    covariance: Float[Array, "n_obs n_obs"]
    latents: tuple[str, ...] = eqx.field(static=True)

    def observed_precision(self, mask: Bool[Array, "n_time n_obs"]) -> Array:
        """``inv(covariance[obs, obs])`` at each time point, zero-padded to ``(T, D, D)``."""
        m = mask.astype(self.covariance.dtype)
        outer = m[:, :, None] * m[:, None, :]
        # Replace the rows and columns of missing entries by the identity, invert,
        # then zero them again. The observed block of the result is inv(cov_oo).
        padded = self.covariance * outer + jnp.eye(m.shape[1]) * (1.0 - m)[:, None, :]
        return jnp.linalg.inv(padded) * outer

    def log_likelihood(self, z, y, mask):
        residual = jnp.where(mask, y - z @ self.loadings.T - self.offset, 0.0)
        precision = self.observed_precision(mask)
        return -0.5 * jnp.einsum("ti,tij,tj->", residual, precision, residual)

    def negative_hessian(self, z, y, mask):
        precision = self.observed_precision(mask)
        return jnp.einsum("ik,tij,jl->tkl", self.loadings, precision, self.loadings)
