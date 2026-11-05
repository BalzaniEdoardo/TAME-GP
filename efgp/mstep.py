"""M-step for the GP time constants, in JAX.

Each latent ``k`` has a squared-exponential prior over the time points of a trial,

.. code-block:: text

    K(tau_k)[t, s] = variance * exp(-(t - s)^2 / (2 tau_k^2)) + nugget * delta_ts,

and the M-step maximises the expected log prior under the posterior ``q``:

.. code-block:: text

    L_k(tau_k) = -1/2 sum_trials [ logdet K(tau_k) + tr(K(tau_k)^{-1} E_q[z_k z_k^T]) ].

This is the objective of ``core/learnGPParams.py`` (``all_trial_GPLL``), computed
with three simplifications:

1. **One problem per latent.** ``L = sum_k L_k`` and ``L_k`` depends only on
   ``tau_k``, so all time constants are fitted at once as independent problems.
2. **Trials with the same time grid are summed first.** ``L_k`` depends on the
   posterior only through ``E_q[z_k z_k^T] = cov_k + mean_k mean_k^T``, which is
   linear, so trials that share their time points are reduced to one summed
   second moment and a count (:func:`timescale_statistics`). Each evaluation then
   costs one ``T x T`` Cholesky per latent and per distinct time grid.
3. **Bounds by a change of variables.** ``log tau = log lo + (log hi - log lo) *
   sigmoid(u)`` maps the unconstrained ``u`` into ``[lo, hi]``, and nemos' ``Newton``
   minimises over ``u``.
"""

from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jaxtyping import Array, Float
from nemos.regularizer import UnRegularized
from nemos.solvers import Newton


class TimescaleStatistics(eqx.Module):
    """Summed posterior second moments of the trials that share one time grid.

    Attributes
    ----------
    times :
        The shared time stamps, shape ``(T,)``.
    n_trials :
        Number of trials summed.
    second_moment :
        ``sum_trials (cov_k + mean_k mean_k^T)`` for each latent, shape ``(K, T, T)``.
    """

    times: Float[Array, " n_time"]
    n_trials: Float[Array, ""]
    second_moment: Float[Array, "n_latents n_time n_time"]


def timescale_statistics(
    times: Sequence[Array],
    means: Sequence[Array],
    covariances: Sequence[Array],
) -> list[TimescaleStatistics]:
    """Group trials by time grid and sum their posterior second moments.

    Parameters
    ----------
    times :
        Time stamps of each trial, each of shape ``(T_i,)``.
    means :
        Posterior mean of each trial, each of shape ``(K, T_i)``.
    covariances :
        Posterior covariance over time of each latent, each of shape ``(K, T_i, T_i)``.
    """
    groups: dict[bytes, list] = {}
    for t, m, c in zip(times, means, covariances):
        t = np.asarray(t, dtype=float)
        m, c = np.asarray(m), np.asarray(c)
        moment = c + m[:, :, None] * m[:, None, :]
        key = t.tobytes()
        if key in groups:
            groups[key][1] += 1
            groups[key][2] += moment
        else:
            groups[key] = [t, 1, moment]
    return [
        TimescaleStatistics(jnp.asarray(t), jnp.asarray(float(n)), jnp.asarray(s))
        for t, n, s in groups.values()
    ]


def expected_log_prior(
    tau: Float[Array, " n_latents"],
    statistics: Sequence[TimescaleStatistics],
    variance: float,
    nugget: float,
) -> Float[Array, " n_latents"]:
    """``L_k(tau_k)`` for every latent, summed over trials (no ``2 pi`` constant)."""
    total = jnp.zeros_like(tau)
    for group in statistics:  # one iteration per distinct time grid
        sq_dist = (group.times[:, None] - group.times[None, :]) ** 2
        kernel = variance * jnp.exp(-sq_dist / (2 * tau[:, None, None] ** 2))
        kernel = kernel + nugget * jnp.eye(group.times.shape[0])
        chol = jnp.linalg.cholesky(kernel)  # (K, T, T)
        logdet = 2 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)), axis=1)
        solved = jax.vmap(lambda c, s: jsl.cho_solve((c, True), s))(chol, group.second_moment)
        trace = jnp.trace(solved, axis1=1, axis2=2)
        total = total - 0.5 * (group.n_trials * logdet + trace)
    return total


def _tau_from_u(u, log_lo, log_hi):
    return jnp.exp(log_lo + (log_hi - log_lo) * jax.nn.sigmoid(u))


def _u_from_tau(tau, log_lo, log_hi):
    frac = (jnp.log(tau) - log_lo) / (log_hi - log_lo)
    frac = jnp.clip(frac, 1e-6, 1 - 1e-6)  # a bound itself maps to u = +-infinity
    return jnp.log(frac) - jnp.log1p(-frac)


def _negative_objective(u, statistics, log_lo, log_hi, variance, nugget, n_trials):
    tau = _tau_from_u(u, log_lo, log_hi)
    return -jnp.sum(expected_log_prior(tau, statistics, variance, nugget)) / n_trials


def _diagonal_hessian(u, *args):
    """The objective is a sum of one-variable terms, so its Hessian is diagonal and
    one Hessian-vector product with a vector of ones gives all of it (instead of one
    pass per latent with ``jax.hessian``)."""
    grad = jax.grad(_negative_objective)
    _, diag = jax.jvp(lambda x: grad(x, *args), (u,), (jnp.ones_like(u),))
    return jnp.diag(diag)


class TimescaleFitter:
    """Bounded fit of all GP time constants, with a solver that compiles once.

    Parameters
    ----------
    tau_bounds :
        ``(lower, upper)``, shared by all latents.
    variance, nugget :
        Kernel ``variance * SE + nugget * I``.
    maxiter, tol :
        For nemos' ``Newton``, on the unconstrained parameters.
    """

    def __init__(self, tau_bounds, variance: float, nugget: float, maxiter: int = 100, tol: float = 1e-10):
        self.log_lo, self.log_hi = (float(np.log(b)) for b in tau_bounds)
        self.variance = float(variance)
        self.nugget = float(nugget)
        self.maxiter = maxiter
        self.tol = tol
        self._solver = None

    def __call__(self, tau_init: Array, statistics: Sequence[TimescaleStatistics]):
        """Return ``(tau, converged)``, ``tau`` of shape ``(n_latents,)``."""
        u0 = _u_from_tau(jnp.asarray(tau_init, dtype=float), self.log_lo, self.log_hi)
        if self._solver is None:
            self._solver = Newton(
                _negative_objective,
                UnRegularized(),
                None,
                False,
                init_params=u0,
                maxiter=self.maxiter,
                tol=self.tol,
                hess_fn=_diagonal_hessian,
            )
        n_trials = sum(float(g.n_trials) for g in statistics)
        u, state, _ = self._solver.run(
            u0, list(statistics), self.log_lo, self.log_hi, self.variance, self.nugget, n_trials
        )
        return _tau_from_u(u, self.log_lo, self.log_hi), state.stats.converged
