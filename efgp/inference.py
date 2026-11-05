"""Laplace inference of the latents, one trial at a time or over a batch of trials.

Steps for one trial:

1. Find the MAP coefficients ``beta_hat`` by minimising :func:`negative_log_joint`.
   The default solver is :class:`NewtonCG`; any nemos solver with the same calling
   convention can be passed instead (for example nemos' dense ``Newton`` as a
   reference).
2. The Laplace precision of the coefficients is ``A = I + Phi^T H Phi`` at
   ``beta_hat``. The latents ``z = Phi beta`` then have posterior mean
   ``Phi beta_hat`` and covariance ``Phi A^{-1} Phi^T``.

Covariances
-----------
The EM M-step needs the ``K x K`` covariance at each time point (``cov_t``) and the
``T x T`` covariance of each latent (``cov_k``). Both are computed from a dense
Cholesky factor of ``A``, which has size ``K (2M + 1)``: small compared to the
``K T`` latent values when trials are long. For products with the full covariance,
without forming any matrix, use :func:`posterior_covariance_operator`.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, Float

from .model import LatentModel, Trial, negative_log_joint
from .operators import fourier_features_transpose
from .preconditioners import latent_block_preconditioner
from .solver import NewtonCG, efgp_newton_system


class LaplacePosterior(eqx.Module):
    """Laplace approximation of the latent posterior of one trial.

    Attributes
    ----------
    coefficients :
        MAP Fourier coefficients ``beta_hat``, shape ``(K, 2M + 1)``.
    mean :
        Posterior mean of the latents, shape ``(T, K)``.
    cov_t :
        Covariance between latents at each time point, shape ``(T, K, K)``, or ``None``.
    cov_k :
        Covariance over time of each latent, shape ``(K, T, T)``, or ``None``.
    objective :
        :func:`negative_log_joint` at the MAP.
    num_steps, converged :
        Solver diagnostics.
    """

    coefficients: Float[Array, "n_latents n_coef"]
    mean: Float[Array, "n_time n_latents"]
    cov_t: Float[Array, "n_time n_latents n_latents"] | None
    cov_k: Float[Array, "n_latents n_time n_time"] | None
    objective: Float[Array, ""]
    num_steps: Array
    converged: Array


def newton_cg_solver(
    model: LatentModel,
    *,
    preconditioner: Callable = latent_block_preconditioner,
    cg: lx.CG | None = None,
    maxiter: int = 50,
    tol: float = 1e-8,
    rtol: float = 0.0,
    jit: bool = True,
) -> NewtonCG:
    """The default solver: :class:`NewtonCG` on :func:`negative_log_joint`.

    The instance only depends on the coefficient shape of ``model``, so it can be
    reused for other trials and for other parameter values with the same prior grid.
    """
    return NewtonCG(
        negative_log_joint,
        init_params=model.init_coefficients(),
        hess_fn=efgp_newton_system(preconditioner),
        cg=cg,
        maxiter=maxiter,
        tol=tol,
        rtol=rtol,
        jit=jit,
    )


def infer_trial(
    model: LatentModel,
    trial: Trial,
    *,
    solver=None,
    init_coefficients: Array | None = None,
    covariance: str = "blocks",
) -> LaplacePosterior:
    """Laplace posterior of the latents of one trial.

    Parameters
    ----------
    model :
        Parameters and prior.
    trial :
        Time stamps and observations. All time stamps must lie in the prior domain
        ``[prior.t0, prior.t0 + prior.duration]``.
    solver :
        Any nemos solver built for the objective
        ``negative_log_joint(coefficients, model, trial)``. Default:
        :func:`newton_cg_solver`. Build it once and reuse it: building a new one
        each call triggers a new compilation.
    init_coefficients :
        Starting point, for example the MAP of the previous EM iteration. Default: zeros.
    covariance :
        ``"blocks"`` for ``cov_t`` and ``cov_k``, ``"time"`` for ``cov_t`` only,
        ``"none"`` for neither.
    """
    if covariance not in ("blocks", "time", "none"):
        raise ValueError(f"covariance must be 'blocks', 'time' or 'none', got {covariance!r}.")
    if solver is None:
        solver = newton_cg_solver(model)
    if init_coefficients is None:
        init_coefficients = model.init_coefficients()
    _check_domain(model, trial)

    coefficients, state, _ = solver.run(init_coefficients, model, trial)
    stats = solver._get_optim_info(state)
    mean = model.latents(coefficients, trial)

    cov_t = cov_k = None
    if covariance != "none":
        cov_t, cov_k = laplace_covariance_blocks(
            model, trial, coefficients, with_cov_k=covariance == "blocks"
        )
    return LaplacePosterior(
        coefficients=coefficients,
        mean=mean,
        cov_t=cov_t,
        cov_k=cov_k,
        objective=negative_log_joint(coefficients, model, trial),
        num_steps=stats.num_steps,
        converged=stats.converged,
    )


def infer_trials(
    model: LatentModel,
    trials: Trial,
    *,
    solver=None,
    init_coefficients: Array | None = None,
    covariance: str = "blocks",
) -> LaplacePosterior:
    """:func:`infer_trial` mapped over the leading axis of stacked trials.

    Make ``trials`` with :func:`efgp.stack_trials`. Every field of the returned
    posterior has an extra leading trial axis; rows of padded time points are
    meaningless and should be dropped with ``trials.valid``.
    """
    if solver is None:
        solver = newton_cg_solver(model)
    if init_coefficients is None:
        n_trials = trials.times.shape[0]
        init_coefficients = jnp.zeros((n_trials,) + model.prior.coefficient_shape)
    _check_domain(model, trials)

    def one(trial, init):
        return infer_trial(
            model, trial, solver=solver, init_coefficients=init, covariance=covariance
        )

    return jax.vmap(one)(trials, init_coefficients)


def laplace_covariance_blocks(
    model: LatentModel,
    trial: Trial,
    coefficients: Array,
    *,
    with_cov_k: bool = True,
):
    """``cov_t`` (``(T, K, K)``) and ``cov_k`` (``(K, T, T)`` or ``None``) at ``coefficients``.

    Uses ``Cov[z] = Phi A^{-1} Phi^T = G^T G`` with ``G = L^{-1} Phi^T`` and
    ``A = L L^T``. ``A`` is assembled densely from the per-latent feature matrices;
    its size is ``K (2M + 1)``, independent of the trial length.
    """
    prior = model.prior
    n_lat, n_coef = prior.coefficient_shape
    features = prior.features(trial.times)  # (K, T, C)
    z = model.latents(coefficients, trial)
    blocks = model.negative_hessian(z, trial)  # (T, K, K)

    precision = jnp.einsum("ktc,tkl,lts->kcls", features, blocks, features)
    precision = precision.reshape(n_lat * n_coef, n_lat * n_coef)
    precision = precision + jnp.eye(n_lat * n_coef)
    chol = jnp.linalg.cholesky(precision)

    # Phi^T as a (K*C, T, K) array: row (k, c), column (t, l) holds delta_kl F[k, t, c].
    phi_t = jnp.einsum("ktc,kl->kctl", features, jnp.eye(n_lat))
    phi_t = phi_t.reshape(n_lat * n_coef, -1)
    g = jsl.solve_triangular(chol, phi_t, lower=True)
    g = g.reshape(n_lat * n_coef, trial.n_time, n_lat)

    cov_t = jnp.einsum("ntk,ntl->tkl", g, g)
    cov_k = jnp.einsum("ntk,nsk->kts", g, g) if with_cov_k else None
    return cov_t, cov_k


def posterior_covariance_operator(
    model: LatentModel,
    trial: Trial,
    coefficients: Array,
    *,
    cg: lx.CG | None = None,
    preconditioner: Callable = latent_block_preconditioner,
) -> lx.AbstractLinearOperator:
    """Matrix-free Laplace covariance of the latents, ``Phi A^{-1} Phi^T``.

    A ``(T, K) -> (T, K)`` lineax operator. Each product costs one CG solve with
    ``A`` and two NUFFTs, and no matrix is formed.
    """
    if cg is None:
        cg = lx.CG(rtol=1e-10, atol=1e-10, max_steps=1000)
    z = model.latents(coefficients, trial)
    blocks = model.negative_hessian(z, trial)
    precision = model.precision_operator(trial, blocks)
    options = {}
    p = preconditioner(model, coefficients, trial, blocks)
    if p is not None:
        options["preconditioner"] = p

    def solve(v):
        return lx.linear_solve(precision, v, cg, options=options).value

    precision_inverse = lx.FunctionLinearOperator(
        solve,
        precision.in_structure(),
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    phi = model.features(trial)
    phi_t = fourier_features_transpose(model.prior, trial.times)
    return lx.TaggedLinearOperator(
        phi @ precision_inverse @ phi_t, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )


def _check_domain(model: LatentModel, trial: Trial) -> None:
    """Raise if concrete time stamps fall outside the prior domain (they would alias)."""
    times = trial.times
    if isinstance(times, jax.core.Tracer):
        return
    lo, hi = model.prior.t0, model.prior.t0 + model.prior.duration
    if float(jnp.min(times)) < lo - 1e-12 or float(jnp.max(times)) > hi + 1e-12:
        raise ValueError(
            f"Trial time stamps span [{float(jnp.min(times))}, {float(jnp.max(times))}], "
            f"outside the prior domain [{lo}, {hi}]. Rebuild the prior with a larger "
            "duration: the Fourier representation is periodic and would alias."
        )
