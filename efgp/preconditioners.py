"""Preconditioners for the Newton system ``A = I + Phi^T H Phi``.

A preconditioner is a function ``(model, coefficients, trial, hessian_blocks)`` that
returns a positive definite lineax operator ``P`` with ``P ~ A^{-1}``, or ``None``
for no preconditioning. :class:`efgp.NewtonCG` passes ``P`` to ``lineax.CG`` through
``options={"preconditioner": P}``.

Choice of the default
---------------------
``A`` couples coefficients in two ways: modes of the same latent (the Fourier
modes are not orthogonal on a single trial window), and latents that share a
time point (``H_t[k, l]``). On the synthetic data of ``core/gen_synthetic_data.py``
the first coupling dominates:

==================  ===============  ===============  ================
preconditioner      cond(PA), T=50   CG steps, T=50   CG steps, T=400
==================  ===============  ===============  ================
none                1.8e2            73               195
diagonal (Jacobi)   4.2e2            107              268
latent blocks       1.5e1            34               88
==================  ===============  ===============  ================

(CG to relative tolerance 1e-10, at the MAP, 10 latents.)

so the default is :func:`latent_block_preconditioner`, the exact inverse of the
``(2M + 1) x (2M + 1)`` diagonal block of ``A`` for each latent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jax_finufft import nufft1

from .operators import NUFFT_EPS, NUFFT_OPTS


def no_preconditioner(model, coefficients, trial, hessian_blocks):
    """Plain CG. ``A`` has every eigenvalue >= 1, so this works when rates are low."""
    return None


def latent_block_preconditioner(model, coefficients, trial, hessian_blocks):
    """``P = blockdiag_k(A_kk)^{-1}``: the exact within-latent blocks of ``A``, inverted.

    For latent ``k`` with ``h(t) = H_t[k, k]`` and ``x(t)`` its phase, the block in the
    unweighted cos/sin basis follows from product-to-sum formulas:

    .. code-block:: text

        cos_j cos_i  ->  (Cq[j - i] + Cq[j + i]) / 2
        sin_j sin_i  ->  (Cq[j - i] - Cq[j + i]) / 2
        cos_j sin_i  ->  (Sq[j + i] - Sq[j - i]) / 2

    with ``Cq[q] = sum_t h(t) cos(q x(t))`` and ``Sq[q] = sum_t h(t) sin(q x(t))``.
    All of them come from one type-1 NUFFT of ``h`` on frequencies ``-2M..2M``. Then
    each block is weighted, the identity is added, and the block is factored with
    Cholesky. Coupling between different latents is ignored.

    Cost: one NUFFT plus ``K`` Cholesky factorisations of size ``2M + 1``.
    """
    prior = model.prior
    n_freq = prior.n_frequencies
    h_diag = jnp.diagonal(hessian_blocks, axis1=1, axis2=2).T  # (K, T)
    phases = prior.phases(trial.times)

    spectrum = nufft1(
        4 * n_freq + 1,
        h_diag.astype(jnp.result_type(h_diag, 1j)),
        phases,
        iflag=-1,
        eps=NUFFT_EPS,
        opts=NUFFT_OPTS,
    )  # (K, 4M + 1); entry q + 2M is sum_t h e^{-i q x}
    cq = spectrum.real
    sq = -spectrum.imag

    j = jnp.arange(n_freq + 1)
    diff = 2 * n_freq + (j[:, None] - j[None, :])  # index of q = j - i
    total = 2 * n_freq + (j[:, None] + j[None, :])  # index of q = j + i
    cos_cos = (cq[:, diff] + cq[:, total]) / 2
    sin_sin = (cq[:, diff] - cq[:, total]) / 2
    cos_sin = (sq[:, total] - sq[:, diff]) / 2

    block = jnp.concatenate(
        [
            jnp.concatenate([cos_cos, cos_sin[:, :, 1:]], axis=2),
            jnp.concatenate([jnp.swapaxes(cos_sin, 1, 2)[:, 1:, :], sin_sin[:, 1:, 1:]], axis=2),
        ],
        axis=1,
    )  # (K, C, C), unweighted
    weights = jnp.concatenate([prior.cos_weights, prior.sin_weights], axis=1)  # (K, C)
    block = weights[:, :, None] * block * weights[:, None, :]
    block = block + jnp.eye(prior.n_coefficients)
    chol = jnp.linalg.cholesky(block)

    def apply_inverse(v):
        return jax.vmap(lambda c, b: jsl.cho_solve((c, True), b))(chol, v)

    structure = jax.ShapeDtypeStruct(prior.coefficient_shape, h_diag.dtype)
    return lx.FunctionLinearOperator(
        apply_inverse, structure, tags=(lx.symmetric_tag, lx.positive_semidefinite_tag)
    )
