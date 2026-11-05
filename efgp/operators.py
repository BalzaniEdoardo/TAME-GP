"""Matrix-free linear operators for EFGP inference, written as lineax operators.

The three building blocks are

* ``Phi`` (:func:`fourier_features`): Fourier coefficients ``(K, C)`` to latent
  values on the trial time points ``(T, K)``. One type-2 NUFFT per latent.
* ``Phi^T`` (:func:`fourier_features_transpose`): the exact adjoint, one type-1 NUFFT.
* ``H`` (:func:`block_diagonal_operator`): a ``K x K`` block per time point, used for
  the negative log-likelihood Hessian.

They compose with the usual lineax algebra (``@``, ``+``, scalar ``*``), for example
the Newton system of the weight-space posterior,
``I + Phi^T @ H @ Phi`` (:func:`newton_operator`).

The adjoint is written out with a type-1 NUFFT instead of taken from
``Phi.T``: lineax would transpose ``Phi`` with ``jax.linear_transpose``. That is
correct, but the explicit version shows the math and can be tested on its own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jax_finufft import nufft1, nufft2
from jax_finufft.options import Opts
from jaxtyping import Array, Float

from .prior import EFGPPrior

#: Default relative accuracy of every NUFFT. Keep it well below the kernel
#: tolerance of the prior so the NUFFT error never dominates.
NUFFT_EPS = 1e-12

#: FINUFFT options for every call. One thread: the problems here are small (a few
#: thousand points), and with the default multithreading each call spends about
#: 3 ms on thread start-up, against 0.3 ms single-threaded (measured on Apple M-series
#: CPU, 10 latents, 1000 time points).
NUFFT_OPTS = Opts(nthreads=1)


def _coefficients_to_modes(prior: EFGPPrior, coefficients: Array) -> Array:
    """Real ``[a_0..a_M, b_1..b_M]`` to Hermitian complex modes ``k = -M..M``.

    ``a cos(jx) + b sin(jx) = Re[(a - i b) e^{ijx}]``, and splitting each mode
    ``j > 0`` evenly between ``+j`` and ``-j`` makes the sum real.
    """
    n_freq = prior.n_frequencies
    a = coefficients[:, : n_freq + 1] * prior.cos_weights
    b = coefficients[:, n_freq + 1 :] * prior.sin_weights
    g = a.astype(jnp.result_type(a, 1j))
    g = g.at[:, 1:].add(-1j * b)
    negative = jnp.conj(g[:, 1:])[:, ::-1] / 2
    return jnp.concatenate([negative, g[:, :1], g[:, 1:] / 2], axis=1)


def _coefficient_structure(prior: EFGPPrior, dtype) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(prior.coefficient_shape, dtype)


def fourier_features(
    prior: EFGPPrior,
    times: Float[Array, " n_time"],
    *,
    eps: float = NUFFT_EPS,
) -> lx.FunctionLinearOperator:
    """The EFGP feature map ``Phi``: coefficients ``(K, C)`` to latents ``(T, K)``.

    Parameters
    ----------
    prior :
        Frequency grid and weights of every latent.
    times :
        Time stamps of one trial. They need not be equispaced; time points with
        missing data can simply be left out, or kept and masked in the likelihood.
    eps :
        NUFFT relative accuracy.
    """
    times = jnp.asarray(times)
    phases = prior.phases(times)  # (K, T)

    def matvec(coefficients):
        modes = _coefficients_to_modes(prior, coefficients)
        return nufft2(modes, phases, iflag=1, eps=eps, opts=NUFFT_OPTS).real.T

    return lx.FunctionLinearOperator(matvec, _coefficient_structure(prior, times.dtype))


def fourier_features_transpose(
    prior: EFGPPrior,
    times: Float[Array, " n_time"],
    *,
    eps: float = NUFFT_EPS,
) -> lx.FunctionLinearOperator:
    """The adjoint ``Phi^T``: latent-space vectors ``(T, K)`` to coefficients ``(K, C)``."""
    times = jnp.asarray(times)
    phases = prior.phases(times)
    n_freq = prior.n_frequencies

    def matvec(values):
        sources = values.T.astype(jnp.result_type(values, 1j))  # (K, T)
        # F_j = sum_t r(t) e^{-i j x(t)} = sum_t r cos(jx) - i sum_t r sin(jx)
        spectrum = nufft1(2 * n_freq + 1, sources, phases, iflag=-1, eps=eps, opts=NUFFT_OPTS)
        spectrum = spectrum[:, n_freq:]  # j = 0..M
        a = prior.cos_weights * spectrum.real
        b = -prior.sin_weights * spectrum[:, 1:].imag
        return jnp.concatenate([a, b], axis=1)

    structure = jax.ShapeDtypeStruct((times.shape[0], prior.n_latents), times.dtype)
    return lx.FunctionLinearOperator(matvec, structure)


def block_diagonal_operator(
    blocks: Float[Array, "n_time n_latents n_latents"],
    *,
    positive_semidefinite: bool = True,
) -> lx.FunctionLinearOperator:
    """``(T, K)`` to ``(T, K)``, multiplying time point ``t`` by ``blocks[t]``.

    Tagged symmetric (and positive semidefinite, by default), which is what
    the negative Hessian of a log-concave likelihood is.
    """
    tags = [lx.symmetric_tag]
    if positive_semidefinite:
        tags.append(lx.positive_semidefinite_tag)
    structure = jax.ShapeDtypeStruct(blocks.shape[:2], blocks.dtype)
    return lx.FunctionLinearOperator(
        lambda z: jnp.einsum("tkl,tl->tk", blocks, z), structure, tags=tags
    )


def newton_operator(
    features: lx.AbstractLinearOperator,
    features_transpose: lx.AbstractLinearOperator,
    likelihood_hessian: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Weight-space posterior precision ``I + Phi^T H Phi``, tagged positive definite.

    The identity is the ``N(0, I)`` prior on the coefficients, so every eigenvalue
    is at least one.
    """
    identity = lx.IdentityLinearOperator(features.in_structure())
    precision = identity + features_transpose @ likelihood_hessian @ features
    return lx.TaggedLinearOperator(
        precision, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )
