"""Equispaced-Fourier (EFGP) approximation of the squared-exponential GP prior.

Each latent dimension ``k`` is written in weight space as

.. code-block:: text

    z_k(t) = sum_j  a_kj * wc_kj * cos(2 pi j h_k (t - t0))
           + sum_j  b_kj * ws_kj * sin(2 pi j h_k (t - t0)),     a, b ~ N(0, 1)

so that ``Cov[z_k(t), z_k(s)]`` approximates ``variance * exp(-(t-s)^2 / 2 l_k^2)``
to the absolute tolerance ``tol`` on ``|t - s| <= duration``.

The frequency spacing ``h_k``, the number of frequencies ``m_k`` and the weights
come from :class:`nemos.basis.FourierGP` (Barnett, Greengard & Rachh, 2024). Only
these numbers are kept; the dense basis matrix is never built.

Coefficient layout
------------------
All latents share one array of shape ``(n_latents, 2 * M + 1)``, where
``M = max_k m_k``. Row ``k`` holds ``[a_k0, ..., a_kM, b_k1, ..., b_kM]``, the
same column order as ``FourierGP.compute_features``. A latent with ``m_k < M`` has
zero weights on its extra modes, so those coefficients do not affect ``z`` and
stay at zero during inference.
"""

from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from nemos.basis import FourierGP


class EFGPPrior(eqx.Module):
    """Independent EFGP priors for every latent dimension of one model.

    Build it with :meth:`from_lengthscales`; the constructor takes the
    pre-computed grid.

    Attributes
    ----------
    lengthscales :
        SE lengthscale of each latent, shape ``(n_latents,)``. Kept for reference.
    frequency_spacing :
        Frequency spacing ``h_k`` of each latent, shape ``(n_latents,)``.
    cos_weights :
        Prior standard deviation of each cosine coefficient, shape
        ``(n_latents, M + 1)``. Zero on padded modes.
    sin_weights :
        Prior standard deviation of each sine coefficient, shape
        ``(n_latents, M)``. Zero on padded modes.
    t0, duration :
        The time domain ``[t0, t0 + duration]``. Every trial must lie inside it:
        the Fourier representation is periodic, so time points outside the domain
        are silently aliased.
    tol :
        Kernel approximation tolerance that was used to build the grid.
    """

    lengthscales: Float[Array, " n_latents"]
    frequency_spacing: Float[Array, " n_latents"]
    cos_weights: Float[Array, "n_latents n_cos"]
    sin_weights: Float[Array, "n_latents n_sin"]
    t0: float = eqx.field(static=True)
    duration: float = eqx.field(static=True)
    tol: float = eqx.field(static=True)

    @classmethod
    def from_lengthscales(
        cls,
        lengthscales: Sequence[float],
        duration: float,
        *,
        t0: float = 0.0,
        variance: float = 1.0,
        tol: float = 1e-6,
        lengthscale_bounds: tuple[float, float] | None = None,
    ) -> "EFGPPrior":
        """Build the frequency grid of every latent with :class:`nemos.basis.FourierGP`.

        Parameters
        ----------
        lengthscales :
            One SE lengthscale per latent dimension, in the units of the trial
            time stamps (for example seconds).
        duration :
            Length of the time domain. Use the longest trial duration.
        t0 :
            Start of the time domain.
        variance :
            SE kernel variance, shared by all latents.
        tol :
            Absolute tolerance on the kernel approximation. Smaller values add
            frequencies; the count grows like ``sqrt(log(1 / tol))``.
        lengthscale_bounds :
            ``(lower, upper)``: the range the lengthscales may take, for example
            while they are fitted with EM. The coefficient array is then sized for
            the worst case, the lower bound, which needs the most frequencies, so
            its shape (and the compiled solver) is the same for every lengthscale
            in the range. Latents with longer lengthscales are padded with
            zero-weight modes, which leaves their prior unchanged. Lengthscales
            outside the range raise a ``ValueError``.

        Notes
        -----
        This runs eagerly, outside ``jit``: the number of frequencies is a Python
        integer that depends on the lengthscale. Rebuild the prior whenever the
        lengthscales change.
        """
        lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)
        if duration <= 0:
            raise ValueError(f"``duration`` must be positive, got {duration}.")
        if lengthscale_bounds is not None:
            lower, upper = lengthscale_bounds
            outside = (lengthscales < lower) | (lengthscales > upper)
            if outside.any():
                raise ValueError(
                    f"Lengthscales {lengthscales[outside]} are outside "
                    f"lengthscale_bounds={lengthscale_bounds}."
                )

        spacings, cos_w, sin_w = [], [], []
        for lengthscale in lengthscales:
            basis = FourierGP(
                lengthscale=float(lengthscale),
                bounds=(t0, t0 + duration),
                eps=tol,
                variance=variance,
            )
            m = basis.n_frequencies
            weights = np.asarray(basis.weights)  # [cos 0..m, sin 1..m]
            spacings.append(basis.frequency_spacing)
            cos_w.append(weights[: m + 1])
            sin_w.append(weights[m + 1 :])

        n_max = max(len(w) for w in cos_w) - 1  # M
        if lengthscale_bounds is not None:
            worst_case = FourierGP(
                lengthscale=float(lengthscale_bounds[0]),
                bounds=(t0, t0 + duration),
                eps=tol,
                variance=variance,
            )
            n_max = max(n_max, worst_case.n_frequencies)
        cos_weights = np.zeros((len(lengthscales), n_max + 1))
        sin_weights = np.zeros((len(lengthscales), n_max))
        for k, (wc, ws) in enumerate(zip(cos_w, sin_w)):
            cos_weights[k, : len(wc)] = wc
            sin_weights[k, : len(ws)] = ws

        return cls(
            lengthscales=jnp.asarray(lengthscales),
            frequency_spacing=jnp.asarray(spacings),
            cos_weights=jnp.asarray(cos_weights),
            sin_weights=jnp.asarray(sin_weights),
            t0=float(t0),
            duration=float(duration),
            tol=float(tol),
        )

    @property
    def n_latents(self) -> int:
        return self.cos_weights.shape[0]

    @property
    def n_frequencies(self) -> int:
        """``M``: the largest number of positive frequencies over latents."""
        return self.sin_weights.shape[1]

    @property
    def n_coefficients(self) -> int:
        """Coefficients per latent, ``2 * M + 1``."""
        return 2 * self.n_frequencies + 1

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        return (self.n_latents, self.n_coefficients)

    def phases(self, times: Float[Array, " n_time"]) -> Float[Array, "n_latents n_time"]:
        """Base phase ``x_k(t) = 2 pi h_k (t - t0)`` of each latent at ``times``.

        Mode ``j`` of latent ``k`` oscillates as ``cos(j x_k(t))``. The EFGP grid
        guarantees ``h_k * duration < 1``, so on the domain ``x`` stays in
        ``[0, 2 pi)``, which is inside the range FINUFFT accepts.
        """
        return 2.0 * jnp.pi * self.frequency_spacing[:, None] * (times[None, :] - self.t0)

    def features(self, times: Float[Array, " n_time"]) -> Float[Array, "n_latents n_time n_coef"]:
        """Dense feature matrix ``Phi_k(t)`` of each latent, for tests and small problems.

        ``z_k(times) = features(times)[k] @ coefficients[k]``.
        """
        x = self.phases(times)[..., None]  # (K, T, 1)
        j = jnp.arange(self.n_frequencies + 1)
        cos = jnp.cos(j * x) * self.cos_weights[:, None, :]
        sin = jnp.sin(j[1:] * x) * self.sin_weights[:, None, :]
        return jnp.concatenate([cos, sin], axis=-1)

    def kernel(
        self, t1: Float[Array, " n1"], t2: Float[Array, " n2"]
    ) -> Float[Array, "n_latents n1 n2"]:
        """The approximate kernel ``Phi(t1) Phi(t2)^T`` of each latent."""
        return jnp.einsum("kic,kjc->kij", self.features(t1), self.features(t2))
