"""The latent variable model and the per-trial data container.

The model
---------
Latent groups (for example ``"shared"`` and one private group per brain area) each
have some dimensions, and every latent dimension is an independent GP over time with
an EFGP prior. Each observation block depends on some of the groups:

.. code-block:: python

    model = LatentModel(
        latents={"shared": 2, "V1": 3},
        prior=EFGPPrior.from_lengthscales([0.2, 0.3, 0.1, 0.1, 0.1], duration=5.0),
        observations={
            "task": GaussianObservations(C, d, Psi, latents=("shared",)),
            "V1": PoissonObservations(W, b, latents=("shared", "V1")),
        },
    )

The column order of the latents ``z`` is the order of the ``latents`` dict, and the
lengthscales of the prior follow the same order.

Inference works on the Fourier coefficients ``beta`` of shape ``(K, 2M + 1)``, with
``z = Phi beta`` and ``beta ~ N(0, I)``. The function minimised per trial is
:func:`negative_log_joint`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Bool, Float

from .observations import ObservationBlock
from .operators import (
    block_diagonal_operator,
    fourier_features,
    fourier_features_transpose,
    newton_operator,
)
from .prior import EFGPPrior


class Trial(eqx.Module):
    """Observations of one trial (or of a batch of padded trials).

    Attributes
    ----------
    times :
        Time stamps, shape ``(T,)``, in the units of the prior lengthscales. They need
        not be equispaced.
    observations :
        One ``(T, D)`` array per observation block, keyed like
        ``LatentModel.observations``. NaN marks a missing entry.
    valid :
        ``False`` on padding time points, shape ``(T,)``. Padding contributes no
        likelihood.
    """

    times: Float[Array, " n_time"]
    observations: dict[str, Float[Array, "n_time n_obs"]]
    valid: Bool[Array, " n_time"]

    def __init__(self, times, observations: Mapping[str, Array], valid=None):
        self.times = jnp.asarray(times, dtype=float)
        self.observations = {k: jnp.asarray(v, dtype=float) for k, v in observations.items()}
        if valid is None:
            valid = jnp.ones(self.times.shape, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)

    @property
    def n_time(self) -> int:
        return self.times.shape[-1]

    def mask(self, name: str) -> Bool[Array, "n_time n_obs"]:
        """Observed entries of block ``name``: not NaN and not padding."""
        y = self.observations[name]
        return ~jnp.isnan(y) & self.valid[..., None]


def stack_trials(trials: Sequence[Trial], n_time: int | None = None) -> Trial:
    """Pad trials to a common length and stack them on a leading axis.

    Use the result with :func:`efgp.infer_trials`. Padding time points repeat the
    last time stamp of the trial, have NaN observations and ``valid = False``.
    Group trials of similar length together to limit the padding.
    """
    if n_time is None:
        n_time = max(tr.n_time for tr in trials)

    def pad(x, fill):
        n_pad = n_time - x.shape[0]
        if n_pad < 0:
            raise ValueError(f"A trial has {x.shape[0]} time points, more than n_time={n_time}.")
        filler = jnp.full((n_pad,) + x.shape[1:], fill, dtype=x.dtype)
        return jnp.concatenate([x, filler])

    names = list(trials[0].observations)
    return Trial(
        times=jnp.stack([pad(tr.times, tr.times[-1]) for tr in trials]),
        observations={
            name: jnp.stack([pad(tr.observations[name], jnp.nan) for tr in trials])
            for name in names
        },
        valid=jnp.stack([pad(tr.valid, False) for tr in trials]),
    )


class LatentModel(eqx.Module):
    """Latent groups with EFGP priors, and the observation blocks that depend on them.

    Parameters
    ----------
    latents :
        Ordered ``{group name: number of dimensions}``.
    prior :
        One lengthscale per latent dimension, in the same order.
    observations :
        ``{block name: observation model}``. Each model lists the groups it depends on.
    """

    prior: EFGPPrior
    observations: dict[str, ObservationBlock]
    latent_groups: tuple[tuple[str, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        latents: Mapping[str, int],
        prior: EFGPPrior,
        observations: Mapping[str, ObservationBlock],
    ):
        self.latent_groups = tuple((str(k), int(v)) for k, v in latents.items())
        self.prior = prior
        self.observations = dict(observations)

        n_latents = sum(size for _, size in self.latent_groups)
        if prior.n_latents != n_latents:
            raise ValueError(
                f"The prior has {prior.n_latents} lengthscales but the latent groups "
                f"have {n_latents} dimensions in total."
            )
        for name, block in self.observations.items():
            unknown = set(block.latents) - set(dict(self.latent_groups))
            if unknown:
                raise ValueError(f"Block {name!r} depends on unknown latent groups {unknown}.")
            n_cols = len(self.latent_index(block.latents))
            if block.loadings.shape[1] != n_cols:
                raise ValueError(
                    f"Block {name!r}: loadings have {block.loadings.shape[1]} columns, "
                    f"but its latent groups {block.latents} have {n_cols} dimensions."
                )

    # ------------------------------------------------------------------ structure
    @property
    def n_latents(self) -> int:
        return self.prior.n_latents

    def latent_index(self, groups: Sequence[str]) -> np.ndarray:
        """Columns of ``z`` that belong to ``groups``, in the given order."""
        start, slices = 0, {}
        for name, size in self.latent_groups:
            slices[name] = np.arange(start, start + size)
            start += size
        return np.concatenate([slices[g] for g in groups])

    def split_latents(self, z: Float[Array, "... n_latents"]) -> dict[str, Array]:
        """``{group name: z[..., columns of that group]}``."""
        return {name: z[..., self.latent_index([name])] for name, _ in self.latent_groups}

    def init_coefficients(self) -> Float[Array, "n_latents n_coef"]:
        """All-zero coefficients, the prior mean."""
        return jnp.zeros(self.prior.coefficient_shape)

    # ------------------------------------------------------------------ operators
    def features(self, trial: Trial) -> lx.AbstractLinearOperator:
        """``Phi`` for the time points of ``trial``."""
        return fourier_features(self.prior, trial.times)

    def latents(self, coefficients: Array, trial: Trial) -> Float[Array, "n_time n_latents"]:
        """Latent trajectories ``z = Phi beta`` on the time points of ``trial``."""
        return self.features(trial).mv(coefficients)

    # ------------------------------------------------------------------ likelihood
    def log_likelihood(self, z: Array, trial: Trial) -> Float[Array, ""]:
        total = jnp.zeros(())
        for name, block in self.observations.items():
            z_block = z[:, self.latent_index(block.latents)]
            total = total + block.log_likelihood(
                z_block, trial.observations[name], trial.mask(name)
            )
        return total

    def negative_hessian(self, z: Array, trial: Trial) -> Float[Array, "n_time n_latents n_latents"]:
        """Negative Hessian of the log-likelihood in ``z``, one ``K x K`` block per time."""
        blocks = jnp.zeros((z.shape[0], self.n_latents, self.n_latents), dtype=z.dtype)
        for name, block in self.observations.items():
            idx = self.latent_index(block.latents)
            h = block.negative_hessian(z[:, idx], trial.observations[name], trial.mask(name))
            blocks = blocks.at[:, idx[:, None], idx[None, :]].add(h)
        return blocks

    def precision_operator(
        self, trial: Trial, hessian_blocks: Float[Array, "n_time n_latents n_latents"]
    ) -> lx.AbstractLinearOperator:
        """Precision of the coefficients, ``I + Phi^T H Phi``, for given Hessian blocks.

        With ``hessian_blocks = negative_hessian(z, trial)`` at ``z = Phi beta``, this
        is the Hessian of :func:`negative_log_joint` at ``beta``: the Newton matrix,
        and at the MAP the Laplace precision.
        """
        return newton_operator(
            self.features(trial),
            fourier_features_transpose(self.prior, trial.times),
            block_diagonal_operator(hessian_blocks),
        )


def negative_log_joint(coefficients: Array, model: LatentModel, trial: Trial) -> Float[Array, ""]:
    """``0.5 |beta|^2 - log p(y | z = Phi beta)``: the objective minimised per trial.

    This is a module-level function of ``(coefficients, model, trial)`` so that one
    solver instance works for any parameter values and any trial.
    """
    z = model.latents(coefficients, trial)
    return 0.5 * jnp.sum(coefficients**2) - model.log_likelihood(z, trial)
