"""Adapter to the original numpy implementation in ``core/`` (the ``P_GPCCA`` class).

* :func:`model_from_pgpcca` turns the current parameters of a ``P_GPCCA`` object into
  a :class:`LatentModel`.
* :func:`trial_from_pgpcca` extracts one trial.
* :func:`store_posterior` writes a :class:`LaplacePosterior` into
  ``data.posterior_inf[trial_id]`` in the format the existing M-step expects.

Differences with ``core/inference.py``
--------------------------------------
* The white-noise term ``epsNoise`` of the old kernel is dropped; the SE part keeps
  variance ``1 - epsNoise``.
* Time stamps are ``bin index * binSize / 1000`` seconds, so ``tau`` is the
  lengthscale in seconds, as in ``makeK_big``.
* Missing task variables (NaN) are masked entry by entry, and the time point is
  kept. The old ``get_observations(remove_nan=True)`` deleted the time point and
  then built the kernel on consecutive indices, which treated the time points on
  either side of a gap as adjacent. Build the ``P_GPCCA`` object with
  ``filter_out_nan=False`` so that NaN time points are still in the data.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from .inference import LaplacePosterior
from .model import LatentModel, Trial
from .observations import GaussianObservations, PoissonObservations
from .prior import EFGPPrior

SHARED = "shared"
TASK = "task"


def _area_names(data) -> list[str]:
    return [str(area) for area in data.area_list]


def model_from_pgpcca(
    data,
    *,
    tol: float = 1e-8,
    use_task_variables: bool = True,
) -> LatentModel:
    """Build a :class:`LatentModel` from the current parameters of a ``P_GPCCA`` object.

    The latent groups are ``"shared"`` (``z0``) and one group per area, named after
    ``data.area_list``. The task variables, if any, are the ``"task"`` block.
    """
    dt = data.binSize / 1000.0
    duration = (max(data.trialDur.values()) - 1) * dt
    lengthscales = np.concatenate([np.atleast_1d(p["tau"]) for p in data.priorPar])
    prior = EFGPPrior.from_lengthscales(
        lengthscales, duration=duration, variance=1.0 - data.epsNoise, tol=tol
    )

    areas = _area_names(data)
    latents = {SHARED: int(data.zdims[0])}
    latents.update({area: int(k) for area, k in zip(areas, data.zdims[1:])})

    observations = {}
    if use_task_variables and len(data.var_list) > 0:
        observations[TASK] = GaussianObservations(
            loadings=jnp.asarray(data.stimPar["W0"]),
            offset=jnp.asarray(data.stimPar["d"]),
            covariance=jnp.asarray(np.linalg.inv(data.stimPar["PsiInv"])),
            latents=(SHARED,),
        )
    for area, par in zip(areas, data.xPar):
        observations[area] = PoissonObservations(
            loadings=jnp.asarray(np.hstack([par["W0"], par["W1"]])),
            offset=jnp.asarray(par["d"]),
            latents=(SHARED, area),
        )
    return LatentModel(latents, prior, observations)


def trial_from_pgpcca(data, trial_id, *, use_task_variables: bool = True) -> Trial:
    """Observations of one trial, with NaN task variables kept (and later masked)."""
    stim, counts = data.get_observations(trial_id, remove_nan=False)
    dt = data.binSize / 1000.0
    observations = {area: x for area, x in zip(_area_names(data), counts)}
    if use_task_variables and stim.shape[1] > 0:
        observations[TASK] = stim
    return Trial(np.arange(stim.shape[0]) * dt, observations)


def store_posterior(data, trial_id, model: LatentModel, posterior: LaplacePosterior) -> None:
    """Write ``posterior`` into ``data.posterior_inf[trial_id]`` in the old format.

    Keys are latent group indices (0 for ``z0``, ``i`` for area ``i``):

    * ``mean[i]``: ``(K_i, T)``
    * ``cov_t[i]``: ``(T, K_i, K_i)``
    * ``cross_cov_t[i]`` for ``i >= 1``: ``(T, K_0, K_i)``, the covariance of ``z0`` with ``z_i``
    * ``cov_k[i]``: ``(K_i, T, T)``

    ``posterior`` must have been computed with ``covariance="blocks"``.
    """
    if posterior.cov_t is None or posterior.cov_k is None:
        raise ValueError("store_posterior needs cov_t and cov_k: use covariance='blocks'.")
    if not hasattr(data, "posterior_inf"):
        data.posterior_inf = {}

    shared = model.latent_index([SHARED])
    mean, cov_t, cross_cov_t, cov_k = {}, {}, {}, {}
    for i, (group, _) in enumerate(model.latent_groups):
        idx = model.latent_index([group])
        mean[i] = np.asarray(posterior.mean[:, idx].T)
        cov_t[i] = np.asarray(posterior.cov_t[:, idx[:, None], idx[None, :]])
        cov_k[i] = np.asarray(posterior.cov_k[idx])
        if i > 0:
            cross_cov_t[i] = np.asarray(posterior.cov_t[:, shared[:, None], idx[None, :]])

    data.posterior_inf[trial_id] = SimpleNamespace(
        mean=mean, cov_t=cov_t, cross_cov_t=cross_cov_t, cov_k=cov_k
    )
