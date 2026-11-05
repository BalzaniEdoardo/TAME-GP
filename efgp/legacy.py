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

import jax
import jax.numpy as jnp
import numpy as np

from .inference import LaplacePosterior, infer_trials, newton_cg_solver
from .model import LatentModel, Trial, stack_trials
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
    tau_bounds: tuple[float, float] | None = None,
) -> LatentModel:
    """Build a :class:`LatentModel` from the current parameters of a ``P_GPCCA`` object.

    The latent groups are ``"shared"`` (``z0``) and one group per area, named after
    ``data.area_list``. The task variables, if any, are the ``"task"`` block.
    ``tau_bounds`` (seconds) is passed to :meth:`EFGPPrior.from_lengthscales` as
    ``lengthscale_bounds``.
    """
    dt = data.binSize / 1000.0
    duration = (max(data.trialDur.values()) - 1) * dt
    lengthscales = np.concatenate([np.atleast_1d(p["tau"]) for p in data.priorPar])
    prior = EFGPPrior.from_lengthscales(
        lengthscales,
        duration=duration,
        variance=1.0 - data.epsNoise,
        tol=tol,
        lengthscale_bounds=tau_bounds,
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


class EFGPEStep:
    """EFGP E-step for the EM loop of ``core/expectation_maximization.py``.

    .. code-block:: python

        from expectation_maximization import expectation_maximization
        bounds = (0.05, 2.0)  # seconds
        expectation_maximization(data, maxIter=20, method="L-BFGS-B",
                                 e_step=EFGPEStep(tau_bounds=bounds), tau_bounds=bounds)

    Each call rebuilds the model from the current parameters of ``data``, infers
    every trial and writes ``data.posterior_inf`` with :func:`store_posterior`, so the
    original M-steps run unchanged.

    The number of Fourier frequencies depends on the GP time constants. The prior is
    therefore sized once for the shortest allowed time constant, ``tau_bounds[0]``,
    and every latent is padded to that size. The coefficient shape then never
    changes during the fit and the solver compiles once. Pass the same bounds to the
    GP M-step (``tau_bounds`` of ``expectation_maximization``) so the time constants
    stay in the range; a time constant outside it raises a ``ValueError``.

    The MAP of each trial is kept and used as the starting point of the next E-step.

    Parameters
    ----------
    tau_bounds :
        ``(lower, upper)`` range of the GP time constants, in seconds. The lower
        bound sets the cost: the number of frequencies grows like
        ``trial duration / lower``.
    tol :
        Kernel approximation tolerance of the prior.
    solver_kwargs :
        Passed to :func:`efgp.newton_cg_solver`.
    """

    def __init__(self, tau_bounds: tuple[float, float], tol: float = 1e-8, **solver_kwargs):
        self.tau_bounds = tuple(float(b) for b in tau_bounds)
        self.tol = tol
        self.solver_kwargs = solver_kwargs
        self._solver = None
        self._coefficients = {}

    def __call__(self, data, plot_trial: bool = False, **_ignored) -> float:
        _check_no_missing_task_variables(data)
        model = model_from_pgpcca(data, tol=self.tol, tau_bounds=self.tau_bounds)
        if self._solver is None:
            self._solver = newton_cg_solver(model, **self.solver_kwargs)

        # Trials of the same length are solved together in one vmapped call. The
        # groups are the same at every EM iteration, so each compiles once.
        by_length = {}
        for tr in data.trialDur:
            by_length.setdefault(data.trialDur[tr], []).append(tr)

        total = 0.0
        zeros = model.init_coefficients()
        for n_time, ids in by_length.items():
            if plot_trial:
                print(f"infer {len(ids)} trials of {n_time} time points")
            posteriors = infer_trials(
                model,
                stack_trials([trial_from_pgpcca(data, tr) for tr in ids]),
                solver=self._solver,
                init_coefficients=jnp.stack([self._coefficients.get(tr, zeros) for tr in ids]),
            )
            for i, tr in enumerate(ids):
                posterior = jax.tree.map(lambda x: x[i], posteriors)
                store_posterior(data, tr, model, posterior)
                self._coefficients[tr] = posterior.coefficients
                total += float(posterior.objective)
        return total


class GPTimescaleMStep:
    """JAX M-step for the GP time constants, for the EM loop in ``core/``.

    .. code-block:: python

        expectation_maximization(data, ..., e_step=EFGPEStep(bounds), tau_bounds=bounds,
                                 gp_m_step=GPTimescaleMStep(bounds))

    It maximises the same objective as the L-BFGS-B M-step of ``core/`` (the
    expected log GP prior, ``all_trial_GPLL``), with the time constants of all
    latent groups fitted at once (see :mod:`efgp.mstep`). Its only input from the
    E-step is ``data.posterior_inf``, so it works with either E-step.

    Calling it updates ``data.priorPar`` and returns the value the EM loop adds to
    its negative log-likelihood: ``-sum(all_trial_GPLL)`` at the new time constants.
    """

    def __init__(self, tau_bounds: tuple[float, float], maxiter: int = 100, tol: float = 1e-10):
        self.tau_bounds = tuple(float(b) for b in tau_bounds)
        self.maxiter = maxiter
        self.tol = tol
        self._fitter = None

    def __call__(self, data) -> float:
        from .mstep import TimescaleFitter, expected_log_prior

        if self._fitter is None:
            self._fitter = TimescaleFitter(
                self.tau_bounds,
                variance=1.0 - data.epsNoise,
                nugget=data.epsNoise,
                maxiter=self.maxiter,
                tol=self.tol,
            )
        statistics = self.statistics(data)
        tau0 = np.concatenate([np.atleast_1d(p["tau"]) for p in data.priorPar])
        tau, _ = self._fitter(tau0, statistics)
        tau = np.asarray(tau)

        start = 0
        for p in data.priorPar:
            n = np.atleast_1d(p["tau"]).shape[0]
            p["tau"] = tau[start : start + n]
            start += n

        value = expected_log_prior(
            jnp.asarray(tau), statistics, self._fitter.variance, self._fitter.nugget
        )
        return -float(jnp.sum(value)) / len(data.trialDur)

    @staticmethod
    def statistics(data):
        """Posterior second moments from ``data.posterior_inf``, grouped by trial length."""
        from .mstep import timescale_statistics

        dt = data.binSize / 1000.0
        n_groups = len(data.zdims)
        times, means, covs = [], [], []
        for tr in data.trialDur:
            post = data.posterior_inf[tr]
            means.append(np.concatenate([post.mean[g] for g in range(n_groups)], axis=0))
            covs.append(np.concatenate([post.cov_k[g] for g in range(n_groups)], axis=0))
            times.append(np.arange(means[-1].shape[1]) * dt)
        return timescale_statistics(times, means, covs)


def _check_no_missing_task_variables(data) -> None:
    """The original M-steps drop time points with NaN task variables; EFGP keeps them.

    The two would disagree on the trial length, so refuse instead of failing later
    with a shape error.
    """
    for tr in data.trialDur:
        stim, _ = data.get_observations(tr, remove_nan=False)
        if np.isnan(stim).any():
            raise NotImplementedError(
                f"Trial {tr} has missing task variables. The M-steps in core/ delete "
                "those time points (get_observations(remove_nan=True)) and the GP "
                "M-step then treats the remaining time points as consecutive, while "
                "the EFGP posterior keeps every time point. Use the EFGP E-step on data "
                "without missing task variables, or with a mask-aware M-step."
            )


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
