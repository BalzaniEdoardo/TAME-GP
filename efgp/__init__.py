"""EFGP Laplace inference for the Poisson-Gaussian GP-CCA latent model.

Each latent dimension is a GP with a squared-exponential kernel, approximated in
weight space with the equispaced Fourier grid of :class:`nemos.basis.FourierGP`.
Products with the feature matrix are NUFFTs (jax-finufft), wrapped as lineax
operators, and the MAP is found with a matrix-free Newton-CG solver built on nemos.

Typical use:

.. code-block:: python

    import jax
    jax.config.update("jax_enable_x64", True)
    import efgp

    prior = efgp.EFGPPrior.from_lengthscales(lengthscales, duration=5.0)
    model = efgp.LatentModel(
        latents={"shared": 2, "V1": 3},
        prior=prior,
        observations={
            "task": efgp.GaussianObservations(C, d, Psi, latents=("shared",)),
            "V1": efgp.PoissonObservations(W, b, latents=("shared", "V1")),
        },
    )
    trial = efgp.Trial(times, {"task": s, "V1": counts})
    solver = efgp.newton_cg_solver(model)
    posterior = efgp.infer_trial(model, trial, solver=solver)

Double precision (``jax_enable_x64``) is strongly recommended.
"""

from . import _compat  # noqa: F401  (must run before any NUFFT is vmapped)
from .inference import (
    LaplacePosterior,
    infer_trial,
    infer_trials,
    laplace_covariance_blocks,
    newton_cg_solver,
    posterior_covariance_operator,
)
from .model import LatentModel, Trial, negative_log_joint, stack_trials
from .observations import GaussianObservations, ObservationBlock, PoissonObservations
from .operators import (
    block_diagonal_operator,
    fourier_features,
    fourier_features_transpose,
    newton_operator,
)
from .preconditioners import latent_block_preconditioner, no_preconditioner
from .prior import EFGPPrior
from .solver import NewtonCG, NewtonSystem, efgp_newton_system

__all__ = [
    "EFGPPrior",
    "LatentModel",
    "Trial",
    "stack_trials",
    "negative_log_joint",
    "ObservationBlock",
    "PoissonObservations",
    "GaussianObservations",
    "fourier_features",
    "fourier_features_transpose",
    "block_diagonal_operator",
    "newton_operator",
    "latent_block_preconditioner",
    "no_preconditioner",
    "NewtonCG",
    "NewtonSystem",
    "efgp_newton_system",
    "newton_cg_solver",
    "LaplacePosterior",
    "infer_trial",
    "infer_trials",
    "laplace_covariance_blocks",
    "posterior_covariance_operator",
]
