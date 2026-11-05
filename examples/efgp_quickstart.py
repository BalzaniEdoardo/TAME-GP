"""EFGP Laplace inference on synthetic data: one shared latent, one area.

Run from the repository root:  PYTHONPATH=. python examples/efgp_quickstart.py
(or install the package first: pip install -e .)
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import efgp  # noqa: E402

rng = np.random.default_rng(0)

# ----------------------------------------------------------------- simulate data
dt, n_time = 0.05, 200  # 50 ms bins, 10 s trials
times = np.arange(n_time) * dt
lengthscales = [0.4, 0.25, 0.3]  # seconds: shared, V1 (2 dims)

prior = efgp.EFGPPrior.from_lengthscales(lengthscales, duration=times[-1], tol=1e-6)
true_beta = rng.normal(size=prior.coefficient_shape)
true_z = np.asarray(efgp.fourier_features(prior, times).mv(jnp.asarray(true_beta)))

task_loadings = rng.normal(size=(2, 1))
task_noise = 0.1 * np.eye(2)
task = true_z[:, :1] @ task_loadings.T + rng.multivariate_normal(np.zeros(2), task_noise, n_time)
task[40:80, 1] = np.nan  # a missing stretch of one task variable

unit_loadings = 0.5 * rng.normal(size=(40, 3))
unit_offset = np.full(40, -1.0)
counts = rng.poisson(np.exp(true_z @ unit_loadings.T + unit_offset))

# ----------------------------------------------------------------- model and data
model = efgp.LatentModel(
    latents={"shared": 1, "V1": 2},
    prior=prior,
    observations={
        "task": efgp.GaussianObservations(
            jnp.asarray(task_loadings), jnp.zeros(2), jnp.asarray(task_noise),
            latents=("shared",),
        ),
        "V1": efgp.PoissonObservations(
            jnp.asarray(unit_loadings), jnp.asarray(unit_offset),
            latents=("shared", "V1"),
        ),
    },
)
trial = efgp.Trial(times, {"task": task, "V1": counts})

# ----------------------------------------------------------------- inference
solver = efgp.newton_cg_solver(model)  # build once, reuse for every trial
posterior = efgp.infer_trial(model, trial, solver=solver)

print(f"Newton steps: {int(posterior.num_steps)}, converged: {bool(posterior.converged)}")
for k in range(3):
    r = np.corrcoef(posterior.mean[:, k], true_z[:, k])[0, 1]
    print(f"latent {k}: corr(posterior mean, truth) = {r:.3f}")
print("cov_t:", posterior.cov_t.shape, " cov_k:", posterior.cov_k.shape)

# Several trials at once: pad, stack, vmap.
short = efgp.Trial(times[:120], {"task": task[:120], "V1": counts[:120]})
batch = efgp.stack_trials([trial, short])
posteriors = efgp.infer_trials(model, batch, solver=solver)
print("batched mean:", posteriors.mean.shape, "(padded rows: batch.valid == False)")

# The matrix-free posterior covariance is an ordinary lineax operator.
cov = efgp.posterior_covariance_operator(model, trial, posterior.coefficients)
probe = jnp.zeros((n_time, 3)).at[100, 0].set(1.0)
print("Cov[z_shared(t), z_shared(5 s)] at t = 5 s:", float(cov.mv(probe)[100, 0]))
