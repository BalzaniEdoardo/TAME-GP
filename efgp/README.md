# `efgp`: EFGP Laplace inference for TAME-GP

JAX replacement for the E-step of `core/inference.py`. The model is the same
Poisson-Gaussian probabilistic CCA with GP priors; only the inference changes.

## Model

- Latent groups, for example `"shared"` (z0) and one private group per brain area.
  Every latent dimension is an independent GP with a squared-exponential kernel,
  and every trial is an independent draw.
- Observation blocks. Each block depends on some latent groups:
  - `GaussianObservations`: `y_t ~ N(C z_t + d, Psi)`, for example task variables.
  - `PoissonObservations`: `y_t ~ Poisson(exp(W z_t + b))`, for example spike counts.

## How inference works

1. **Prior in weight space.** `EFGPPrior` takes the frequency grid and weights from
   `nemos.basis.FourierGP`. Each latent is `z_k(t) = Phi_k(t) beta_k` with
   `beta_k ~ N(0, I)`, and `Phi_k Phi_k^T` approximates the SE kernel to `tol`.
   The dense basis matrix is never built.
2. **Operators.** `Phi` (type-2 NUFFT) and `Phi^T` (type-1 NUFFT, from jax-finufft)
   are lineax operators. The Newton matrix is `I + Phi^T H Phi`, where `H` holds one
   `K x K` block per time point. All of them compose with `@` and `+`.
3. **Solver.** `NewtonCG` is a nemos second-order solver: nemos' Newton loop,
   Armijo line search and convergence test, with the Newton step solved by
   `lineax.CG` and a preconditioner. The default preconditioner inverts the exact
   per-latent diagonal blocks of the Newton matrix, built with one NUFFT.
4. **Laplace covariance.** `cov_t` (`T x K x K`) and `cov_k` (`K x T x T`) come from a
   dense Cholesky factor of the coefficient precision, of size `K (2M + 1)`.
   `posterior_covariance_operator` gives matrix-free products with the full
   covariance.

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)   # strongly recommended
import efgp

prior = efgp.EFGPPrior.from_lengthscales(lengthscales, duration=t_max, tol=1e-6)
model = efgp.LatentModel(
    latents={"shared": 2, "V1": 3},
    prior=prior,
    observations={
        "task": efgp.GaussianObservations(C, d, Psi, latents=("shared",)),
        "V1": efgp.PoissonObservations(W, b, latents=("shared", "V1")),
    },
)
trial = efgp.Trial(times, {"task": s, "V1": counts})   # NaN = missing entry

solver = efgp.newton_cg_solver(model)                  # build once, reuse
post = efgp.infer_trial(model, trial, solver=solver)
post.mean, post.cov_t, post.cov_k

# many trials: pad, stack, vmap
posts = efgp.infer_trials(model, efgp.stack_trials(trials), solver=solver)
```

`examples/efgp_quickstart.py` is a complete runnable script.
`examples/compare_legacy_efgp.py` (a jupytext notebook) compares this code with
`core/inference.py` on the synthetic data of `core/gen_synthetic_data.py`.

Any nemos solver built for `negative_log_joint(coefficients, model, trial)` can
replace the default. For example, nemos' dense `Newton` serves as a reference:

```python
from nemos.solvers import Newton
from nemos.regularizer import UnRegularized
dense = Newton(efgp.negative_log_joint, UnRegularized(), None, False,
               init_params=model.init_coefficients())
post = efgp.infer_trial(model, trial, solver=dense)
```

To change the CG tolerances or the preconditioner:

```python
import lineax as lx
solver = efgp.newton_cg_solver(
    model,
    cg=lx.CG(rtol=1e-8, atol=1e-8, max_steps=500),
    preconditioner=efgp.no_preconditioner,
)
```

A preconditioner is any function `(model, coefficients, trial, hessian_blocks)` that
returns a positive semidefinite lineax operator close to the inverse Newton matrix.

## With the existing `P_GPCCA` code

```python
from efgp.legacy import model_from_pgpcca, trial_from_pgpcca, store_posterior

model = model_from_pgpcca(data)           # current parameters of the P_GPCCA object
solver = efgp.newton_cg_solver(model)
for tr in data.trialDur:
    post = efgp.infer_trial(model, trial_from_pgpcca(data, tr), solver=solver)
    store_posterior(data, tr, model, post)  # same format as multiTrialInference
```

The existing M-step then runs unchanged. Rebuild the model after each M-step. The
solver can be reused unless the lengthscales changed, because the number of
frequencies depends on them.

## Differences with `core/inference.py`

- **No white-noise term.** The old kernel had `epsNoise * I` added. The EFGP prior
  keeps the SE part, with variance `1 - epsNoise`. On the synthetic data the
  posterior means differ by about 0.5% of their scale, and the covariances by about
  1e-3. The old optimiser is converged, so this is the model difference.
- **Missing task variables are masked, not deleted.** The old code deleted time
  points with NaN task variables, then built the kernel on consecutive indices, so
  the time points on either side of a gap were treated as adjacent. Here the time
  point is kept, the spikes there are used, and the Gaussian likelihood uses only
  the observed entries (precision `inv(Psi[obs, obs])`). Build `P_GPCCA` with
  `filter_out_nan=False`.
- **Arbitrary time stamps.** Trials can have irregular time stamps. Every time
  stamp must lie in `[prior.t0, prior.t0 + prior.duration]`. The Fourier
  representation is periodic, and `infer_trial` raises on time stamps outside the
  domain.

## Practical notes

- **Cost.** Each NUFFT call has a fixed cost of about 0.3 ms (single thread, set
  in `operators.NUFFT_OPTS`). For short trials, the dense covariance step and the
  NUFFT overhead dominate. The NUFFT path pays off for long trials and many
  frequencies.
- **Changing lengthscales.** The number of frequencies is a Python integer that
  depends on the lengthscales. Changing them means a new prior and a new
  compilation.
- **Library internals.** `solver.py` imports private nemos modules
  (`nemos.solvers._second_order`), tested against the nemos `development` branch.
  `_compat.py` patches a jax-finufft 1.3.1 incompatibility with JAX >= 0.10 in
  `vmap`; remove the patch when jax-finufft fixes it.

## Tests

```
pytest tests/efgp
```

The tests check each piece against a dense reference: kernel accuracy, the NUFFT
operators against `FourierGP.compute_features`, the preconditioner, the closed form
of a Gaussian-only model with missing entries, nemos' dense `Newton`, batching, and
the original `core/inference.py`.
