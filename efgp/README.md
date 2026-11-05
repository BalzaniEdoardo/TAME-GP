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

1. **Prior in weight space.** `EFGPPrior` holds the frequency grid and weights of
   `nemos.basis.FourierGP`. Each latent is `z_k(t) = Phi_k(t) beta_k` with
   `beta_k ~ N(0, I)`, and `Phi_k Phi_k^T` approximates the SE kernel to `tol`.
   The grid is computed by `efgp.prior.efgp_grid`, a numpy copy of nemos'
   formula. The nemos version compiles small JAX kernels for every new
   lengthscale, about 0.2 s per EM iteration, while the numpy copy takes about
   0.1 ms. A test checks that both give the same grid. The dense basis matrix is
   never built.
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

## Install

In the project environment (not in the environment that runs Jupyter):

```
pip install -e path/to/nemos      # development branch; or git+https://github.com/flatironinstitute/nemos.git@development
pip install -e ".[docs,test]"     # docs: matplotlib for the notebooks; test: pytest. Both add numba, scikit-learn, scipy for core/
```

Leave out `[docs,test]` for the inference code alone.

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

## Fitting the parameters (EM) with the existing `P_GPCCA` code

The M-steps in `core/` read the latents only from `data.posterior_inf`. The EFGP
E-step fills it in the same format, so EM is the original loop with the E-step
swapped:

```python
from expectation_maximization import expectation_maximization   # core/
from efgp.legacy import EFGPEStep, GPTimescaleMStep

tau_bounds = (0.05, 2.0)   # seconds: the range the GP time constants may take
expectation_maximization(
    data, maxIter=20, method="L-BFGS-B",
    tau_bounds=tau_bounds,
    e_step=EFGPEStep(tau_bounds),             # EFGP Laplace inference
    gp_m_step=GPTimescaleMStep(tau_bounds),   # JAX M-step for the time constants
)
```

- **Why bounds.** The number of Fourier frequencies depends on the time constant
  (about `trial duration / tau`). The prior is sized once for the shortest allowed
  time constant, so the solver compiles once for the whole fit. Longer time
  constants are padded with zero-weight modes, which leaves their prior unchanged.
  The lower bound sets the cost: do not make it smaller than needed.
- **Same bounds everywhere.** A time constant outside `tau_bounds` makes the E-step
  raise a `ValueError`.
- **GP M-step.** `GPTimescaleMStep` (`efgp/mstep.py`) maximises the same objective as
  the L-BFGS-B M-step of `core/learnGPParams.py`, and the tests check that it finds
  the same time constants. Each latent's time constant is a separate
  one-dimensional problem, trials with the same time grid are summed first, and
  the bounds are enforced by a change of variables solved with nemos' `Newton`.
  It is 15-50x faster than the L-BFGS-B M-step (10-40 trials of 50-200 bins).
  Without `gp_m_step`, the EM loop uses the original M-step.
- **Poisson M-step start.** It now starts from the current parameters
  (`warm_start_poisson=True`); pass `False` for the earlier start from zeros. The
  problem is convex, so the optimum is the same.
- **Poisson M-step.** Use `method="L-BFGS-B"`. The default `"sparse-Newton"` needs
  the `csr` package.
- **Missing task variables.** The M-steps in `core/` delete time points with NaN
  task variables, while EFGP keeps them, so `EFGPEStep` refuses data with NaN task
  variables. Supporting them needs a mask-aware M-step.

To run only the E-step:

```python
from efgp.legacy import model_from_pgpcca, trial_from_pgpcca, store_posterior

model = model_from_pgpcca(data)           # current parameters of the P_GPCCA object
solver = efgp.newton_cg_solver(model)
for tr in data.trialDur:
    post = efgp.infer_trial(model, trial_from_pgpcca(data, tr), solver=solver)
    store_posterior(data, tr, model, post)  # same format as multiTrialInference
```

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
  depends on the lengthscales. Pass `lengthscale_bounds` to
  `EFGPPrior.from_lengthscales` to fix it for a whole range of lengthscales;
  otherwise each new value can change it and trigger a recompilation.
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
