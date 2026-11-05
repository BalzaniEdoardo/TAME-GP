# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (tame_gp)
#     language: python
#     name: tame_gp
# ---

# %% [markdown]
# # Original Laplace inference vs EFGP inference
#
# Both methods compute the same Laplace approximation of the latent posterior of
# one trial. They differ in two ways:
#
# | | `core/inference.py` | `efgp` |
# |---|---|---|
# | GP prior | exact SE kernel + white noise `epsNoise`, as a dense `T x T` inverse | SE kernel in the Fourier weight space of `nemos.basis.FourierGP` |
# | MAP | L-BFGS (scipy) on the latents | Newton-CG (nemos + lineax) on the Fourier coefficients |
#
# The simulation is the one in `core/gen_synthetic_data.py`: a shared latent (2 dims)
# drives 4 Gaussian task variables and two Poisson areas, plus private latents
# per area (5 and 3 dims). Trials are 50 bins of 50 ms, short enough for the
# original code. The inference sections use the true parameters; the EM section
# learns them from a random initialisation.
#
# Runs from the repository root or from `examples/`.

# %%
import sys
import time
import types
import warnings
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "core" / "inference.py").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "core"))
sys.modules.setdefault("csr", types.ModuleType("csr"))  # imported by core/, not used here
warnings.filterwarnings("ignore", category=DeprecationWarning)

from data_processing_tools import parse_fullCov, parse_fullCov_latDim
from gen_synthetic_data import dataGen
from inference import inferTrial

import efgp
from efgp.legacy import model_from_pgpcca, store_posterior, trial_from_pgpcca

OLD, NEW = "#2a78d6", "#eb6834"  # original code, EFGP
TRUTH = "#3d3d3a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3dc", "grid.linewidth": 0.6, "lines.linewidth": 2})

# %% [markdown]
# ## Simulate

# %%
n_trials, n_time = 5, 50
data = dataGen(n_trials, T=n_time, infer=False).cca_input
groups = ["shared"] + [str(a) for a in data.area_list]
print("latent dims:", dict(zip(groups, data.zdims)))
print("tau (s):", [np.round(p["tau"], 2) for p in data.priorPar])

# %% [markdown]
# ## Original inference

# %%
old = {}
t0 = time.perf_counter()
for tr in range(n_trials):
    mean, cov, _ = inferTrial(data, tr)
    _, cov_t, _ = parse_fullCov(data, mean, cov, n_time)
    mean_k, cov_k = parse_fullCov_latDim(data, mean, cov, n_time)
    old[tr] = dict(mean=mean_k, cov_t=cov_t, cov_k=cov_k)
time_old = (time.perf_counter() - t0) / n_trials
print(f"original: {time_old:.3f} s per trial")

# %% [markdown]
# ## EFGP inference
#
# `model_from_pgpcca` builds the model from the current parameters of `data`. The solver is built once; the
# first call compiles.

# %%
model = model_from_pgpcca(data)
solver = efgp.newton_cg_solver(model)
print("Fourier coefficients per latent:", model.prior.n_coefficients)

t0 = time.perf_counter()
efgp.infer_trial(model, trial_from_pgpcca(data, 0), solver=solver)
time_compile = time.perf_counter() - t0

t0 = time.perf_counter()
for tr in range(n_trials):
    post = efgp.infer_trial(model, trial_from_pgpcca(data, tr), solver=solver)
    jax.block_until_ready(post.cov_k)
    store_posterior(data, tr, model, post)  # same format as the original code
time_new = (time.perf_counter() - t0) / n_trials
print(f"EFGP: first call {time_compile:.2f} s (compilation), then {time_new:.3f} s per trial")

# %%
# All trials at once: pad, stack, vmap.
batch = efgp.stack_trials([trial_from_pgpcca(data, tr) for tr in range(n_trials)])
efgp.infer_trials(model, batch, solver=solver)  # compile
t0 = time.perf_counter()
jax.block_until_ready(efgp.infer_trials(model, batch, solver=solver).cov_k)
print(f"EFGP batched: {(time.perf_counter() - t0) / n_trials:.3f} s per trial")

new = {tr: data.posterior_inf[tr] for tr in range(n_trials)}

# %% [markdown]
# ## Posterior means and standard deviations, trial 0
#
# First dimension of each latent group. EFGP: mean and ± 2 sd band. Original: mean
# and ± 2 sd as dashed lines on top. Where only one color shows, the two coincide.

# %%
tr = 0
t = np.arange(n_time) * data.binSize / 1000
truth = data.ground_truth_latent[tr]
start = np.cumsum([0] + list(data.zdims))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharex=True)
for i, (ax, g) in enumerate(zip(axes, groups)):
    ax.plot(t, truth[:, start[i]], color=TRUTH, lw=1, label="ground truth")
    m, sd = np.asarray(new[tr].mean[i][0]), np.sqrt(new[tr].cov_t[i][:, 0, 0])
    ax.plot(t, m, color=NEW, lw=3, label="EFGP")
    ax.fill_between(t, m - 2 * sd, m + 2 * sd, color=NEW, alpha=0.15, lw=0)
    m, sd = old[tr]["mean"][i][0], np.sqrt(old[tr]["cov_t"][i][:, 0, 0])
    ax.plot(t, m, "--", color=OLD, lw=1.5, label="original")
    ax.plot(t, m - 2 * sd, "--", color=OLD, lw=1)
    ax.plot(t, m + 2 * sd, "--", color=OLD, lw=1)
    ax.set_title(f"{g}, dim 0")
    ax.set_xlabel("time (s)")
axes[0].set_ylabel("latent")
fig.tight_layout()
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=3,
           loc="lower center", bbox_to_anchor=(0.5, 1.0))

# %% [markdown]
# ## Agreement over all trials and latents

# %%
def stack(res, key):
    out = []
    for tr in range(n_trials):
        r = res[tr]
        d = r[key] if isinstance(r, dict) else getattr(r, key)
        out.append(np.concatenate([np.asarray(d[i]).ravel() for i in range(len(groups))]))
    return np.concatenate(out)

def stack_sd(res):
    out = []
    for tr in range(n_trials):
        r = res[tr]
        c = r["cov_t"] if isinstance(r, dict) else r.cov_t
        out.append(np.concatenate([np.sqrt(np.diagonal(c[i], axis1=1, axis2=2)).ravel()
                                   for i in range(len(groups))]))
    return np.concatenate(out)

mean_old, mean_new = stack(old, "mean"), stack(new, "mean")
sd_old, sd_new = stack_sd(old), stack_sd(new)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (a, b, name) in zip(axes, [(mean_old, mean_new, "posterior mean"),
                                   (sd_old, sd_new, "posterior sd")]):
    lim = [min(a.min(), b.min()), max(a.max(), b.max())]
    ax.plot(lim, lim, color=TRUTH, lw=1)
    ax.scatter(a, b, s=8, color=NEW, alpha=0.5, lw=0)
    ax.set_xlabel(f"original {name}")
    ax.set_ylabel(f"EFGP {name}")
    ax.set_aspect("equal")
fig.tight_layout()

# %%
rows = []
for i, g in enumerate(groups):
    d_mean = max(np.abs(np.asarray(new[tr].mean[i]) - old[tr]["mean"][i]).max() for tr in range(n_trials))
    scale = max(np.abs(old[tr]["mean"][i]).max() for tr in range(n_trials))
    d_cov_t = max(np.abs(new[tr].cov_t[i] - old[tr]["cov_t"][i]).max() for tr in range(n_trials))
    d_cov_k = max(np.abs(new[tr].cov_k[i] - old[tr]["cov_k"][i]).max() for tr in range(n_trials))
    rows.append((g, d_mean, scale, d_cov_t, d_cov_k))
print(f"{'group':8s} {'max|Δmean|':>11s} {'max|mean|':>10s} {'max|Δcov_t|':>12s} {'max|Δcov_k|':>12s}")
for g, a, s, b, c in rows:
    print(f"{g:8s} {a:11.2e} {s:10.2f} {b:12.2e} {c:12.2e}")

# %% [markdown]
# ## Recovery of the ground truth
#
# Correlation between posterior mean and true latent, per latent dimension, pooled
# over trials. The two methods should be indistinguishable here.

# %%
def recovery(res):
    r = []
    for i in range(len(groups)):
        for k in range(data.zdims[i]):
            est = np.concatenate([np.asarray(res[tr]["mean"][i][k] if isinstance(res[tr], dict)
                                             else res[tr].mean[i][k]) for tr in range(n_trials)])
            true = np.concatenate([data.ground_truth_latent[tr][:, start[i] + k] for tr in range(n_trials)])
            r.append(np.corrcoef(est, true)[0, 1])
    return np.array(r)

r_old, r_new = recovery(old), recovery(new)
labels = [f"{g}{k}" for i, g in enumerate(groups) for k in range(data.zdims[i])]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(x - 0.2, r_old, 0.38, color=OLD, label="original")
ax.bar(x + 0.2, r_new, 0.38, color=NEW, label="EFGP")
ax.set_xticks(x, labels)
ax.set_ylabel("corr(mean, truth)")
ax.set_ylim(0, 1)
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()

# %% [markdown]
# ## Where the remaining difference comes from
#
# The means differ by about 0.5 % of their scale. Two candidate causes are the
# optimiser and the model. Re-optimising the original objective much more tightly
# moves its MAP by less than that, so the original solution is converged, and the
# difference is the model: the white-noise term `epsNoise * I` (sd ≈ 0.03 per bin)
# that the EFGP prior drops.

# %%
from inference import PpCCA_logLike, grad_PpCCA_logLike
from scipy.optimize import minimize

stim, x_list = data.get_observations(0)
args = (stim, x_list, data.priorPar, data.stimPar, data.xPar, data.binSize, data.epsNoise)
f = lambda z: -PpCCA_logLike(z, *args)
grad = lambda z: -grad_PpCCA_logLike(z, *args)

z_old = inferTrial(data, 0)[0]
z_new = np.concatenate([np.asarray(new[0].mean[i]).T.ravel() for i in range(len(groups))])
tight = minimize(f, z_old, jac=grad, method="L-BFGS-B",
                 options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 20000})
print(f"|original MAP - tightly re-optimised|: {np.abs(tight.x - z_old).max():.1e}")
print(f"|EFGP MAP     - tightly re-optimised|: {np.abs(tight.x - z_new).max():.1e}")

# %% [markdown]
# ## Learning the parameters with EM
#
# The M-steps in `core/` take the latents only from `data.posterior_inf`, which
# `store_posterior` fills in the same format. So EM with EFGP is the original EM
# loop with two parts swapped:
#
# ```python
# tau_bounds = (0.05, 2.0)  # seconds
# expectation_maximization(
#     data, method="L-BFGS-B", tau_bounds=tau_bounds,
#     e_step=EFGPEStep(tau_bounds),          # EFGP Laplace inference
#     gp_m_step=GPTimescaleMStep(tau_bounds),  # JAX M-step for the time constants
# )
# ```
#
# * `EFGPEStep`: the EFGP prior is sized once for the shortest allowed time
#   constant, so the coefficient shape never changes during the fit and the solver
#   compiles only once.
# * `GPTimescaleMStep`: the same objective as the original GP M-step, with all time
#   constants fitted at once (one independent problem per latent), trials of equal
#   length summed first, and the bounds enforced by a change of variables.
# * The Poisson M-step now starts from the current parameters instead of zeros
#   (`warm_start_poisson=True`, the default); it is convex, so the optimum is the
#   same.
#
# The "original" fit below is the old pipeline unchanged: Laplace E-step of
# `core/inference.py`, L-BFGS-B GP M-step, Poisson M-step from zeros. Both fits use
# the same bounds.
#
# Both fits below start from the same random initialisation (`initializeParam`).
# `method="L-BFGS-B"` is used for the Poisson M-step because the default
# `"sparse-Newton"` needs the `csr` package.
#
# The printed "LL" of the EM loop is the expected complete-data log-likelihood under
# each E-step's own posterior, so the two curves show that each fit improves; their
# values are not comparable. The comparison is on what was learned: the GP time
# constants against the truth, and the firing rates against the true rates.
#
# What to expect: both fits recover the firing rates equally well. Neither moves the
# time constants far from their initial values in 10 iterations, and both stay
# below the truth. This comes from the GP M-step of `core/learnGPParams.py`, which
# both fits share, not from the E-step.

# %%
import contextlib
import copy
import io

from expectation_maximization import expectation_maximization
from efgp.legacy import EFGPEStep, GPTimescaleMStep

n_iter = 10
tau_bounds = (0.05, 2.0)
em_data = dataGen(10, T=n_time, infer=False, setTruePar=False).cca_input
fits = {}
pipelines = {
    "original": dict(e_step=None, gp_m_step=None, warm_start_poisson=False),
    "EFGP": dict(e_step=EFGPEStep(tau_bounds), gp_m_step=GPTimescaleMStep(tau_bounds)),
}
for name, options in pipelines.items():
    d = copy.deepcopy(em_data)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # the EM loop prints a lot
        ll = expectation_maximization(d, maxIter=n_iter, method="L-BFGS-B",
                                      tau_bounds=tau_bounds, **options)
    fits[name] = dict(data=d, ll=np.asarray(ll), seconds=time.perf_counter() - t0)
    print(f"{name}: {fits[name]['seconds'] / n_iter:.1f} s per EM iteration")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))

ax = axes[0]
for name, color in [("original", OLD), ("EFGP", NEW)]:
    ax.plot(np.arange(1, n_iter + 1), fits[name]["ll"], "o-", ms=4, color=color, label=name)
ax.set_xlabel("EM iteration")
ax.set_ylabel("expected complete-data LL")

ax = axes[1]
# Within a latent group the fit is defined up to a rotation of the latents, so learned
# dimension k need not match true dimension k: compare sorted time constants per group.
def sorted_tau(prior_par):
    return np.concatenate([np.sort(p["tau"]) for p in prior_par])

tau_true = sorted_tau(em_data.ground_truth_priorPar)
lim = [0, 1.1 * max(tau_true.max(), *(sorted_tau(f["data"].priorPar).max() for f in fits.values()))]
ax.plot(lim, lim, color=TRUTH, lw=1)
ax.scatter(tau_true, sorted_tau(em_data.priorPar), s=20, color="#b4b2a9", label="initial", zorder=2)
for name, color, marker in [("original", OLD, "o"), ("EFGP", NEW, "s")]:
    ax.scatter(tau_true, sorted_tau(fits[name]["data"].priorPar), s=24, color=color, marker=marker,
               label=name, zorder=3)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")
ax.set_xlabel("true τ, sorted per group (s)")
ax.set_ylabel("learned τ, sorted per group (s)")


def log_rates(d, tr):
    """Log firing rates of all units from the posterior mean and the learned parameters."""
    post = d.posterior_inf[tr]
    return np.hstack([post.mean[0].T @ par["W0"].T + post.mean[k + 1].T @ par["W1"].T + par["d"]
                      for k, par in enumerate(d.xPar)])


ax = axes[2]
true_log_rate = np.concatenate([np.log(em_data.preproc.data[tr]["R"]).ravel() for tr in em_data.trialDur])
lim = [true_log_rate.min(), true_log_rate.max()]
ax.plot(lim, lim, color=TRUTH, lw=1)
for name, color in [("original", OLD), ("EFGP", NEW)]:
    d = fits[name]["data"]
    est = np.concatenate([log_rates(d, tr).ravel() for tr in d.trialDur])
    r = np.corrcoef(est, true_log_rate)[0, 1]
    ax.scatter(true_log_rate, est, s=3, color=color, alpha=0.3, lw=0, label=f"{name} (r = {r:.3f})")
ax.set_xlabel("true log rate")
ax.set_ylabel("fitted log rate")
ax.set_aspect("equal")

fig.tight_layout()
handles = {}
for a in axes:
    for hnd, lab in zip(*a.get_legend_handles_labels()):
        handles.setdefault(lab, hnd)
fig.legend(handles.values(), handles.keys(), frameon=False, ncol=len(handles),
           loc="lower center", bbox_to_anchor=(0.5, 1.0), markerscale=2)

# %% [markdown]
# ## Missing task variables
#
# Set the task variables of trial 0 to NaN between 0.75 s and 1.5 s.
#
# * The original `get_observations` deletes those 15 time points, spikes included,
#   and builds the kernel on the remaining 35 consecutive indices. The time points
#   on either side of the gap are treated as 50 ms apart instead of 800 ms.
# * EFGP keeps the time points, uses the spikes there, and leaves the task
#   variables out of the likelihood only where they are missing.

# %%
gap = slice(15, 30)
for var in data.var_list:
    data.preproc.covariates[var][0] = data.preproc.covariates[var][0].copy()
    data.preproc.covariates[var][0][gap] = np.nan
kept = np.ones(n_time, bool)
kept[gap] = False

mean_gap_old, _, _ = inferTrial(data, 0)
z0_old = mean_gap_old[: kept.sum() * data.zdims[0]].reshape(kept.sum(), data.zdims[0])

post_gap = efgp.infer_trial(model, trial_from_pgpcca(data, 0), solver=solver)
z0_new = np.asarray(model.split_latents(post_gap.mean)["shared"])

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axvspan(t[gap][0], t[gap][-1], color="#e4e3dc", lw=0)
ax.text(t[gap].mean(), ax.get_ylim()[1], "task variables missing", ha="center", va="top", color="#73726c")
ax.plot(t, truth[:, 0], color=TRUTH, lw=1, label="ground truth")
ax.plot(t[kept], z0_old[:, 0], "o-", ms=3, color=OLD, label="original (gap removed, time compressed)")
ax.plot(t, z0_new[:, 0], color=NEW, label="EFGP (gap masked)")
ax.set_xlabel("time (s)")
ax.set_ylabel("shared, dim 0")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
