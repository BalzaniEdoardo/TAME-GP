"""Matrix-free Newton-CG, built on the second-order solver framework of nemos.

A nemos second-order solver is put together from a *curvature* model, a
*direction* and a *line search*. nemos' Newton curvature already takes a
constructor, ``hess_fn(params, *args)``, calls it once per iteration and passes
whatever it returns to the direction. So a matrix-free Newton-CG only needs:

* a ``hess_fn`` that returns a :class:`NewtonSystem` (a lineax operator plus an
  optional preconditioner) instead of a dense Hessian;
* :class:`ConjugateGradientDirection`, which solves ``A d = -grad`` with
  ``lineax.CG``, passing the preconditioner through ``options={"preconditioner": P}``.

:class:`NewtonCG` is nemos' Newton (curvature, Armijo line search, convergence test,
jitted loop) with this direction. Nothing in this module depends on the GP model;
the EFGP-specific part is the ``hess_fn`` built by :func:`efgp_newton_system`.

.. note::
    nemos' ``NewtonCurvature.hvp`` still assumes a dense tensor. CG does not call
    it, so this works unchanged. A port to nemos should make ``hvp`` accept a
    :class:`NewtonSystem` too, so other directions can use it.

    This imports private nemos modules (``nemos.solvers._second_order``), as of the
    ``development`` branch at commit ``21a0fc88e``.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import lineax as lx
import optax
from nemos.regularizer import UnRegularized
from nemos.solvers._second_order._direction import AbstractDirection
from nemos.solvers._second_order._linesearches import ArmijoBacktracking
from nemos.solvers._second_order._newton import (
    DEFAULT_ATOL,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    BaseNewtonSolver,
)

from .preconditioners import latent_block_preconditioner


class NewtonSystem(eqx.Module):
    """What ``hess_fn`` returns for :class:`NewtonCG`: the linear system of one step.

    Attributes
    ----------
    operator :
        The Hessian of the objective as a lineax operator. It must be tagged
        positive semidefinite, which ``lineax.CG`` requires.
    preconditioner :
        A positive definite operator close to ``operator^{-1}``, or ``None``.
    """

    operator: lx.AbstractLinearOperator
    preconditioner: lx.AbstractLinearOperator | None = None


class ConjugateGradientDirection(AbstractDirection):
    """Newton direction from a (preconditioned) conjugate gradient solve.

    The solve does not raise when CG stops at ``max_steps``: every CG iterate is a
    descent direction, so the line search can still use the partial solution.
    """

    cg: lx.CG

    def init(self, params):
        return None

    def update(self, params, grad, hvp_fn, hessian_tensor, direction_state):
        system: NewtonSystem = hessian_tensor
        options = {}
        if system.preconditioner is not None:
            options["preconditioner"] = system.preconditioner
        negative_grad = jax.tree.map(lambda g: -g, grad)
        solution = lx.linear_solve(
            system.operator, negative_grad, self.cg, options=options, throw=False
        )
        return solution.value, direction_state


class NewtonCG(BaseNewtonSolver):
    """Newton's method with matrix-free, preconditioned CG solves (lineax).

    The constructor follows the nemos solver convention, so an instance can be used
    wherever a nemos ``Newton`` instance can: ``params, state, aux =
    solver.run(init_params, *args)``.

    Parameters
    ----------
    unregularized_loss :
        Objective ``loss(params, *args)``.
    regularizer, regularizer_strength :
        A nemos regularizer. Only regularizers without a Hessian of their own
        (``UnRegularized``) are supported: nemos would add the penalty Hessian as a
        dense tensor to the operator. Put any penalty into the loss and the operator.
    has_aux :
        Whether the loss returns ``(value, aux)``.
    init_params :
        Example parameters, used for their structure.
    hess_fn :
        ``hess_fn(params, *args) -> NewtonSystem``: the Hessian of the loss at
        ``params`` as a lineax operator, and an optional preconditioner.
    cg :
        The lineax CG solver. Its tolerances control how exactly each Newton step is
        solved.
    jit, maxiter, tol, rtol :
        As for nemos' ``Newton``. ``tol`` and ``rtol`` act on the step norm.
    """

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer=None,
        regularizer_strength: float | None = None,
        has_aux: bool = False,
        init_params: Any = None,
        jit: bool = True,
        maxiter: int = DEFAULT_MAX_STEPS,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        *,
        hess_fn: Callable[..., NewtonSystem],
        cg: lx.CG | None = None,
    ) -> None:
        if regularizer is None:
            regularizer = UnRegularized()
        if regularizer._get_hess_fn(init_params, regularizer_strength, batch_axes=None) is not None:
            raise ValueError(
                "NewtonCG takes its Hessian from hess_fn and cannot add a dense "
                "regularizer Hessian to it. Add the penalty to the loss and to the "
                "operator instead, and pass UnRegularized()."
            )
        if cg is None:
            cg = lx.CG(rtol=1e-10, atol=1e-10, max_steps=1000)
        self.cg = cg

        penalized_loss = regularizer.penalized_loss(
            unregularized_loss, init_params, strength=regularizer_strength
        )
        super().__init__(
            unregularized_loss,
            regularizer,
            line_search=ArmijoBacktracking(
                optax.scale_by_backtracking_linesearch(30), penalized_loss
            ),
            direction_factory=lambda tag, prox: ConjugateGradientDirection(cg),
            regularizer_strength=regularizer_strength,
            has_aux=has_aux,
            init_params=init_params,
            jit=jit,
            maxiter=maxiter,
            tol=tol,
            rtol=rtol,
            hess_fn=hess_fn,
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return super().get_accepted_arguments() | {"cg"}


def efgp_newton_system(preconditioner=latent_block_preconditioner) -> Callable[..., NewtonSystem]:
    """``hess_fn`` for :class:`NewtonCG` on :func:`efgp.negative_log_joint`.

    Returns ``hess_fn(coefficients, model, trial) -> NewtonSystem`` with operator
    ``I + Phi^T H Phi`` and the preconditioner built by
    ``preconditioner(model, coefficients, trial, hessian_blocks)``.
    """

    def hess_fn(coefficients, model, trial) -> NewtonSystem:
        z = model.latents(coefficients, trial)
        blocks = model.negative_hessian(z, trial)
        return NewtonSystem(
            operator=model.precision_operator(trial, blocks),
            preconditioner=preconditioner(model, coefficients, trial, blocks),
        )

    return hess_fn
