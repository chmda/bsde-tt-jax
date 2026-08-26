import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray

from bsde_tt_jax.basis.hermite import make_hermite_polynomials
from bsde_tt_jax.io.results import SolverResult
from bsde_tt_jax.solver.problem import BSDEProblem
from bsde_tt_jax.tt.als import tt_als_christoffel
from bsde_tt_jax.tt.compute import (
    compute_t0,
    evaluate_conditional_expectation,
    evaluate_z,
)
from bsde_tt_jax.tt.core import TensorTrain
from bsde_tt_jax.tt.ops import tt_norm_diff


@dataclass(frozen=True)
class PicardResult:
    tt: TensorTrain
    Y0: Float[Array, "d"]
    Z0: Float[Array, "d d_b"]
    als_stag: float
    als_iters: int


def fit_tt_chaos(
    key: PRNGKeyArray,
    problem: BSDEProblem,
    tt_hist: list[TensorTrain],
    prev_tt: TensorTrain,
) -> tuple[TensorTrain, tuple[float, int]]:
    dims = prev_tt.dims
    ranks = prev_tt.ranks
    init_key, sampler_key = random.split(key, 2)

    xi = problem.xi
    f = problem.f
    h = problem.T / problem.N
    K = len(tt_hist)

    def _target(x: Float[Array, "N*d_b"]) -> Float[Array, "d"]:
        inputs = jnp.reshape(x, (problem.N, problem.d_b))

        # compute Y_tj^{(q)}, Z_tj^{(q)}
        Y = jnp.zeros((problem.N, problem.d))
        Z = jnp.zeros((problem.N, problem.d, problem.d_b))
        for q in range(K):
            current_tt = tt_hist[q]
            sum_term = 0.0
            new_Y = Y.copy()
            new_Z = Z.copy()
            for j in range(1, problem.N + 1):
                # compute Y_tj = E[tt | F_tj] - sum_i=1^j h f(ti, Y_ti, Z_ti)
                cond_exp = evaluate_conditional_expectation(
                    current_tt, j, problem.d_b, inputs[:j, :]
                )
                sum_term = sum_term + h * f(j * h, Y[j - 1, :], Z[j - 1, :, :])
                new_Y = new_Y.at[j - 1, :].set(cond_exp - sum_term)
                new_Z = new_Z.at[j - 1, :, :].set(
                    evaluate_z(current_tt, j, problem.d_b, h, inputs[:j, :])
                )
            Y = new_Y
            Z = new_Z

        # compute F^{(q)}
        BT = jnp.sqrt(h) * jnp.sum(inputs, axis=0)
        F = xi(BT)
        for i in range(1, problem.N + 1):
            F = F + h * f(i * h, Y[i - 1, :], Z[i - 1, :])
        return F

    degree = problem.chaos.degree
    init_tt = TensorTrain.random(init_key, dims, ranks)
    basis = make_hermite_polynomials(degree)
    bases = [basis] * init_tt.order
    new_tt, (stag, iters) = tt_als_christoffel(
        sampler_key,
        init_tt,
        bases,
        _target,
        max_iters=problem.als.max_iters,
        tol=problem.als.tol,
        regularization=problem.als.regularization,
        stability=problem.christoffel.stability,
        prob=problem.christoffel.prob,
        max_resample=problem.christoffel.max_resample,
    )
    return new_tt, (stag, iters)


# @jax.jit
def picard_step(
    key: PRNGKeyArray,
    problem: BSDEProblem,
    tt_hist: list[TensorTrain],
    prev_tt: TensorTrain,
) -> PicardResult:
    new_tt, (als_stag, als_iters) = jax.jit(fit_tt_chaos, static_argnames=("problem"))(
        key, problem, tt_hist, prev_tt
    )
    Y0, Z0 = compute_t0(new_tt, problem.d_b, problem.T / problem.N)
    return PicardResult(tt=new_tt, Y0=Y0, Z0=Z0, als_stag=als_stag, als_iters=als_iters)


def run(key: PRNGKeyArray, problem: BSDEProblem) -> SolverResult:
    dims = [problem.chaos.degree + 1] * (problem.d_b * problem.N)
    ranks = (
        [problem.d] + [problem.chaos.max_tt_rank] * (problem.d_b * problem.N - 1) + [1]
    )
    ranks = [ranks[0]] + [
        min(r, min(math.prod(dims[: k + 1]), math.prod(dims[k + 1 :])))
        for k, r in enumerate(ranks[1:])
    ]

    prev_tt = TensorTrain.zeros(dims, ranks)
    tt_hist = []
    Y0_hist = []
    Z0_hist = []
    prev_Y0 = jnp.zeros((problem.d,))

    residual_history = []
    converged = False

    for q in range(problem.picard.max_iters):
        key, step_key = random.split(key)
        result = picard_step(step_key, problem, tt_hist, prev_tt)
        jax.debug.print(
            "Picard iter {q}: ALS(stag={stag}, iters={iters}), Y0={Y0}, Z0={Z0}",
            q=q,
            stag=result.als_stag,
            iters=result.als_iters,
            Y0=result.Y0,
            Z0=result.Z0,
        )
        residual_history.append(tt_norm_diff(result.tt, prev_tt))

        prev_tt = result.tt
        tt_hist.append(prev_tt)
        Y0_hist.append(result.Y0)
        Z0_hist.append(result.Z0)

        y0_diff = float(jnp.linalg.norm(result.Y0 - prev_Y0))
        if y0_diff < problem.picard.tol:
            converged = True
            prev_Y0 = result.Y0
            break

        prev_Y0 = result.Y0

    return SolverResult(
        name=problem.name,
        d=problem.d,
        d_b=problem.d_b,
        N=problem.N,
        T=problem.T,
        F_hist=tt_hist,
        Y0_hist=Y0_hist,
        Z0_hist=Z0_hist,
        n_iterations=q,
        converged=converged,
        residual_history=residual_history,
        reference_Y0=problem.reference.Y0 if problem.reference else None,
        reference_Z0=problem.reference.Z0 if problem.reference else None,
    )
