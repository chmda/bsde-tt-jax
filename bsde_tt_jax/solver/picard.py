import math
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

from bsde_tt_jax.basis.hermite import make_hermite_polynomials
from bsde_tt_jax.io.results import SolverResult
from bsde_tt_jax.solver.problem import BSDEProblem
from bsde_tt_jax.tt.als import tt_als_christoffel
from bsde_tt_jax.tt.compute import (
    _hermite_derivative_core,
    compute_t0,
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


# ---------------
# TTHistoryBuffer
# this allows a single JIT compilation of fit_tt_chaos
# by keeping the structure the same instead of using list
# ---------------


class TTHistoryBuffer(NamedTuple):
    cores: list[Float[Array, "max_len r0 m r1"]]  # one stacked array per TT-mode
    valid_len: Int[Array, ""]


def init_history_buffer(
    dims: list[int], ranks: list[int], max_len: int, dtype=float
) -> TTHistoryBuffer:
    core_shapes = [(ranks[k], dims[k], ranks[k + 1]) for k in range(len(dims))]
    cores = [jnp.zeros((max_len,) + shape, dtype=dtype) for shape in core_shapes]
    return TTHistoryBuffer(cores=cores, valid_len=jnp.array(0))


def push_history(buf: TTHistoryBuffer, tt: TensorTrain) -> TTHistoryBuffer:
    idx = buf.valid_len
    new_cores = [
        jax.lax.dynamic_update_index_in_dim(hist_core, core[None], idx, axis=0)
        for hist_core, core in zip(buf.cores, tt.cores)
    ]
    return TTHistoryBuffer(cores=new_cores, valid_len=idx + 1)


def get_tt(buf: TTHistoryBuffer, idx) -> TensorTrain:
    cores = [
        jax.lax.dynamic_index_in_dim(c, idx, axis=0, keepdims=False) for c in buf.cores
    ]
    return TensorTrain(cores=cores)


# -----------------------------------------------
# Helpers for `fit_tt_chaos`, saving computations
# -----------------------------------------------
def _tail_boundaries(tt: TensorTrain, d_b: int, N: int) -> list[Float[Array, "r 1"]]:
    r"""Precompute R_j := collapse of tt.cores[j*d_b:], for j = 0..N.
    It computes ONCE per tt and shared across every time step
    and every Malliavin component, instead of being rebuilt inside
    `_truncate_tt` on every one of the O(N) calls `evaluate_conditional_expectation`
    / `evaluate_z` currently make per historical iterate.
    """
    r_final = tt.cores[-1].shape[-1]
    boundaries: list[Float[Array, "r 1"]] = [None] * (N + 1)
    R = jnp.ones((r_final, 1), dtype=tt.cores[0].dtype)
    boundaries[N] = R
    for j in range(N - 1, -1, -1):
        block = tt.cores[j * d_b : (j + 1) * d_b]
        for core in reversed(block):
            R = core[:, 0, :] @ R
        boundaries[j] = R
    return boundaries


def _mode_bases(tt: TensorTrain) -> tuple[list, list]:
    """Hermite bases (normal, and Malliavin-derivative-reduced) for every
    core. Depends only on `tt`'s shapes (static), computed once per tt."""
    normal = [make_hermite_polynomials(core.shape[1] - 1) for core in tt.cores]
    deriv = [make_hermite_polynomials(max(core.shape[1] - 2, 0)) for core in tt.cores]
    return normal, deriv


def _trajectory(
    tt: TensorTrain,
    d_b: int,
    h: float,
    xi_std: Float[Array, "N d_b"],
) -> tuple[Float[Array, "N d"], Float[Array, "N d d_b"]]:
    r"""Compute (E[tt|F_tj])_{j=1}^N and (D_tj E[tt|F_tj])_{j=1}^N for one
    sample path.
    """
    N = xi_std.shape[0]
    d = tt.cores[0].shape[0]
    tail = _tail_boundaries(tt, d_b, N)
    normal_bases, deriv_bases = _mode_bases(tt)

    def step(L, j):
        # L: (d, r_{j*d_b}) running left state; j is the 0-indexed step (-> t_{j+1})
        block_start = j * d_b
        prefixes = [L]
        A_list = []
        for l in range(d_b):
            k = block_start + l
            phi = normal_bases[k](xi_std[j, l])
            A = jnp.einsum("imr,m->ir", tt.cores[k], phi)
            A_list.append(A)
            prefixes.append(prefixes[-1] @ A)
        L_next = prefixes[-1]  # (d, r_{(j+1)*d_b})

        R_j = tail[j + 1]  # (r_{(j+1)*d_b}, 1)
        suffixes = [None] * (d_b + 1)
        suffixes[d_b] = R_j
        for l in range(d_b - 1, -1, -1):
            suffixes[l] = A_list[l] @ suffixes[l + 1]

        y_j = (L_next @ R_j)[:, 0]  # (d,)

        z_cols = []
        for l in range(d_b):
            k = block_start + l
            deriv_core = _hermite_derivative_core(tt.cores[k], h)
            phi_prime = deriv_bases[k](xi_std[j, l])
            A_prime = jnp.einsum("imr,m->ir", deriv_core, phi_prime)
            z_cols.append((prefixes[l] @ A_prime @ suffixes[l + 1])[:, 0])
        z_j = jnp.stack(z_cols, axis=1)  # (d, d_b)

        return L_next, (y_j, z_j)

    L = jnp.eye(d, dtype=xi_std.dtype)
    # _, (Y, Z) = jax.lax.scan(step, L, jnp.arange(N))
    Y_list, Z_list = [], []
    for j in range(N):
        L, (y_j, z_j) = step(L, j)
        Y_list.append(y_j)
        Z_list.append(z_j)
    Y = jnp.stack(Y_list, axis=0)
    Z = jnp.stack(Z_list, axis=0)
    return Y, Z


# -------------
# Picard solver
# -------------


@partial(jax.jit, static_argnames=("problem",))
def fit_tt_chaos(
    key: PRNGKeyArray,
    problem: BSDEProblem,
    tt_hist: TTHistoryBuffer,
) -> tuple[TensorTrain, tuple[float, int]]:
    tmp_cores = [tt_hist.cores[i][0, :, :, :] for i in range(len(tt_hist.cores))]
    dims = [core.shape[1] for core in tmp_cores]
    ranks = [core.shape[0] for core in tmp_cores] + [tmp_cores[-1].shape[2]]
    init_key, sampler_key = random.split(key, 2)

    xi = problem.xi
    f = problem.f
    h = problem.T / problem.N

    # def _target(x: Float[Array, "N*d_b"]) -> Float[Array, "d"]:
    #     inputs = jnp.reshape(x, (problem.N, problem.d_b))

    #     # compute Y_tj^{(q)}, Z_tj^{(q)}
    #     Y = jnp.zeros((problem.N, problem.d))
    #     Z = jnp.zeros((problem.N, problem.d, problem.d_b))
    #     for q in range(K):
    #         current_tt = tt_hist[q]
    #         sum_term = 0.0
    #         new_Y = Y.copy()
    #         new_Z = Z.copy()
    #         for j in range(1, problem.N + 1):
    #             # compute Y_tj = E[tt | F_tj] - sum_i=1^j h f(ti, Y_ti, Z_ti)
    #             cond_exp = evaluate_conditional_expectation(
    #                 current_tt, j, problem.d_b, inputs[:j, :]
    #             )
    #             sum_term = sum_term + h * f(j * h, Y[j - 1, :], Z[j - 1, :, :])
    #             new_Y = new_Y.at[j - 1, :].set(cond_exp - sum_term)
    #             new_Z = new_Z.at[j - 1, :, :].set(
    #                 evaluate_z(current_tt, j, problem.d_b, h, inputs[:j, :])
    #             )
    #         Y = new_Y
    #         Z = new_Z

    #     # compute F^{(q)}
    #     BT = jnp.sqrt(h) * jnp.sum(inputs, axis=0)
    #     F = xi(BT)
    #     for i in range(1, problem.N + 1):
    #         F = F + h * f(i * h, Y[i - 1, :], Z[i - 1, :])
    #     return F

    def _target(x: Float[Array, "N*d d_b"]) -> Float[Array, "d"]:
        inputs = jnp.reshape(x, (problem.N, problem.d_b))
        times = jnp.arange(1, problem.N + 1) * h

        Y = jnp.zeros((problem.N, problem.d))
        Z = jnp.zeros((problem.N, problem.d, problem.d_b))

        def q_body(q, carry):
            Y, Z = carry
            current_tt = get_tt(tt_hist, q)  # see point 3
            cond_exp_Y, cond_exp_Z = _trajectory(current_tt, problem.d_b, h, inputs)
            f_vals = jax.vmap(f)(times, Y, Z)  # uses OLD Y,Z = Y^{(q)}
            driver_cumsum = h * jnp.cumsum(
                f_vals, axis=0
            )  # sum_{i<=j} h f(ti,Y_ti,Z_ti)
            return cond_exp_Y - driver_cumsum, cond_exp_Z

        Y, Z = jax.lax.fori_loop(0, tt_hist.valid_len, q_body, (Y, Z))

        BT = jnp.sqrt(h) * jnp.sum(inputs, axis=0)
        f_final = jax.vmap(f)(times, Y, Z)
        return xi(BT) + h * jnp.sum(f_final, axis=0)

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


def picard_step(
    key: PRNGKeyArray,
    problem: BSDEProblem,
    tt_hist: TTHistoryBuffer,
) -> PicardResult:
    new_tt, (als_stag, als_iters) = fit_tt_chaos(key, problem, tt_hist)
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
    tt_hist = init_history_buffer(dims, ranks, problem.picard.max_iters)
    Y0_hist = []
    Z0_hist = []

    residual_history = []
    q = -1
    for q in range(problem.picard.max_iters):
        key, step_key = random.split(key)
        result = picard_step(step_key, problem, tt_hist)
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
        tt_hist = push_history(tt_hist, prev_tt)
        Y0_hist.append(result.Y0)
        Z0_hist.append(result.Z0)

    F_hist = [
        TensorTrain([tt_hist.cores[j][i, :, :] for j in range(len(tt_hist.cores))])
        for i in range(problem.picard.max_iters)
    ]
    return SolverResult(
        name=problem.name,
        d=problem.d,
        d_b=problem.d_b,
        N=problem.N,
        T=problem.T,
        F_hist=F_hist,
        Y0_hist=Y0_hist,
        Z0_hist=Z0_hist,
        n_iterations=q,
        residual_history=residual_history,
        reference_Y0=problem.reference.Y0 if problem.reference else None,
        reference_Z0=problem.reference.Z0 if problem.reference else None,
    )
