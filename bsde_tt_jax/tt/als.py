from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray

from bsde_tt_jax.basis import Basis
from bsde_tt_jax.sampling.christoffel import christoffel_sampler
from bsde_tt_jax.tt.core import TensorTrain, TTCore
from bsde_tt_jax.tt.ops import tt_norm_diff, tt_orth_left, tt_orth_right


def _solve_core(
    feat: Float[Array, "n d r0*m*r1"],
    weights: Float[Array, "n"],
    y: Float[Array, "n d"],
    regularization: float | None = None,
) -> Float[Array, "r0*m*r1"]:
    d = y.shape[1]

    feat = feat.reshape(feat.shape[0] * feat.shape[1], -1)  # (n*d, r0*m*r1)
    y = y.reshape(-1)
    w = jnp.repeat(weights, d)  # (n*d,)
    sqrt_w = jnp.sqrt(w)

    A = feat * sqrt_w[:, None]
    b = y * sqrt_w

    if regularization:
        AtA = A.T @ A
        Atb = A.T @ b
        reg = regularization * jnp.eye(AtA.shape[0], dtype=AtA.dtype)
        return jnp.linalg.solve(AtA + reg, Atb)

    return jnp.linalg.lstsq(A, b)[0]


def tt_als_christoffel(
    key: PRNGKeyArray,
    init_tt: TensorTrain,
    bases: Sequence[Basis],
    target: Callable[[Float[Array, "n"]], Float[Array, "d"]],
    max_iters: int,
    tol: float,
    regularization: float | None = None,
    stability: float = 0.5,
    prob: float = 0.5,
    max_resample: int = 10,
) -> tuple[TensorTrain, tuple[float, int]]:

    def _sweep_core(key: PRNGKeyArray, cores: list[TTCore], k: int) -> list[TTCore]:
        points, weights, feat = christoffel_sampler(
            key, TensorTrain(cores), k, bases, stability, prob, max_resample
        )
        y = jax.vmap(target)(points)
        new_core = _solve_core(feat, weights, y, regularization)
        new_core = new_core.reshape(cores[k].shape)
        cores[k] = new_core
        return cores

    order = init_tt.order

    def _als_step(key: PRNGKeyArray, prev_tt: TensorTrain):
        cores = list(map(jnp.copy, prev_tt.cores))
        # Left-to-right pass: solve core k, then left-orthonormalize it
        # and push the R-factor into core k+1
        for k in range(order - 1):
            key, subkey = random.split(key)
            cores = _sweep_core(subkey, cores, k)
            Q, R = tt_orth_left(cores[k])
            cores[k] = Q
            cores[k + 1] = jnp.einsum("ij,jkl->ikl", R, cores[k + 1])

        key, subkey = random.split(key)
        cores = _sweep_core(subkey, cores, order - 1)

        # Right-to-left pass: symmetric
        for k in range(order - 1, 0, -1):
            key, subkey = random.split(key)
            cores = _sweep_core(subkey, cores, k)
            Q, R = tt_orth_right(cores[k])
            cores[k] = Q
            cores[k - 1] = jnp.einsum("ijk,kl->ijl", cores[k - 1], R)

        key, subkey = random.split(key)
        cores = _sweep_core(subkey, cores, 0)

        residual = tt_norm_diff(prev_tt, TensorTrain(cores))
        return TensorTrain(cores), residual

    def _cond(val):
        _, residual, iters, _ = val
        return (residual >= tol) & (iters < max_iters)

    def _body(val):
        prev_tt, _, iters, key = val
        key, subkey = random.split(key)
        new_tt, residual = _als_step(subkey, prev_tt)
        return new_tt, residual, iters + 1, key

    opt_tt, residual, iters, _ = jax.lax.while_loop(
        _cond, _body, (init_tt, jnp.inf, 0, key)
    )

    return opt_tt, (residual, iters)
