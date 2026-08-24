import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from jaxtyping import Array, Float, PRNGKeyArray

from bsde_tt_jax.basis import Basis
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

    return jnp.linalg.lstsq(A, b)


def _sample_1d(
    key: PRNGKeyArray, basis: Basis, n: int, grid_size: int = 256
) -> tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "n m"]]:
    # 1D sampling by discretizing the support and evaluating the density on it
    p = 1.0 - 2e-8
    q = norm.ppf(p)
    grid = jnp.linspace(-q, q, num=grid_size)
    phi = jax.vmap(basis)(grid)  # (grid_size, m)
    inv_christoffel = jnp.sum(phi**2, axis=1) / phi.shape[1]
    density = inv_christoffel * norm.pdf(grid)  # (grid_size,)
    idx = random.choice(key, grid.shape[0], shape=(n,), p=density, replace=False)
    return grid[idx], 1.0 / inv_christoffel[idx], phi[idx]


def _weight_matrix(
    core: TTCore, state: Float[Array, "r r"], side: str
) -> Float[Array, "r r"]:
    """Q[m,m'] such that the conditional weight is phi(x)^T Q phi(x).

    side='left':  Q[m,m'] = tr(core[:,m,:]^T state core[:,m',:])   (state: r_in x r_in)
    side='right': Q[m,m'] = tr(core[:,m,:]   state core[:,m',:]^T) (state: r_out x r_out)
    """
    if side == "left":
        return jnp.einsum("amc,ab,bnc->mn", core, state, core)
    return jnp.einsum("amc,cd,and->mn", core, state, core)


def _update_state(
    core: TTCore, phi_x: Float[Array, "m"], state: Float[Array, "r r"], side: str
) -> Float[Array, "r r"]:
    A = jnp.einsum("amc,m->ac", core, phi_x)  # (r_in, r_out)
    if side == "left":
        return jnp.einsum("ac,ab,bd->cd", A, state, A)  # -> (r_out, r_out)
    return jnp.einsum("ac,cd,bd->ab", A, state, A)  # -> (r_in, r_in)


def _sample_sequential(
    key: PRNGKeyArray,
    cores: list[TTCore],
    modes: Sequence[int],
    bases: Sequence[Basis],
    side: str,
    n: int,
    grid_size: int = 256,
) -> Float[Array, "n k"]:
    dtype = cores[0].dtype
    d = cores[0].shape[0]
    if not modes:
        return (
            jnp.zeros((n, 0), dtype=dtype),
            jnp.ones((n,), dtype=dtype),
            jnp.ones((n, 1, 0), dtype=dtype),
        )

    p = 1.0 - 2e-8
    q = norm.ppf(p)
    grid = jnp.linspace(-q, q, num=grid_size)

    def _conditional_step(
        key: PRNGKeyArray, core: TTCore, basis: Basis, state: Float[Array, "r r"]
    ) -> tuple[float, Float[Array, "r r"]]:
        Q = _weight_matrix(core, state, side)

        phi = jax.vmap(basis)(grid)  # (n,m)
        inv_christoffel = jnp.einsum("ij,jk,ik->i", phi, Q, phi)
        density = inv_christoffel * norm.pdf(grid)
        idx = random.choice(key, grid.shape[0], p=density)
        x = grid[idx]

        psi = basis(x)
        new_state = _update_state(core, psi, state, side)
        return x, new_state

    if side == "left":
        R = jnp.eye(d, dtype=dtype)
    elif side == "right":
        R = jnp.ones((1, 1), dtype=dtype)

    state = jnp.broadcast_to(R, (n, R.shape[0], R.shape[0]))

    x = []
    for mode in modes:
        key, step_key = random.split(key)
        draw_keys = random.split(step_key, n)
        xk, state = jax.vmap(_conditional_step, in_axes=(0, None, None, 0))(
            draw_keys, cores[mode], bases[mode], state
        )
        x.append(xk)
    return jnp.stack(x, axis=1)


def _left_environment(
    cores: list[TTCore], k: int, bases: Sequence[Basis], x: Float[Array, "n d"]
) -> Float[Array, "n d r"]:
    n, d = x.shape[0], cores[0].shape[0]
    L = jnp.broadcast_to(jnp.eye(d, dtype=x.dtype), (n, d, d))
    for i in range(k):
        basis_k = jax.vmap(bases[i])(x[:, i])
        core_k = jnp.einsum("anb,sn->sab", cores[k], basis_k)
        L = jnp.einsum("sia,sab->sib", L, core_k)
    return L


def _right_environment(
    cores: list[TTCore], k: int, bases: Sequence[Basis], x: Float[Array, "n d"]
) -> Float[Array, "n r"]:
    order = len(cores)
    n = x.shape[0]
    r_final = cores[-1].shape[-1]
    R = jnp.ones((n, r_final), dtype=x.dtype)
    for i in range(order - 1, k - 1, -1):
        basis_k = jax.vmap(bases[i])(x[:, i])
        core_k = jnp.einsum("anb,sn->sab", cores[k], basis_k)
        R = jnp.einsum("sab,sb->sa", core_k, R)
    return R


def _christoffel_sampler(
    key: PRNGKeyArray,
    tt: TensorTrain,
    k: int,
    bases: Sequence[Basis],
    stability: float,
    prob: float,
    max_resample: int,
) -> tuple[Float[Array, "n d"], Float[Array, "n"], Float[Array, "n d r0 m r1"]]:
    # d nu(x) propto K_L(x_{<k}) * K_k(x_k) * K_R(x_{>k}) d mu(x)
    #
    n_feat = tt.cores[k].size
    c = int(stability + (1 - stability) * math.log(1 - stability))
    n = int(n_feat * math.log(n_feat / prob) / c)
    d = tt.cores[0].shape[0]
    order = tt.order

    def _sample(
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "n d"], Float[Array, "n"], Float[Array, "n d r0 m r1"]]:
        key_L, key_k, key_R = random.split(key, 3)

        x_left = _sample_sequential(key_L, tt.cores, list(range(k)), bases, "left", n)
        x_k, weights_k, feat_k = _sample_1d(key_k, bases[k], n)
        x_right = _sample_sequential(
            key_R, tt.cores, list(range(order - 1, k, -1)), bases, "right", n
        )
        x_right = x_right[:, ::-1]

        points = jnp.concatenate([x_left, x_k[:, None], x_right], axis=1)  # (n, order)

        L = _left_environment(tt.cores, k, bases, points)
        inv_christoffel_L = jnp.sum(L**2, axis=(1, 2)) / L.shape[2]
        weights_left = 1.0 / inv_christoffel_L
        R = _right_environment(tt.cores, k + 1, bases, points)
        inv_christoffel_R = jnp.sum(R**2, axis=-1) / R.shape[1]
        weights_right = 1.0 / inv_christoffel_R

        weights = weights_left * weights_k * weights_right
        feat = jnp.einsum("bli,bj,bk->blijk", L, feat_k, R)

        return points, weights, feat

    def _body(val):
        key, iters, _, _, _ = val
        key, subkey = random.split(key)
        points, weights, feat = _sample(subkey)
        return key, iters + 1, points, weights, feat

    def _cond(val):
        _, iters, _, weights, feat = val
        feat = feat.reshape(feat.shape[0] * feat.shape[1], -1)  # (n*d, r0*m*r1)
        w = jnp.repeat(weights, d)  # (n*d,)
        sqrt_w = jnp.sqrt(w)

        A = feat * sqrt_w[:, None]
        G = A.T @ A / n
        smallest_eigval = jnp.linalg.eigvalsh(G)[0]
        return (smallest_eigval <= 1 - stability) & (iters < max_resample)

    dtype = tt.cores[0].dtype
    init_val = (
        key,
        0,
        jnp.zeros((n, d), dtype=dtype),
        jnp.zeros((n,), dtype=dtype),
        jnp.zeros((n, d) + tt.cores[k].shape, dtype=dtype),
    )
    _, _, points, weights, feat = jax.lax.while_loop(_cond, _body, init_val)
    return points, weights, feat


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
        points, weights, feat = _christoffel_sampler(
            key, TensorTrain(cores), k, bases, stability, prob, max_resample
        )
        y = jax.vmap(target)(points)
        new_core = _solve_core(feat, weights, y, regularization)
        new_core = new_core.reshape(cores[k].shape)
        cores[k] = new_core
        return cores

    order = init_tt.order
    cores = init_tt.cores
    prev_tt = init_tt
    for sweep in range(max_iters):
        # Left-to-right pass: solve core k, then left-orthonormalize it
        # and push the R-factor into core k+1
        for k in range(order - 1):
            key, subkey = random.split(key)
            cores = _sweep_core(subkey, cores, k)
            Q, R = tt_orth_left(cores[k])
            cores[k] = Q
            cores[k + 1] = jnp.einsum("ij,jkl->", R, cores[k + 1])

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
        if residual < tol:
            break

        prev_tt = TensorTrain(cores)
