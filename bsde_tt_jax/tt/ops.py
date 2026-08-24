from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bsde_tt_jax.tt.core import TensorTrain, TTCore

__all__ = ["tt_at", "tt_dot", "tt_matvec", "tt_norm", "tt_orth_left", "tt_orth_right"]


@jax.jit
def tt_dot(a: TensorTrain, b: TensorTrain) -> float:
    assert a.order == b.order, "The two TTs must be of the same order."
    res = jnp.einsum("oab,oac->obc", a.cores[0], b.cores[0])
    for i in range(1, a.order):
        res = jnp.einsum("obc,bnd,cnf->odf", res, a.cores[i], b.cores[i])
    return jnp.sum(res)


@jax.jit
def tt_norm(a: TensorTrain) -> float:
    return jnp.sqrt(abs(tt_dot(a, a)))


@jax.jit
def tt_orth_left(core: TTCore) -> tuple[TTCore, TTCore]:
    r0, m, r1 = core.shape
    Q, R = jnp.linalg.qr(core.reshape(r0 * m, r1))
    return Q.reshape(r0, m, r1), R


@jax.jit
def tt_orth_right(core: TTCore) -> tuple[TTCore, TTCore]:
    r0, m, r1 = core.shape
    Q, R = jnp.linalg.qr(core.reshape(r0, m * r1).T)
    QT = Q.T
    return QT.reshape(r0, m, r1), R.T


@jax.jit
def tt_matvec(tt: TensorTrain, x: Sequence[Float[Array, "m"]]) -> Float[Array, "r0 rd"]:
    res = jnp.einsum("oab,a->ob", tt.cores[0], x[0])
    for i in range(1, tt.order):
        res = jnp.einsum("ob,bnd,n->od", res, tt.cores[i], x[i])
    return res


@partial(jax.jit, static_argnames=("indices"))
def tt_at(tt: TensorTrain, indices: Sequence[int]) -> Float[Array, "r0 rd"]:
    matrices = [core[:, idx, :] for core, idx in zip(tt.cores, indices)]
    return jnp.linalg.multi_dot(matrices)


@jax.jit
def tt_norm_diff(a: TensorTrain, b: TensorTrain) -> float:
    return jnp.sqrt(abs(tt_dot(a, a) + tt_dot(b, b) - 2 * tt_dot(a, b)))
