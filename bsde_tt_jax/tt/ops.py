from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bsde_tt_jax.tt.core import TensorTrain, TTCore


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
def matvec(tt: TensorTrain, x: Sequence[Float[Array, "m"]]) -> Float[Array, "r0 rd"]:
    res = jnp.einsum("oab,a->ob", tt.cores[0], x[0])
    for i in range(1, tt.order):
        res = jnp.einsum("ob,bnd,n->od", res, tt.cores[i], x[i])
    return res
