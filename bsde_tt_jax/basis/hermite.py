from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["make_hermite_polynomials"]


def make_hermite_polynomials(degree: int) -> Callable:
    m = degree + 1

    def _basis(x: float) -> Float[Array, "m"]:
        def _recc(carry, n):
            pn, pnm1 = carry
            pnp1 = (x * pn - jnp.sqrt(n) * pnm1) / jnp.sqrt(n + 1)
            return (pnp1, pn), pnp1

        _, val = jax.lax.scan(_recc, (1.0, 0.0), jnp.arange(0, m - 1))
        val = jnp.hstack((1.0, val))
        return val

    return jax.jit(_basis)
