from collections.abc import Callable

from jaxtyping import Array, Float

from bsde_tt_jax.basis.hermite import make_hermite_polynomials

__all__ = ["Basis", "make_hermite_polynomials"]

Basis = Callable[[float], Float[Array, "m"]]
