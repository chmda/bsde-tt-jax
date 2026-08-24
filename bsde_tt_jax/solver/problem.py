from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from jaxtyping import Array, Float


class ReferenceSolution(NamedTuple):
    Y0: Float[Array, "d"]
    Z0: Float[Array, "d d_b"]


@dataclass(frozen=True)
class ChaosSpec:
    degree: int
    max_tt_rank: int


@dataclass(frozen=True)
class ALSSpec:
    max_iters: int
    tol: float
    regularization: float | None = None


@dataclass(frozen=True)
class PicardSpec:
    max_iters: int
    tol: float


@dataclass(frozen=True)
class BSDEProblem:
    name: str
    d: str
    d_b: str
    T: float
    N: int
    xi: Callable[[Float[Array, "d_b"]], Float[Array, "d"]]
    f: Callable[[float, Float[Array, "d"], Float[Array, "d d_b"]], Float[Array, "d"]]
    chaos: ChaosSpec
    als: ALSSpec
    picard: PicardSpec
    reference: ReferenceSolution | None = None
