from collections.abc import Sequence

import jax.numpy as jnp
from jax import random
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, PRNGKeyArray

__all__ = ["TTCore", "TensorTrain"]

TTCore = Float[Array, "r0 m r1"]


def _validate_ranks(dims: Sequence[int], ranks: Sequence[int]) -> bool:
    d = len(dims)
    assert len(ranks) == d + 1, "'ranks' must haven length d+1"
    for i in range(d):
        assert ranks[i + 1] <= ranks[i] * dims[i]
        assert ranks[i] <= dims[i] * ranks[i + 1]


@register_pytree_node_class
class TensorTrain:
    cores: Sequence[TTCore]

    def __init__(self, cores: Sequence[TTCore]):
        self.cores = cores

    @property
    def size(self) -> int:
        return sum(core.size for core in self.cores)

    @property
    def dims(self) -> list[int]:
        return [core.shape[1] for core in self.cores]

    @property
    def ranks(self) -> list[int]:
        return [core.shape[0] for core in self.cores] + [self.cores[-1].shape[2]]

    @property
    def order(self) -> int:
        return len(self.cores)

    def tree_flatten(self):
        return self.cores, None

    @classmethod
    def tree_unflatten(cls, aux, cores):
        return cls(cores)

    @staticmethod
    def random(
        key: PRNGKeyArray, dims: Sequence[int], ranks: Sequence[int]
    ) -> "TensorTrain":
        assert _validate_ranks(dims, ranks), "'dims' and 'ranks' are incompatible."

        def _random_core(key: PRNGKeyArray, r0: int, m: int, r1: int) -> TTCore:
            scale = 1.0 / jnp.sqrt(r0 * m)
            return scale * random.normal(key, (r0, m, r1))

        d = len(dims)
        subkeys = random.split(key, d)
        cores = list(map(_random_core, zip(subkeys, ranks[:d], ranks[1:])))
        return TensorTrain(cores)
