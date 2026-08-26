import jax.numpy as jnp
from jaxtyping import Array, Float

from bsde_tt_jax.basis.hermite import make_hermite_polynomials
from bsde_tt_jax.tt.core import TensorTrain, TTCore
from bsde_tt_jax.tt.ops import tt_at

__all__ = [
    "compute_t0",
    "conditional_expectation",
    "evaluate_conditional_expectation",
    "evaluate_z",
    "malliavin_derivative",
]


def _truncate_tt(cores: list[TTCore], idx: int) -> Float[Array, "rk 1"]:
    cores_slice = cores[idx:]
    h0_slice = [
        core[:, 0, :] for core in cores_slice
    ]  # expectation of linear combination of Hermite polynomials is a_0*E[H_0]=a_0
    if len(h0_slice) == 0:
        R = jnp.ones((1, 1), dtype=cores[0].dtype)
    elif len(h0_slice) == 1:
        R = h0_slice[0]
    else:
        R = jnp.linalg.multi_dot(h0_slice)  # shape: (r_idx, 1)
    return R


def conditional_expectation(tt: TensorTrain, j: int, d_b: int) -> TensorTrain:
    r"""Computes :math:`E[tt | \mathcal{F}_{t_j}]` as a TT over the first `j*d_b` modes.

    For `j=0`, this returns an order-1 TT, whose only "value" is the scalar :math:`E[tt]`.

    Parameters
    ----------
    tt : TensorTrain
        TT representing the random variable `F`.
    j : int
        Time step :math:`t_j`.
    d_b : int
        Dimension of the Brownian motion.

    Returns
    -------
    TensorTrain
        The truncated TT
    """
    order = tt.order
    cutoff = j * d_b

    if cutoff >= order:
        return tt

    if cutoff == 0:
        # everything collapses as a single core equal to E[tt]
        R = _truncate_tt(tt.cores, 0)
        core = R.reshape(R.shape[0], 1, 1)
        return TensorTrain(cores=[core])

    R = _truncate_tt(tt.cores, cutoff)  # (r_cutoff, 1)
    new_cores = list(tt.cores[: cutoff - 1])
    last = tt.cores[cutoff - 1]  # (r_{cutoff-1}, n_{cutoff-1}, r_cutoff)
    absorbed = jnp.einsum("ijk,kl->ijl", last, R)  # (r_{cutoff-1}, n_{cutoff-1}, 1)
    new_cores.append(absorbed)
    return TensorTrain(cores=new_cores)


def _hermite_derivative_core(core: TTCore, h: float) -> TTCore:
    r0, m, r1 = core.shape
    if m <= 1:
        # Derivative of a constant is 0.
        return jnp.zeros((r0, 1, r1), dtype=core.dtype)
    scale = jnp.sqrt(jnp.arange(1, m, dtype=core.dtype) / h)
    return core[:, 1:, :] * scale[None, :, None]


def malliavin_derivative(
    tt: TensorTrain, j: int, l: int, d_b: int, h: float
) -> TensorTrain:
    r"""Computes the Malliavin derivative :math:`D_{t_j}^{(l)} E[tt | \mathcal{F}_{t_j}]`
    with respect to the `l`-th dimension of the Brownian motion and at time :math:`t_j`.

    Parameters
    ----------
    tt : TensorTrain
        TT representing the random variable `F`.
    j : int
        Time steps :math:`t_j`.
    l : int
        Index of the Brownian motion.
    d_b : int
        Dimension of the Brownian motion.
    h : float
        Step-size.

    Returns
    -------
    TensorTrain
        The Malliavin derivative of `tt`.
    """
    cutoff = j * d_b
    k_star = (j - 1) * d_b + (l - 1)
    if not (0 <= k_star < cutoff):
        raise ValueError(
            f"(j={j}, l={l}) does not index a mode within the first {cutoff} modes."
        )

    R = _truncate_tt(tt.cores, cutoff)

    new_cores = []
    for k in range(cutoff):
        core = tt.cores[k]
        if k == k_star:
            core = _hermite_derivative_core(core, h)
        if k == cutoff - 1:
            core = jnp.einsum("ijk,kl->ijl", core, R)
        new_cores.append(core)

    return TensorTrain(cores=new_cores)


def compute_t0(
    tt: TensorTrain, d_b: int, h: float
) -> tuple[Float[Array, "d"], Float[Array, "d d_b"]]:
    order = tt.order

    zero_idx = jnp.zeros((order,), dtype=int)
    Y0 = tt_at(tt, zero_idx)[:, 0]

    Z0_cols = [
        tt_at(tt, zero_idx.at[l - 1].set(1))[:, 0] for l in range(1, d_b + 1)
    ]  # compute Z_{t0}^{(ell)}
    Z0 = jnp.stack(Z0_cols, axis=1) / jnp.sqrt(h)  # (d, d_b)

    return Y0, Z0


def evaluate_conditional_expectation(
    tt: TensorTrain, j: int, d_b: int, increments: Float[Array, "j d_b"]
) -> Float[Array, "d"]:
    r"""Evaluate :math:`E[tt | \mathcal{F}_{t_j}]` at one sample path's
    increments up to :math:`t_j`. These must be standardized increments,
    i.e. `increments[i,l] = (B[i,l] - B[i-1,l])/sqrt(h)`.
    """
    cond_tt = conditional_expectation(tt, j, d_b)
    bases = [make_hermite_polynomials(m - 1) for m in cond_tt.dims]
    flat_increments = jnp.ravel(increments)
    contractions = [
        jnp.einsum("ijk,j->ik", core, basis(increment))
        for core, basis, increment in zip(cond_tt.cores, bases, flat_increments)
    ]
    if len(contractions) == 1:
        R = contractions[0]
    else:
        R = jnp.linalg.multi_dot(contractions)
    return R[:, 0]


def evaluate_z(
    tt: TensorTrain, j: int, d_b: int, h: float, increments: Float[Array, "j d_b"]
) -> Float[Array, "d d_b"]:
    r"""Evaluate :math:`Z_{t_j} = (D_{t_j}^{(l)} E[tt | \mathcal{F}_{t_j}])_{l=1}^{d_b}` at one path.
    These must be standardized increments,
    i.e. `increments[i,l] = (B[i,l] - B[i-1,l])/sqrt(h)`.
    """
    flat_increments = jnp.ravel(increments)

    cols = []
    for l in range(1, d_b + 1):
        dtt = malliavin_derivative(tt, j, l, d_b, h)
        bases = [make_hermite_polynomials(m - 1) for m in dtt.dims]
        contractions = [
            jnp.einsum("ijk,j->ik", core, basis(increment))
            for core, basis, increment in zip(dtt.cores, bases, flat_increments)
        ]
        if len(contractions) == 1:
            R = contractions[0]
        else:
            R = jnp.linalg.multi_dot(contractions)
        cols.append(R[:, 0])

    return jnp.stack(cols, axis=1)
