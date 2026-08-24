import yaml

from bsde_tt_jax.expr.compile import parse_math
from bsde_tt_jax.solver.problem import (
    ALSSpec,
    BSDEProblem,
    ChaosSpec,
    ChristoffelSpec,
    PicardSpec,
)

__all__ = ["load_config"]


def _parse_chaos_spec(data: dict) -> ChaosSpec:
    degree = int(data["degree"])
    max_tt_rank = int(data["max_tt_rank"])
    return ChaosSpec(degree=degree, max_tt_rank=max_tt_rank)


def _parse_als_spec(data: dict) -> ALSSpec:
    max_iters = int(data["max_iters"])
    tol = float(data["tol"])
    regularization = data.get("regularization", None)
    return ALSSpec(max_iters=max_iters, tol=tol, regularization=regularization)


def _parse_picard_spec(data: dict) -> PicardSpec:
    max_iters = int(data["max_iters"])
    tol = float(data["tol"])
    return PicardSpec(max_iters=max_iters, tol=tol)


def _parse_christoffel_spec(data: dict) -> ChristoffelSpec:
    stability = float(data["stability"])
    prob = float(data["prob"])
    max_resample = int(data["max_resample"])
    return ChristoffelSpec(stability=stability, prob=prob, max_resample=max_resample)


def _parse_bsde(data: dict) -> BSDEProblem:
    name = data["name"]
    d = int(data["d"])
    db = int(data["db"])
    T = float(data["T"])
    N = int(data["N"])
    constants = data.get("constants", None)

    xi_expr = data["xi"]
    xi = parse_math(xi_expr, constants=constants)

    f_expr = data["f"]
    f = parse_math(f_expr, constants=constants)

    chaos_spec = _parse_chaos_spec(data["chaos"])
    als_spec = _parse_als_spec(data["als"])
    picard_spec = _parse_picard_spec(data["picard"])
    christoffel_spec = _parse_christoffel_spec(data["christoffel"])

    reference = data.get("reference", None)

    return BSDEProblem(
        name=name,
        d=d,
        d_b=db,
        T=T,
        N=N,
        xi=xi,
        f=f,
        chaos=chaos_spec,
        als=als_spec,
        picard=picard_spec,
        christoffel=christoffel_spec,
        reference=reference,
    )


def load_config(path: str) -> list[BSDEProblem]:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    bsdes = data["bsde"]
    return list(map(_parse_bsde, bsdes))
