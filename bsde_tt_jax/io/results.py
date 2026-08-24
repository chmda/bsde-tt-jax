import json
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
from jaxtyping import Array, Float

from bsde_tt_jax.tt.core import TensorTrain

__all__ = ["SolverResult", "load_result", "save_result"]


@dataclass
class SolverResult:
    name: str
    d: int
    d_b: int
    N: int
    T: float

    F: TensorTrain
    Y0: Float[Array, "d"]
    Z0: Float[Array, "d d_b"]

    n_iterations: int
    converged: bool
    residual_history: list[float] = field(
        default_factory=list
    )  # ||Y^(q+1)-Y^(q)|| per iter

    reference_Y0: Float[Array, "d"] | None = None
    reference_Z0: Float[Array, "d d_b"] | None = None

    def errors(self) -> dict[str, float] | None:
        """Absolute/relative error of Y0, Z0 against the YAML reference, if given."""
        if self.reference_Y0 is None:
            return None
        y0_ref = jnp.asarray(self.reference_Y0)
        y0_err_abs = float(jnp.linalg.norm(self.Y0 - y0_ref))
        y0_err_rel = y0_err_abs / (float(jnp.linalg.norm(y0_ref)) + 1e-14)
        out = {"Y0_abs_error": y0_err_abs, "Y0_rel_error": y0_err_rel}
        if self.reference_Z0 is not None:
            z0_ref = jnp.asarray(self.reference_Z0)
            z0_err_abs = float(jnp.linalg.norm(self.Z0 - z0_ref))
            z0_err_rel = z0_err_abs / (float(jnp.linalg.norm(z0_ref)) + 1e-14)
            out["Z0_abs_error"] = z0_err_abs
            out["Z0_rel_error"] = z0_err_rel
        return out


def save_result(result: SolverResult, path: str | Path) -> None:
    """Save one BSDE's result as a directory: arrays in .npz, metadata in .json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    jnp.savez(
        path / "arrays.npz",
        F=result.F,
        Y0=result.Y0,
        Z0=result.Z0,
        reference_Y0=result.reference_Y0
        if result.reference_Y0 is not None
        else jnp.array([]),
        reference_Z0=result.reference_Z0
        if result.reference_Z0 is not None
        else jnp.array([]),
    )
    meta = {
        "name": result.name,
        "d": result.d,
        "d_b": result.d_b,
        "N": result.N,
        "T": result.T,
        "n_iterations": result.n_iterations,
        "converged": result.converged,
        "residual_history": result.residual_history,
        "errors": result.errors(),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def load_result(path: str | Path) -> SolverResult:
    path = Path(path)
    arrays = jnp.load(path / "arrays.npz")
    meta = json.loads((path / "meta.json").read_text())

    ref_y0 = arrays["reference_Y0"]
    ref_z0 = arrays["reference_Z0"]

    return SolverResult(
        name=meta["name"],
        d=meta["d"],
        d_b=meta["d_b"],
        N=meta["N"],
        T=meta["T"],
        F=TensorTrain(arrays["F"]),
        Y0=arrays["Y0"],
        Z0=arrays["Z0"],
        n_iterations=meta["n_iterations"],
        converged=meta["converged"],
        residual_history=meta["residual_history"],
        reference_Y0=ref_y0 if ref_y0.size else None,
        reference_Z0=ref_z0 if ref_z0.size else None,
    )
