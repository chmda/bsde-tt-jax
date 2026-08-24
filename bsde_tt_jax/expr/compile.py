from collections.abc import Callable

import jax
import sympy as sp


def parse_math(expr: str, constants: dict[str, float] | None = None) -> Callable:
    t = sp.symbols("t")
    y = sp.IndexedBase("y")
    z = sp.IndexedBase("z")
    local_dict = {"t": t, "y": y, "z": z}

    if constants is not None:
        constants_name = list(constants.keys())
        constants_symbols = sp.symbols(" ".join(constants_name))
        local_dict.update(
            {name: sym for name, sym in zip(constants_name, constants_symbols)}
        )

    sp_expr = sp.parse_expr(expr, local_dict=local_dict)

    if constants is not None:
        # replace constants by their values
        sp_expr.subs(
            {sym: val for sym, val in zip(constants_symbols, constants.values())}
        )

    jax_fn = sp.lambdify((t, y, z), sp_expr, modules="jax")

    return jax.jit(jax_fn)
