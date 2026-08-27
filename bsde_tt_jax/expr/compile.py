from collections.abc import Callable

import jax
import sympy as sp

__all__ = ["parse_driver", "parse_terminal"]


def parse_driver(expr: str, constants: dict[str, float] | None = None) -> Callable:
    t = sp.symbols("t")
    y = sp.IndexedBase("y")
    z = sp.IndexedBase("z")
    local_dict = {"t": t, "y": y, "z": z}

    if constants is not None:
        constants_name = list(constants.keys())
        constants_symbols = sp.symbols(" ".join(constants_name) + ",")
        local_dict.update(
            {name: sym for name, sym in zip(constants_name, constants_symbols)}
        )

    sp_expr = sp.parse_expr(expr, local_dict=local_dict)

    if constants is not None:
        # replace constants by their values
        sp_expr = sp_expr.subs(
            {sym: val for sym, val in zip(constants_symbols, constants.values())}
        )

    jax_fn = sp.lambdify((t, y, z), sp_expr, modules="jax")

    return jax.jit(jax_fn)


def parse_terminal(expr: str, constants: dict[str, float] | None = None) -> Callable:
    x = sp.IndexedBase("x")
    local_dict = {"x": x}

    if constants is not None:
        constants_name = list(constants.keys())
        constants_symbols = sp.symbols(" ".join(constants_name) + ",")
        local_dict.update(
            {name: sym for name, sym in zip(constants_name, constants_symbols)}
        )

    sp_expr = sp.parse_expr(expr, local_dict=local_dict)

    if constants is not None:
        # replace constants by their values
        sp_expr = sp_expr.subs(
            {sym: val for sym, val in zip(constants_symbols, constants.values())}
        )

    jax_fn = sp.lambdify((x,), sp_expr, modules="jax")

    return jax.jit(jax_fn)
