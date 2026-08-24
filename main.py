import argparse
import sys
import time
from pathlib import Path

import jax

from bsde_tt_jax.config.loader import load_config
from bsde_tt_jax.io.results import save_result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bsde-solve",
        description="Solve a BSDE specified in a YAML file via TT-chaos Picard iteration.",
    )
    p.add_argument("config", type=Path, help="Path to the YAML configuration file.")
    p.add_argument("bsde", type=str, help="Name of the BSDE.")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory to write per-BSDE results into (default: ./results).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base PRNG seed.",
    )
    p.add_argument(
        "--x64",
        action="store_true",
        help="Enable float64 in JAX (recommended for TT-ALS numerical stability).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.x64:
        jax.config.update("jax_enable_x64", True)

    if not args.config.exists():
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        problems = load_config(args.config)
    except Exception as exc:  # config errors should be readable, not a stack trace
        print(f"error: failed to load {args.config}: {exc}", file=sys.stderr)
        return 1

    problem = next(p for p in problems if p.name == args.bsde)
    if not problem:
        print(
            f"error: the requested BSDE {args.bsde} found in {args.config}",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded BSDE from {args.config}.")
    t0 = time.time()
    result = run(problem, seed=args.seed)
    elapsed = time.time() - t0
    print(f"Solved BSDE in {elapsed:.1f}s.")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / args.bsde).mkdir(parents=True, exist_ok=True)
    save_result(result, args.output / args.bsde)

    converged = result.converged
    return not converged


if __name__ == "__main__":
    raise SystemExit(main())
