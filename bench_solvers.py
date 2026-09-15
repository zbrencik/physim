"""Timing and work comparison across solvers.

Accuracy alone ranks methods badly: any method reaches any accuracy given
enough steps. What matters is the accuracy bought per unit of work, so every
figure here is reported against right-hand-side evaluations and wall clock.

Run with ``python benchmarks/bench_solvers.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from physim.problems import get_problem, list_problems
from physim.solvers import ADAPTIVE, FIXED_STEP, get_solver
from physim.validation.metrics import max_relative_error


def time_run(fn, repeats: int = 3) -> tuple[object, float]:
    """Best of N: the minimum is the least contaminated by other load."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
    return result, best


def bench_problem(problem_name: str, repeats: int) -> list[dict]:
    problem = get_problem(problem_name)
    rows: list[dict] = []

    for name in sorted(FIXED_STEP):
        for n_steps in (200, 2000):
            solver = get_solver(name)
            result, elapsed = time_run(
                lambda s=solver, n=n_steps: s.solve(problem, n_steps=n),
                repeats)
            rows.append(_row(problem, result, name, f"n={n_steps}", elapsed))

    for name in sorted(ADAPTIVE):
        for rtol in (1e-6, 1e-10):
            solver = get_solver(name, rtol=rtol, atol=rtol * 1e-3)
            result, elapsed = time_run(lambda s=solver: s.solve(problem),
                                       repeats)
            rows.append(_row(problem, result, name, f"rtol={rtol:g}", elapsed))

    return rows


def _row(problem, result, method, control, elapsed) -> dict:
    error = None
    if result.success and problem.has_exact:
        error = max_relative_error(result.y, problem.exact_at(result.t))
    return {
        "problem": problem.name,
        "method": method,
        "control": control,
        "success": bool(result.success),
        "error": error,
        "n_steps": result.stats.n_steps,
        "n_rejected": result.stats.n_rejected,
        "n_rhs": result.stats.n_rhs,
        "n_lu": result.stats.n_lu,
        "elapsed_s": elapsed,
        "rhs_per_second": result.stats.n_rhs / elapsed if elapsed else None,
    }


def print_table(rows: list[dict]) -> None:
    header = (f"{'method':16s} {'control':12s} {'error':>11s} {'steps':>7s} "
              f"{'rhs':>8s} {'LU':>6s} {'ms':>8s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        err = "diverged" if not r["success"] else (
            "-" if r["error"] is None else f"{r['error']:.3e}")
        print(f"{r['method']:16s} {r['control']:12s} {err:>11s} "
              f"{r['n_steps']:7d} {r['n_rhs']:8d} {r['n_lu']:6d} "
              f"{1e3 * r['elapsed_s']:8.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+",
                        default=["damped_oscillator", "rlc_step",
                                 "stiff_linear", "transmission_line"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="results/benchmarks.json")
    args = parser.parse_args(argv)

    known = {p["name"] for p in list_problems()}
    unknown = set(args.problems) - known
    if unknown:
        parser.error(f"unknown problems: {sorted(unknown)}")

    payload = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repeats": args.repeats,
        "runs": [],
    }

    for name in args.problems:
        print(f"\n{name}")
        print("=" * len(name))
        rows = bench_problem(name, args.repeats)
        print_table(rows)
        payload["runs"].extend(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
