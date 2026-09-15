"""Command line entry point: ``physim <command>``.

Everything the HTTP API exposes is reachable here too, so the engine is
usable from a shell script or a CI job without running a server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from physim import __version__


# ----------------------------------------------------------------- utilities
def _name_list(values) -> list[str]:
    """Accept ``--solvers rk4 heun`` and ``--solvers rk4,heun`` alike."""
    out: list[str] = []
    for value in values:
        out.extend(part for part in str(value).replace(",", " ").split() if part)
    return out


def _int_list(values) -> list[int]:
    return [int(v) for v in _name_list(values)]


def _parse_range(spec: str) -> np.ndarray:
    """Parse ``start:stop:step`` or a comma-separated list."""
    if ":" in spec:
        parts = [float(p) for p in spec.split(":")]
        if len(parts) == 2:
            start, stop, step = parts[0], parts[1], 1.0
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise argparse.ArgumentTypeError(f"bad range '{spec}'")
        return np.arange(start, stop + 0.5 * step, step)
    return np.array([float(p) for p in spec.split(",")])


def _table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """Render a fixed-width table. ``columns`` is (key, header, format)."""
    headers = [c[1] for c in columns]
    body = []
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            value = row.get(key)
            if value is None:
                cells.append("-")
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, str):
                cells.append(value)
            else:
                try:
                    cells.append(format(value, fmt))
                except (TypeError, ValueError):
                    cells.append(str(value))
        body.append(cells)
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    rule = "  ".join("-" * w for w in widths)
    out = [line, rule]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=False))
            for r in body]
    return "\n".join(out)


# ------------------------------------------------------------------ commands
def cmd_list(args) -> int:
    from physim.phy.modulation import list_modulations
    from physim.problems import list_problems
    from physim.solvers import list_solvers

    print("Solvers")
    print(_table(list_solvers(), [
        ("name", "name", ""), ("order", "order", "d"),
        ("adaptive", "adaptive", ""), ("implicit", "implicit", ""),
        ("stability", "stability", ""),
    ]))
    print("\nProblems")
    print(_table(list_problems(), [
        ("name", "name", ""), ("dim", "dim", "d"),
        ("stiff", "stiff", ""), ("has_exact", "closed form", ""),
        ("stiffness_ratio", "stiffness", ".3g"),
        ("description", "description", ""),
    ]))
    print("\nModulation")
    print(_table(list_modulations(), [
        ("name", "name", ""), ("M", "M", "d"),
        ("bits_per_symbol", "bits/sym", "d"),
        ("min_distance", "d_min", ".4f"),
    ]))
    return 0


def cmd_solve(args) -> int:
    from physim.problems import get_problem
    from physim.solvers import get_solver
    from physim.validation.metrics import conservation_drift, score

    problem = get_problem(args.problem)
    solver = get_solver(args.solver, rtol=args.rtol, atol=args.atol)
    kwargs = {}
    if not solver.adaptive:
        kwargs["n_steps"] = args.steps
    result = solver.solve(problem, **kwargs)

    print(f"problem   {problem.name}  ({problem.description})")
    print(f"method    {solver.name}  order {solver.order}  "
          f"{'adaptive' if solver.adaptive else 'fixed step'}  "
          f"{solver.stability}")
    print(f"interval  [{problem.t_span[0]:g}, {problem.t_span[1]:g}]  "
          f"dimension {problem.dim}")
    print(f"status    {'ok' if result.success else 'FAILED'}: {result.message}")
    s = result.stats
    print(f"work      {s.n_steps} steps ({s.n_rejected} rejected), "
          f"{s.n_rhs} rhs evals, {s.n_jac} Jacobians, {s.n_lu} LU, "
          f"{s.elapsed_s * 1e3:.1f} ms")

    if problem.has_exact and result.success:
        rep = score(result, problem)
        print(f"accuracy  max relative error {rep.max_rel_error_pct:.3e} % "
              f"(target {args.tol_pct:g} %)  ->  "
              f"{'PASS' if rep.passes(args.tol_pct) else 'FAIL'}")
        print(f"          rms relative error {rep.rms_rel_error:.3e}, "
              f"max absolute error {rep.max_abs_error:.3e}")
    drift = conservation_drift(result, problem)
    if drift is not None:
        print(f"invariant relative drift {drift:.3e}")

    print("\nfinal state")
    for label, value in zip(problem.labels, result.y[-1], strict=False):
        print(f"  {label:<10s} {value: .12g}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result.as_dict(max_points=2000),
                                             indent=2))
        print(f"\nwrote {args.out}")
    return 0 if result.success else 1


def cmd_converge(args) -> int:
    from physim.problems import get_problem
    from physim.validation import convergence_study

    problem = get_problem(args.problem)
    counts = tuple(_int_list(args.counts))
    failures = 0
    for name in _name_list(args.solvers):
        study = convergence_study(name, problem, step_counts=counts,
                                  metric=args.metric)
        print(f"{study.method} on {study.problem}")
        rows = [{"h": h, "error": e, "rhs": r}
                for h, e, r in zip(study.step_sizes, study.errors,
                                   study.rhs_evals, strict=False)]
        for i, row in enumerate(rows):
            row["order"] = (study.pairwise_orders[i - 1]
                            if 0 < i <= len(study.pairwise_orders) else None)
        print(_table(rows, [("h", "step size", ".4e"),
                            ("error", "rel error", ".4e"),
                            ("order", "local slope", ".3f"),
                            ("rhs", "rhs evals", "d")]))
        if study.n_fitted == 0:
            print("\nno asymptotic range at these step sizes: "
                  "order not measurable\n")
            failures += 1
            continue
        print(f"\ntheoretical order {study.theoretical_order}, "
              f"observed {study.observed_order:.3f} "
              f"(fitted on {study.n_fitted} points, "
              f"{study.order_error_pct():.1f} % deviation)\n")
    return 1 if failures else 0


def cmd_ber(args) -> int:
    from physim.phy.link import ber_sweep

    ebn0 = _parse_range(args.ebn0)
    sweep = ber_sweep(args.scheme, ebn0, channel=args.channel, seed=args.seed,
                      target_errors=args.target_errors,
                      max_symbols=args.max_symbols)
    print(f"{args.scheme} over {args.channel}, seed {args.seed}")
    print(_table([p.as_dict() for p in sweep.points], [
        ("ebn0_db", "Eb/N0 dB", ".1f"),
        ("ber", "simulated", ".3e"),
        ("theory_ber", "theory", ".3e"),
        ("rel_deviation", "deviation", "+.2%"),
        ("n_bit_errors", "errors", "d"),
        ("n_bits", "bits", "d"),
        ("theory_within_ci", "in 95% CI", ""),
    ]))
    worst = sweep.worst_relative_deviation()
    loss = sweep.implementation_loss_db()
    print(f"\nworst deviation from theory {100 * worst:.2f} % "
          f"(points with >= 50 errors)")
    if np.isfinite(loss):
        print(f"implementation loss {loss:+.3f} dB")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(sweep.as_dict(), indent=2))
        print(f"wrote {args.out}")
    return 0


def cmd_link(args) -> int:
    from physim.phy.channel import (
        breakpoint_distance_m,
        link_budget,
    )
    from physim.phy.modulation import get_modulation
    from physim.phy.theory import required_ebn0_db

    mod = get_modulation(args.scheme)
    required = required_ebn0_db(args.target_ber, mod.M, mod.family)
    required_snr = required + 10 * np.log10(mod.bits_per_symbol)
    budget = link_budget(args.distance, args.freq, tx_power_dbm=args.tx_power,
                         bandwidth_hz=args.bandwidth,
                         noise_figure_db=args.noise_figure,
                         required_snr_db=required_snr, model=args.model,
                         exponent=args.exponent)
    print(f"link at {args.distance:g} m, {args.freq / 1e9:g} GHz, "
          f"{args.model} model")
    print(f"  path loss          {budget.path_loss_db:8.2f} dB")
    print(f"  received power     {budget.rx_power_dbm:8.2f} dBm")
    print(f"  noise floor        {budget.noise_power_dbm:8.2f} dBm "
          f"({args.bandwidth / 1e6:g} MHz, NF {args.noise_figure:g} dB)")
    print(f"  SNR                {budget.snr_db:8.2f} dB")
    print(f"  {args.scheme} needs   {required_snr:8.2f} dB "
          f"for BER {args.target_ber:g}  (Eb/N0 {required:.2f} dB)")
    print(f"  margin             {budget.margin_db:+8.2f} dB  "
          f"{'link closes' if budget.margin_db >= 0 else 'link fails'}")
    if args.model == "two_ray":
        print(f"  breakpoint at      "
              f"{breakpoint_distance_m(args.freq, 10.0, 1.5):8.1f} m")
    if args.fail_on_negative_margin and budget.margin_db < 0:
        return 2
    return 0


def cmd_validate(args) -> int:
    from physim.validation.report import run_all

    report = run_all(quick=args.quick)
    data = report.as_dict()

    print("Accuracy against closed-form solutions "
          f"(target: max relative error <= {report.tolerance_pct:g} %)")
    print(_table(report.accuracy, [
        ("problem", "problem", ""), ("method", "method", ""),
        ("max_rel_error_pct", "max rel err %", ".3e"),
        ("final_rel_error", "final rel err", ".3e"),
        ("n_rhs", "rhs", "d"), ("passed", "pass", ""),
    ]))

    print("\nObserved order of accuracy")
    print(_table(report.convergence, [
        ("method", "method", ""), ("problem", "problem", ""),
        ("theoretical_order", "theory", "d"),
        ("observed_order", "measured", ".3f"),
        ("order_error_pct", "deviation %", ".1f"),
    ]))

    print("\nStiff problem: explicit vs implicit")
    print(_table(report.stiff, [
        ("method", "method", ""), ("problem", "problem", ""),
        ("success", "completed", ""),
        ("h_lambda", "h*|lambda|", ".3g"),
        ("max_rel_error_pct", "max rel err %", ".3e"),
        ("n_rhs", "rhs", "d"), ("n_lu", "LU", "d"),
    ]))

    print("\nMonte Carlo BER vs closed form")
    print(_table(report.phy, [
        ("scheme", "scheme", ""), ("channel", "channel", ""),
        ("worst_relative_deviation_pct", "worst dev %", ".2f"),
        ("implementation_loss_db", "impl loss dB", "+.3f"),
        ("points_with_theory_in_ci", "in CI", "d"),
        ("n_points", "points", "d"),
        ("total_bits", "bits", "d"),
    ]))

    summary = data["summary"]
    print(f"\n{summary['accuracy_passed']}/{summary['accuracy_cases']} "
          f"accuracy cases within {report.tolerance_pct:g} %, "
          f"worst {summary['worst_max_rel_error_pct']:.3e} %")

    path = report.write(args.out)
    print(f"wrote {path}")
    ok = summary["accuracy_passed"] == summary["accuracy_cases"]
    return 0 if ok else 1


def cmd_bench(args) -> int:
    from physim.problems import get_problem
    from physim.validation.convergence import work_precision

    problem = get_problem(args.problem)
    data = work_precision(args.solvers, problem)
    rows = []
    for method, points in data.items():
        for p in points:
            rows.append({"method": method, **p})
    print(f"work-precision on {problem.name}")
    print(_table(rows, [
        ("method", "method", ""), ("control", "control", ""),
        ("error", "rel error", ".3e"), ("n_rhs", "rhs evals", "d"),
        ("n_steps", "steps", "d"), ("n_rejected", "rejected", "d"),
        ("elapsed_s", "seconds", ".4f"),
    ]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"wrote {args.out}")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed: pip install -e '.[api]'",
              file=sys.stderr)
        return 1
    print(f"serving on http://{args.host}:{args.port}")
    uvicorn.run("api.main:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="physim",
        description="Numerical analysis and physical-layer simulation engine",
    )
    p.add_argument("--version", action="version", version=f"physim {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show available solvers, problems and schemes"
                   ).set_defaults(func=cmd_list)

    s = sub.add_parser("solve", help="integrate one problem with one method")
    s.add_argument("--problem", default="damped_oscillator")
    s.add_argument("--solver", default="rk45")
    s.add_argument("--steps", type=int, default=2000,
                   help="fixed-step methods only")
    s.add_argument("--rtol", type=float, default=1e-8)
    s.add_argument("--atol", type=float, default=1e-11)
    s.add_argument("--tol-pct", type=float, default=0.01,
                   help="accuracy target, percent")
    s.add_argument("--out", help="write the trajectory to a JSON file")
    s.set_defaults(func=cmd_solve)

    c = sub.add_parser("converge", help="measure the empirical order of accuracy")
    c.add_argument("--solvers", "--solver", dest="solvers", nargs="+",
                   default=["rk4"],
                   help="one or more fixed-step methods, space or comma separated")
    c.add_argument("--problem", default="exponential_decay")
    c.add_argument("--counts", nargs="+",
                   default=[50, 100, 200, 400, 800])
    c.add_argument("--metric", choices=["max", "final"], default="max")
    c.set_defaults(func=cmd_converge)

    b = sub.add_parser("ber", help="Monte Carlo BER sweep against theory")
    b.add_argument("--scheme", default="qpsk")
    b.add_argument("--channel", default="awgn",
                   choices=["awgn", "rayleigh", "rician"])
    b.add_argument("--ebn0", default="0:12:2", help="start:stop:step or a list")
    b.add_argument("--target-errors", type=int, default=200)
    b.add_argument("--max-symbols", type=int, default=500_000)
    b.add_argument("--seed", type=int, default=12345)
    b.add_argument("--out")
    b.set_defaults(func=cmd_ber)

    l = sub.add_parser("link", help="link budget and required SNR")
    l.add_argument("--distance", type=float, default=1000.0)
    l.add_argument("--freq", type=float, default=2.4e9)
    l.add_argument("--scheme", default="qam16")
    l.add_argument("--target-ber", type=float, default=1e-6)
    l.add_argument("--tx-power", type=float, default=20.0)
    l.add_argument("--bandwidth", type=float, default=20e6)
    l.add_argument("--noise-figure", type=float, default=6.0)
    l.add_argument("--model", default="log_distance",
                   choices=["free_space", "log_distance", "two_ray"])
    l.add_argument("--exponent", type=float, default=3.0)
    l.add_argument("--fail-on-negative-margin", action="store_true",
                   help="exit 2 when the link does not close, for use in scripts")
    l.set_defaults(func=cmd_link)

    v = sub.add_parser("validate", help="run the full validation suite")
    v.add_argument("--quick", action="store_true")
    v.add_argument("--out", default="results/validation.json")
    v.set_defaults(func=cmd_validate)

    w = sub.add_parser("bench", help="work-precision comparison")
    w.add_argument("--problem", default="damped_oscillator")
    w.add_argument("--solvers", nargs="+",
                   default=["rk4", "rk45", "trbdf2", "radau_iia5"])
    w.add_argument("--out")
    w.set_defaults(func=cmd_bench)

    srv = sub.add_parser("serve", help="run the HTTP API and dashboard")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--reload", action="store_true")
    srv.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
