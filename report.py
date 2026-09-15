"""The full validation suite, as one reproducible pass.

``physim validate`` runs this and writes ``results/validation.json``. Every
number quoted in ``docs/VALIDATION.md`` and in the README comes from that
file, so the claims and the artefact cannot drift apart.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from physim.problems import ANALYTIC_PROBLEMS, get_problem
from physim.solvers import get_solver
from physim.validation.convergence import convergence_study
from physim.validation.metrics import conservation_drift, score

#: Accuracy target: worst-case relative error against the closed form.
DEFAULT_TOL_PCT = 0.01

#: Fixed-step counts chosen so every method reaches the target where it can.
_ACCURACY_GRID: dict[str, int] = {
    "rk4": 4000,
    "radau_iia5": 2000,
    "trapezoidal": 20000,
    "backward_euler": 400000,
}

_ADAPTIVE_TOL = {"rk45": 1e-10, "trbdf2": 1e-10, "rk23": 1e-10}


@dataclass
class ValidationReport:
    accuracy: list[dict] = field(default_factory=list)
    convergence: list[dict] = field(default_factory=list)
    stiff: list[dict] = field(default_factory=list)
    phy: list[dict] = field(default_factory=list)
    environment: dict = field(default_factory=dict)
    tolerance_pct: float = DEFAULT_TOL_PCT

    @property
    def accuracy_pass_rate(self) -> float:
        if not self.accuracy:
            return 0.0
        return sum(a["passed"] for a in self.accuracy) / len(self.accuracy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tolerance_pct": self.tolerance_pct,
            "environment": self.environment,
            "summary": {
                "accuracy_pass_rate": self.accuracy_pass_rate,
                "accuracy_cases": len(self.accuracy),
                "accuracy_passed": sum(a["passed"] for a in self.accuracy),
                "worst_max_rel_error_pct": max(
                    (a["max_rel_error_pct"] for a in self.accuracy), default=0.0
                ),
                "convergence_cases": len(self.convergence),
                "worst_order_error_pct": max(
                    (c["order_error_pct"] for c in self.convergence
                     if np.isfinite(c["order_error_pct"])), default=0.0
                ),
                "phy_cases": len(self.phy),
                "worst_phy_deviation_pct": max(
                    (p["worst_relative_deviation_pct"] for p in self.phy
                     if np.isfinite(p["worst_relative_deviation_pct"])),
                    default=0.0
                ),
            },
            "accuracy": self.accuracy,
            "convergence": self.convergence,
            "stiff": self.stiff,
            "phy": self.phy,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path


def _environment() -> dict:
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def run_accuracy(problems: Sequence[str] = ANALYTIC_PROBLEMS,
                 methods: Sequence[str] = ("rk4", "rk45", "radau_iia5",
                                           "trbdf2", "trapezoidal"),
                 tol_pct: float = DEFAULT_TOL_PCT) -> list[dict]:
    """Score every (problem, method) pair against its closed-form solution."""
    rows: list[dict] = []
    for pname in problems:
        problem = get_problem(pname)
        for mname in methods:
            proto = get_solver(mname)
            # An explicit method on a stiff problem is a stability result,
            # not an accuracy result: record it and move on.
            if problem.stiff and not proto.implicit and pname != "transmission_line":
                continue
            if proto.adaptive:
                tol = _ADAPTIVE_TOL.get(mname, 1e-10)
                solver = get_solver(mname, rtol=tol, atol=tol * 1e-3)
                result = solver.solve(problem)
                h = None
            else:
                n = _ACCURACY_GRID.get(mname, 2000)
                solver = get_solver(mname)
                result = solver.solve(problem, n_steps=n)
                h = (problem.t_span[1] - problem.t_span[0]) / n
            rep = score(result, problem, h=h)
            row = rep.as_dict()
            row["passed"] = bool(rep.passes(tol_pct))
            row["drift"] = conservation_drift(result, problem)
            row["stiff"] = problem.stiff
            rows.append(row)
    return rows


def run_convergence(cases: Sequence[tuple[str, str]] | None = None
                    ) -> list[dict]:
    """Measure the empirical order of every fixed-step method."""
    cases = cases or [
        ("forward_euler", "exponential_decay"),
        ("heun", "exponential_decay"),
        ("rk4", "exponential_decay"),
        ("rk4", "damped_oscillator"),
        ("backward_euler", "exponential_decay"),
        ("trapezoidal", "damped_oscillator"),
        ("radau_iia5", "damped_oscillator"),
        ("radau_iia5", "rlc_step"),
    ]
    rows = []
    for method, pname in cases:
        study = convergence_study(method, get_problem(pname))
        rows.append(study.as_dict())
    return rows


def run_stiff() -> list[dict]:
    """Explicit versus implicit on a genuinely stiff problem.

    The point is not that the implicit method is more accurate -- it is that
    the explicit one cannot take the step at all. Both outcomes are recorded.
    """
    rows = []
    problem = get_problem("prothero_robinson")
    lam = abs(problem.params["lambda"])
    for method in ("rk4", "backward_euler", "trapezoidal", "trbdf2",
                   "radau_iia5"):
        proto = get_solver(method)
        if proto.adaptive:
            solver = get_solver(method, rtol=1e-8, atol=1e-10)
            result = solver.solve(problem)
            h_used = float(np.median(np.diff(result.t))) if result.t.size > 2 else None
        else:
            solver = get_solver(method)
            result = solver.solve(problem, n_steps=400)
            h_used = (problem.t_span[1] - problem.t_span[0]) / 400
        row: dict[str, Any] = {
            "method": method,
            "problem": problem.name,
            "stiffness": lam,
            "success": result.success,
            "message": result.message,
            "n_rhs": result.stats.n_rhs,
            "n_lu": result.stats.n_lu,
            "elapsed_s": result.stats.elapsed_s,
            "h_used": h_used,
            "h_lambda": (h_used * lam) if h_used else None,
        }
        if result.success:
            rep = score(result, problem)
            row["max_rel_error_pct"] = rep.max_rel_error_pct
        else:
            row["max_rel_error_pct"] = float("inf")
        rows.append(row)

    # Robertson has no closed form; check the conservation invariant instead.
    rob = get_problem("robertson")
    res = get_solver("trbdf2", rtol=1e-8, atol=1e-12).solve(rob)
    rows.append({
        "method": "trbdf2", "problem": "robertson",
        "stiffness": rob.stiffness_ratio, "success": res.success,
        "message": res.message, "n_rhs": res.stats.n_rhs,
        "n_lu": res.stats.n_lu, "elapsed_s": res.stats.elapsed_s,
        "conservation_drift": conservation_drift(res, rob),
        "max_rel_error_pct": None, "h_used": None, "h_lambda": None,
    })
    return rows


def run_phy(schemes: Sequence[str] = ("bpsk", "qpsk", "qam16", "qam64"),
            channel: str = "awgn", target_errors: int = 400,
            max_symbols: int = 400_000, seed: int = 20241007) -> list[dict]:
    """Monte Carlo BER against the closed forms, per scheme."""
    from physim.phy.link import ber_sweep

    grids = {
        "bpsk": np.arange(0, 11, 2.0),
        "qpsk": np.arange(0, 11, 2.0),
        "psk8": np.arange(4, 17, 2.0),
        "qam16": np.arange(4, 17, 2.0),
        "qam64": np.arange(8, 21, 2.0),
        "qam256": np.arange(14, 27, 2.0),
    }
    rows = []
    for scheme in schemes:
        sweep = ber_sweep(scheme, grids.get(scheme, np.arange(0, 13, 2.0)),
                          channel=channel, seed=seed,
                          target_errors=target_errors, max_symbols=max_symbols)
        worst = sweep.worst_relative_deviation()
        rows.append({
            "scheme": scheme,
            "channel": channel,
            "seed": seed,
            "worst_relative_deviation_pct": 100.0 * worst,
            "implementation_loss_db": sweep.implementation_loss_db(),
            "points_with_theory_in_ci": sum(p.theory_within_ci
                                            for p in sweep.points),
            "n_points": len(sweep.points),
            "total_bits": sum(p.n_bits for p in sweep.points),
            "points": [p.as_dict() for p in sweep.points],
        })
    return rows


def run_all(quick: bool = False) -> ValidationReport:
    """Run every validation stage and return the assembled report."""
    report = ValidationReport(environment=_environment())
    problems = ANALYTIC_PROBLEMS
    methods = ("rk4", "rk45", "radau_iia5", "trbdf2")
    if quick:
        problems = ("exponential_decay", "damped_oscillator", "rlc_step",
                    "prothero_robinson")
        methods = ("rk4", "rk45", "radau_iia5")

    report.accuracy = run_accuracy(problems, methods)
    report.convergence = run_convergence()
    report.stiff = run_stiff()
    report.phy = run_phy(
        schemes=("bpsk", "qpsk", "qam16") if quick
        else ("bpsk", "qpsk", "qam16", "qam64"),
        target_errors=200 if quick else 400,
        max_symbols=200_000 if quick else 600_000,
    )
    return report
