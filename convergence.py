"""Empirical order-of-accuracy measurement.

Claiming a method is "fourth order" is checkable: halve the step size and
the error should fall by a factor of sixteen. This module runs that
experiment and fits the slope, with one guard that matters in practice --
once the discretisation error drops near machine precision the slope flattens
out, so points in the round-off floor are excluded from the fit instead of
dragging the estimate down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from physim.problems.base import ODEProblem
from physim.solvers import get_solver
from physim.validation.metrics import final_relative_error, max_relative_error


@dataclass
class ConvergenceStudy:
    method: str
    problem: str
    theoretical_order: int
    step_sizes: list[float]
    errors: list[float]
    rhs_evals: list[int]
    observed_order: float
    pairwise_orders: list[float] = field(default_factory=list)
    n_fitted: int = 0
    roundoff_floor: float = 1e-13

    def order_error_pct(self) -> float:
        """Deviation of the measured slope from the theoretical one."""
        if self.theoretical_order == 0:
            return float("nan")
        return 100.0 * abs(self.observed_order - self.theoretical_order) \
            / self.theoretical_order

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "problem": self.problem,
            "theoretical_order": self.theoretical_order,
            "observed_order": self.observed_order,
            "order_error_pct": self.order_error_pct(),
            "step_sizes": self.step_sizes,
            "errors": self.errors,
            "rhs_evals": self.rhs_evals,
            "pairwise_orders": self.pairwise_orders,
            "n_fitted": self.n_fitted,
        }


def convergence_study(
    solver_name: str,
    problem: ODEProblem,
    step_counts: Sequence[int] = (50, 100, 200, 400, 800),
    t_span: tuple[float, float] | None = None,
    metric: str = "max",
    roundoff_floor: float = 1e-13,
    pre_asymptotic_ceiling: float = 0.1,
    solver_kwargs: dict | None = None,
) -> ConvergenceStudy:
    """Run ``solver_name`` at several fixed step sizes and fit log(err) vs log(h).

    Parameters
    ----------
    metric
        ``"max"`` (default) scores the worst amplitude-normalised error
        anywhere on the trajectory; ``"final"`` scores the error at ``tf``.
        ``"max"`` is the safer default: on a decaying solution the endpoint
        value tends to zero, so a *relative* endpoint error keeps growing
        even as the method converges, and the fitted slope is biased.
    roundoff_floor
        Errors at or below this level carry no information about the
        discretisation order and are dropped before fitting.
    pre_asymptotic_ceiling
        Errors above this level usually come from step sizes outside the
        asymptotic regime (or outside the stability region), where the
        leading error term does not dominate. Those points are dropped too.
    """
    if not problem.has_exact:
        raise ValueError(
            f"convergence study needs a closed-form reference; "
            f"'{problem.name}' has none"
        )
    span = t_span or problem.t_span
    solver = get_solver(solver_name, **(solver_kwargs or {}))
    if solver.adaptive:
        raise ValueError(
            f"'{solver_name}' is adaptive; order is measured with fixed steps"
        )

    hs: list[float] = []
    errs: list[float] = []
    rhs: list[int] = []

    for n in step_counts:
        h = (span[1] - span[0]) / n
        res = solver.solve(problem, t_span=span, h=h)
        exact = problem.exact_at(res.t)
        if metric == "final":
            err = final_relative_error(res.y, exact)
        else:
            err = max_relative_error(res.y, exact)
        hs.append(float(h))
        errs.append(float(err))
        rhs.append(int(res.stats.n_rhs))

    h_arr = np.array(hs)
    e_arr = np.array(errs)
    usable = ((e_arr > roundoff_floor) & (e_arr < pre_asymptotic_ceiling)
              & np.isfinite(e_arr))

    if usable.sum() >= 2:
        slope = float(np.polyfit(np.log(h_arr[usable]), np.log(e_arr[usable]), 1)[0])
    else:
        slope = float("nan")

    pairwise = []
    for i in range(len(hs) - 1):
        if e_arr[i] > roundoff_floor and e_arr[i + 1] > roundoff_floor:
            pairwise.append(
                float(np.log(e_arr[i] / e_arr[i + 1])
                      / np.log(h_arr[i] / h_arr[i + 1]))
            )

    return ConvergenceStudy(
        method=solver_name, problem=problem.name,
        theoretical_order=solver.order, step_sizes=hs, errors=errs,
        rhs_evals=rhs, observed_order=slope, pairwise_orders=pairwise,
        n_fitted=int(usable.sum()), roundoff_floor=roundoff_floor,
    )


def work_precision(
    solver_names: Sequence[str],
    problem: ODEProblem,
    tolerances: Sequence[float] = (1e-3, 1e-5, 1e-7, 1e-9, 1e-11),
    step_counts: Sequence[int] = (25, 50, 100, 200, 400, 800, 1600),
) -> dict[str, list[dict]]:
    """Work-precision data: achieved error against right-hand-side evaluations.

    This is the comparison that decides which method to actually use. Order
    alone does not: a fifth-order method that needs a step size 100x smaller
    for stability is slower than a second-order method that does not.
    """
    out: dict[str, list[dict]] = {}
    for name in solver_names:
        solver_proto = get_solver(name)
        points: list[dict] = []
        if solver_proto.adaptive:
            for tol in tolerances:
                solver = get_solver(name, rtol=tol, atol=tol * 1e-3)
                res = solver.solve(problem)
                if not res.success:
                    continue
                exact = problem.exact_at(res.t)
                points.append({
                    "control": f"rtol={tol:g}",
                    "error": final_relative_error(res.y, exact),
                    "n_rhs": res.stats.n_rhs,
                    "n_steps": res.stats.n_steps,
                    "n_rejected": res.stats.n_rejected,
                    "elapsed_s": res.stats.elapsed_s,
                })
        else:
            solver = get_solver(name)
            for n in step_counts:
                res = solver.solve(problem, n_steps=n)
                if not res.success:
                    continue
                exact = problem.exact_at(res.t)
                points.append({
                    "control": f"n={n}",
                    "error": final_relative_error(res.y, exact),
                    "n_rhs": res.stats.n_rhs,
                    "n_steps": res.stats.n_steps,
                    "n_rejected": res.stats.n_rejected,
                    "elapsed_s": res.stats.elapsed_s,
                })
        out[name] = points
    return out
