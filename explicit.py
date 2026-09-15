"""Explicit Runge-Kutta methods.

Fast per step and trivial to implement, but their stability region is
bounded: on a stiff problem the step size is capped by ``|h*lambda_max|``
rather than by accuracy, which is exactly what the stiff benchmarks in this
repository are built to expose.
"""

from __future__ import annotations

import numpy as np

from physim.core.types import SolverStats
from physim.solvers.base import ODESolver


class ForwardEuler(ODESolver):
    """First-order explicit Euler. Stable only for ``|1 + h*lambda| <= 1``."""

    name = "forward_euler"
    order = 1
    stability = "explicit, stability limit |1+h*lambda| <= 1"

    def _step(self, problem, t, y, h, stats: SolverStats):
        f0 = np.asarray(problem.f(t, y), dtype=float)
        stats.n_rhs += 1
        return y + h * f0, None


class Heun(ODESolver):
    """Second-order explicit trapezoid (Heun's method)."""

    name = "heun"
    order = 2
    stability = "explicit, 2-stage"

    def _step(self, problem, t, y, h, stats: SolverStats):
        f0 = np.asarray(problem.f(t, y), dtype=float)
        f1 = np.asarray(problem.f(t + h, y + h * f0), dtype=float)
        stats.n_rhs += 2
        return y + 0.5 * h * (f0 + f1), None


class RK4(ODESolver):
    """Classical fourth-order Runge-Kutta.

    The workhorse for smooth non-stiff problems: four evaluations per step
    buy an ``O(h^4)`` global error, which the convergence harness verifies
    to within a few percent of the theoretical slope.
    """

    name = "rk4"
    order = 4
    stability = "explicit, |h*lambda| <~ 2.78 on the negative real axis"

    def _step(self, problem, t, y, h, stats: SolverStats):
        f = problem.f
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        stats.n_rhs += 4
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None


class RK45(ODESolver):
    """Dormand-Prince 5(4) with an embedded fourth-order error estimate.

    First-same-as-last: the last stage of an accepted step is the first stage
    of the next, so a step costs six evaluations rather than seven. The
    embedded pair is what makes the step size respond to the solution instead
    of to a guess.
    """

    name = "rk45"
    order = 5
    error_order = 4
    adaptive = True
    stability = "explicit, adaptive (Dormand-Prince 5(4))"

    C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
    A = [
        np.array([]),
        np.array([1 / 5]),
        np.array([3 / 40, 9 / 40]),
        np.array([44 / 45, -56 / 15, 32 / 9]),
        np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
        np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
        np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),
    ]
    B = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
    B_HAT = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640,
                      -92097 / 339200, 187 / 2100, 1 / 40])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fsal: tuple[float, np.ndarray] | None = None

    def solve(self, *args, **kwargs):
        self._fsal = None  # never reuse a stage across separate integrations
        return super().solve(*args, **kwargs)

    def _step(self, problem, t, y, h, stats: SolverStats):
        f = problem.f
        s = self.C.size
        k = np.empty((s, y.size))

        if self._fsal is not None and np.isclose(self._fsal[0], t, rtol=0, atol=0):
            k[0] = self._fsal[1]
        else:
            k[0] = np.asarray(f(t, y), dtype=float)
            stats.n_rhs += 1

        for i in range(1, s):
            dy = h * (self.A[i] @ k[:i])
            k[i] = np.asarray(f(t + self.C[i] * h, y + dy), dtype=float)
            stats.n_rhs += 1

        y_new = y + h * (self.B @ k)
        err = h * ((self.B - self.B_HAT) @ k)
        # Stage 7 is evaluated at (t+h, y_new): reuse it as the next k1.
        self._fsal = (t + h, k[-1])
        return y_new, err


class RK23(ODESolver):
    """Bogacki-Shampine 3(2): cheap adaptive method for loose tolerances."""

    name = "rk23"
    order = 3
    error_order = 2
    adaptive = True
    stability = "explicit, adaptive (Bogacki-Shampine 3(2))"

    def _step(self, problem, t, y, h, stats: SolverStats):
        f = problem.f
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(f(t + 0.75 * h, y + 0.75 * h * k2), dtype=float)
        y_new = y + h * (2 / 9 * k1 + 1 / 3 * k2 + 4 / 9 * k3)
        k4 = np.asarray(f(t + h, y_new), dtype=float)
        stats.n_rhs += 4
        y_hat = y + h * (7 / 24 * k1 + 1 / 4 * k2 + 1 / 3 * k3 + 1 / 8 * k4)
        return y_new, y_new - y_hat
