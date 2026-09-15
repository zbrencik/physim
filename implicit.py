"""Implicit Runge-Kutta methods for stiff systems.

Every method here is at least A-stable, so the step size is limited by the
accuracy you ask for and not by the fastest eigenvalue in the problem. The
price is a Newton solve per stage; the work counters in
:class:`~physim.core.types.SolverStats` record exactly what that costs.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from physim.core.linalg import finite_difference_jacobian, newton_solve
from physim.core.types import SolverStats
from physim.solvers.base import ODESolver


class ImplicitSolver(ODESolver):
    """Shared Jacobian handling and stage solving for implicit methods."""

    implicit = True

    def __init__(self, *args, newton_max_iter: int = 12, **kwargs):
        super().__init__(*args, **kwargs)
        self.newton_max_iter = int(newton_max_iter)

    def _jacobian(self, problem, t, y, stats: SolverStats) -> np.ndarray:
        if problem.jac is not None:
            jac = np.atleast_2d(np.asarray(problem.jac(t, y), dtype=float))
        else:
            jac, n_extra = finite_difference_jacobian(problem.f, t, y)
            stats.n_rhs += n_extra + 1
        stats.n_jac += 1
        return jac

    def _newton_tol(self, y: np.ndarray) -> float:
        return float(max(self.atol, self.rtol * np.linalg.norm(y), 1e-14))

    def _solve_stage(self, problem, t_stage, y_base, h_aii, guess, lu,
                     stats: SolverStats, tol: float):
        """Solve ``Y = y_base + h*a_ii*f(t_stage, Y)`` by simplified Newton."""

        def residual(Y):
            return Y - y_base - h_aii * np.asarray(
                problem.f(t_stage, Y), dtype=float
            )

        Y, report = newton_solve(residual, lu, guess, tol,
                                 max_iter=self.newton_max_iter)
        stats.n_rhs += report.rhs_evals
        stats.n_newton += report.iterations
        if not report.converged:
            stats.n_newton_failed += 1
        return Y, report.converged


class BackwardEuler(ImplicitSolver):
    """First-order, L-stable. The reference point for stiff stability.

    Damps every fast mode unconditionally, which is why it survives step
    sizes thousands of times larger than the explicit stability limit, at
    the cost of only first-order accuracy.
    """

    name = "backward_euler"
    order = 1
    stability = "L-stable"

    def _step(self, problem, t, y, h, stats: SolverStats):
        jac = self._jacobian(problem, t, y, stats)
        lu = lu_factor(np.eye(y.size) - h * jac)
        stats.n_lu += 1
        f0 = np.asarray(problem.f(t, y), dtype=float)
        stats.n_rhs += 1
        guess = y + h * f0  # explicit Euler predictor
        Y, _ = self._solve_stage(problem, t + h, y, h, guess, lu, stats,
                                 self._newton_tol(y))
        return Y, None


class Trapezoidal(ImplicitSolver):
    """Second-order, A-stable (the Crank-Nicolson time discretisation).

    A-stable but not L-stable: the amplification factor tends to ``-1`` for
    very stiff modes, so fast transients ring rather than die. The
    ``tests/test_stiff.py::test_trapezoid_rings`` case pins that behaviour
    down, and TR-BDF2 below is the standard fix.
    """

    name = "trapezoidal"
    order = 2
    stability = "A-stable (not L-stable)"

    def _step(self, problem, t, y, h, stats: SolverStats):
        jac = self._jacobian(problem, t, y, stats)
        lu = lu_factor(np.eye(y.size) - 0.5 * h * jac)
        stats.n_lu += 1
        f0 = np.asarray(problem.f(t, y), dtype=float)
        stats.n_rhs += 1
        y_base = y + 0.5 * h * f0
        Y, _ = self._solve_stage(problem, t + h, y_base, 0.5 * h, y + h * f0,
                                 lu, stats, self._newton_tol(y))
        return Y, None


class TRBDF2(ImplicitSolver):
    """TR-BDF2: one trapezoidal stage followed by one BDF2 stage.

    Written as a 3-stage DIRK with ``gamma = 2 - sqrt(2)``, which makes the
    two stages share a single iteration matrix ``I - h*d*J`` and therefore a
    single LU factorisation per step. Second order, L-stable, with an
    embedded estimator that drives adaptive step sizing.
    """

    name = "trbdf2"
    order = 2
    # Both the main and the embedded weights are second order, so the
    # difference between them estimates the O(h^3) local error term and the
    # controller exponent is 1/(2+1).
    error_order = 2
    adaptive = True
    stability = "L-stable, adaptive"

    GAMMA = 2.0 - np.sqrt(2.0)
    D = GAMMA / 2.0              # = 1 - sqrt(2)/2, the repeated diagonal entry
    W = np.sqrt(2.0) / 4.0

    def _step(self, problem, t, y, h, stats: SolverStats):
        d, w, gamma = self.D, self.W, self.GAMMA
        n = y.size
        tol = self._newton_tol(y)

        jac = self._jacobian(problem, t, y, stats)
        lu = lu_factor(np.eye(n) - h * d * jac)
        stats.n_lu += 1

        f1 = np.asarray(problem.f(t, y), dtype=float)
        stats.n_rhs += 1

        # Stage 2: trapezoidal rule over [t, t + gamma*h]
        base2 = y + h * d * f1
        Y2, ok2 = self._solve_stage(problem, t + gamma * h, base2, h * d,
                                    y + h * gamma * f1, lu, stats, tol)
        f2 = np.asarray(problem.f(t + gamma * h, Y2), dtype=float)
        stats.n_rhs += 1

        # Stage 3: BDF2 over [t, t + h] using y and Y2
        base3 = y + h * (w * f1 + w * f2)
        Y3, ok3 = self._solve_stage(problem, t + h, base3, h * d, Y2, lu,
                                    stats, tol)
        f3 = np.asarray(problem.f(t + h, Y3), dtype=float)
        stats.n_rhs += 1

        if not (ok2 and ok3):
            # Force a rejection: the controller will retry with a smaller h.
            return Y3, np.full(n, np.inf)

        b_hat = np.array([(1.0 - w) / 3.0, (3.0 * w + 1.0) / 3.0, d / 3.0])
        y_hat = y + h * (b_hat[0] * f1 + b_hat[1] * f2 + b_hat[2] * f3)
        return Y3, Y3 - y_hat


class RadauIIA5(ImplicitSolver):
    """Three-stage Radau IIA, order 5, L-stable and stiffly accurate.

    All three stages are coupled, so one step solves a ``3d x 3d`` Newton
    system with iteration matrix ``I - h*(A kron J)``. That is the most
    expensive step in the engine and also the most accurate per unit of step
    size, which is why the stiff reference trajectories are generated with it.

    The tableau is collocation at the Radau right points, so it satisfies the
    simplifying conditions ``C(3)`` and ``B(5)``; ``tests/test_solvers.py``
    verifies those numerically rather than trusting the transcription.
    """

    name = "radau_iia5"
    order = 5
    stability = "L-stable, stiffly accurate"

    _S6 = np.sqrt(6.0)
    C = np.array([(4.0 - _S6) / 10.0, (4.0 + _S6) / 10.0, 1.0])
    A = np.array([
        [(88.0 - 7.0 * _S6) / 360.0,
         (296.0 - 169.0 * _S6) / 1800.0,
         (-2.0 + 3.0 * _S6) / 225.0],
        [(296.0 + 169.0 * _S6) / 1800.0,
         (88.0 + 7.0 * _S6) / 360.0,
         (-2.0 - 3.0 * _S6) / 225.0],
        [(16.0 - _S6) / 36.0,
         (16.0 + _S6) / 36.0,
         1.0 / 9.0],
    ])
    B = A[-1].copy()  # stiffly accurate: the last stage is the solution

    def _step(self, problem, t, y, h, stats: SolverStats):
        n = y.size
        s = self.C.size
        tol = self._newton_tol(y)

        jac = self._jacobian(problem, t, y, stats)
        iter_matrix = np.eye(s * n) - h * np.kron(self.A, jac)
        lu = lu_factor(iter_matrix)
        stats.n_lu += 1

        # Stage predictor: constant extrapolation of the current state.
        Y = np.tile(y, s)
        t_stages = t + self.C * h

        converged = False
        dz_prev = None
        for _ in range(1, self.newton_max_iter + 1):
            F = np.empty((s, n))
            for i in range(s):
                F[i] = np.asarray(problem.f(t_stages[i], Y[i * n:(i + 1) * n]),
                                  dtype=float)
            stats.n_rhs += s
            # Residual of Y_i - y - h * sum_j a_ij f_j = 0
            G = Y - np.tile(y, s) - h * (np.kron(self.A, np.eye(n)) @ F.ravel())
            dz = lu_solve(lu, -G)
            Y = Y + dz
            stats.n_newton += 1
            norm = float(np.linalg.norm(dz))
            if norm <= 1e-2 * tol:
                converged = True
                break
            if dz_prev is not None and norm > dz_prev:
                break  # diverging; let the caller shrink h
            dz_prev = norm

        if not converged:
            stats.n_newton_failed += 1
        return Y[(s - 1) * n:], None
