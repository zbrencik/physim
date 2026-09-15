"""Jacobians, LU factorisation and the damped Newton iteration.

Implicit Runge-Kutta stages all reduce to the same root-finding problem

    G(z) = 0,      G'(z) = I - h * gamma * J,

so the Newton machinery lives here once and is reused by every implicit
method in :mod:`physim.solvers.implicit`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def finite_difference_jacobian(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    f0: np.ndarray | None = None,
    rel_step: float | None = None,
) -> tuple[np.ndarray, int]:
    """Forward-difference Jacobian ``df_i/dy_j``.

    The per-column perturbation follows the usual compromise between
    truncation error (O(h)) and cancellation error (O(eps/h)), which is
    minimised near ``h ~ sqrt(eps) * |y|``.

    Returns the ``(d, d)`` Jacobian and the number of extra RHS evaluations
    consumed, so callers can keep their work counters honest.
    """
    y = np.asarray(y, dtype=float)
    d = y.size
    if f0 is None:
        f0 = np.asarray(f(t, y), dtype=float)
    if rel_step is None:
        rel_step = np.sqrt(np.finfo(float).eps)

    jac = np.empty((d, d), dtype=float)
    for j in range(d):
        # Scale the bump to the magnitude of the component, never to zero.
        step = rel_step * max(abs(y[j]), 1.0)
        y_pert = y.copy()
        y_pert[j] += step
        # Recover the exact increment actually representable in floating point.
        actual = y_pert[j] - y[j]
        jac[:, j] = (np.asarray(f(t, y_pert), dtype=float) - f0) / actual
    return jac, d


@dataclass
class NewtonReport:
    converged: bool
    iterations: int
    rhs_evals: int
    residual: float


def newton_solve(
    residual: Callable[[np.ndarray], np.ndarray],
    lu,
    z0: np.ndarray,
    tol: float,
    max_iter: int = 12,
    kappa: float = 1e-2,
) -> tuple[np.ndarray, NewtonReport]:
    """Simplified (frozen-Jacobian) Newton iteration with rate monitoring.

    ``lu`` is a prefactorised iteration matrix. Freezing it across iterations
    is the standard trade in stiff ODE codes: it costs a few extra iterations
    but saves an O(d^3) factorisation per iteration.

    The loop bails out early when the estimated contraction rate exceeds one,
    which lets the step-size controller reject the step and retry with a
    smaller ``h`` instead of grinding through iterations that cannot converge.
    """
    z = z0.copy()
    dz_norm_prev = None
    rate = None
    rhs_evals = 0
    last_norm = np.inf

    for it in range(1, max_iter + 1):
        g = residual(z)
        rhs_evals += 1
        dz = lu_solve(lu, -g)
        dz_norm = float(np.linalg.norm(dz))
        last_norm = dz_norm

        if dz_norm_prev is not None and dz_norm_prev > 0:
            rate = dz_norm / dz_norm_prev
            if rate >= 1.0:
                return z, NewtonReport(False, it, rhs_evals, dz_norm)

        z = z + dz

        if dz_norm == 0.0:
            return z, NewtonReport(True, it, rhs_evals, 0.0)
        if rate is not None:
            # Estimated distance to the fixed point for a linear rate `rate`.
            if rate / (1.0 - rate) * dz_norm < kappa * tol:
                return z, NewtonReport(True, it, rhs_evals, dz_norm)
        elif dz_norm < kappa * tol:
            return z, NewtonReport(True, it, rhs_evals, dz_norm)

        dz_norm_prev = dz_norm

    return z, NewtonReport(False, max_iter, rhs_evals, last_norm)


def factorize(matrix: np.ndarray):
    """LU factorisation with partial pivoting, wrapped for counting."""
    return lu_factor(matrix)


def error_norm(err: np.ndarray, y0: np.ndarray, y1: np.ndarray,
               rtol: float, atol: float) -> float:
    """RMS error norm scaled by the usual ``atol + rtol * max(|y|)`` tolerance.

    A norm of 1.0 means "exactly at tolerance", which is what the step-size
    controllers below expect.
    """
    scale = atol + rtol * np.maximum(np.abs(y0), np.abs(y1))
    return float(np.sqrt(np.mean((err / scale) ** 2)))
