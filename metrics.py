"""Error metrics for scoring a trajectory against a closed-form solution.

One definition choice matters and is stated explicitly rather than buried:
pointwise relative error is normalised by the **amplitude of the reference
over the whole trajectory**, not by its instantaneous value. A sinusoid
passes through zero, and dividing by a value that approaches zero produces
relative errors that approach infinity while the solution is in fact
perfectly accurate. Normalising by the trajectory amplitude is the
convention used throughout ``docs/VALIDATION.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class AccuracyReport:
    """Scorecard for one (problem, method, step size) triple."""

    problem: str
    method: str
    n_points: int
    h: float | None
    max_abs_error: float
    max_rel_error: float
    rms_rel_error: float
    final_rel_error: float
    n_rhs: int
    elapsed_s: float
    success: bool = True

    @property
    def max_rel_error_pct(self) -> float:
        return 100.0 * self.max_rel_error

    def passes(self, tol_pct: float = 0.01) -> bool:
        """True when the worst relative error stays under ``tol_pct`` percent."""
        return self.success and self.max_rel_error_pct <= tol_pct

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["max_rel_error_pct"] = self.max_rel_error_pct
        return d


def amplitude_scale(reference: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    """Per-component normalisation constant: the reference amplitude."""
    ref = np.atleast_2d(reference)
    scale = np.max(np.abs(ref), axis=0)
    scale = np.where(scale < floor, 1.0, scale)
    return scale


def relative_error(numeric: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Amplitude-normalised pointwise relative error, shape ``(N, d)``."""
    num = np.atleast_2d(numeric)
    ref = np.atleast_2d(reference)
    if num.shape != ref.shape:
        raise ValueError(f"shape mismatch: {num.shape} vs {ref.shape}")
    return np.abs(num - ref) / amplitude_scale(ref)


def max_relative_error(numeric: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(relative_error(numeric, reference)))


def rms_relative_error(numeric: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean(relative_error(numeric, reference) ** 2)))


def max_abs_error(numeric: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(np.atleast_2d(numeric) - np.atleast_2d(reference))))


def final_relative_error(numeric: np.ndarray, reference: np.ndarray) -> float:
    """Norm-wise relative error at the final time node."""
    a = np.atleast_2d(numeric)[-1]
    b = np.atleast_2d(reference)[-1]
    denom = np.linalg.norm(b)
    if denom < 1e-300:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def score(result, problem, h: float | None = None) -> AccuracyReport:
    """Compare a :class:`SolverResult` against the problem's exact solution."""
    if not problem.has_exact:
        raise ValueError(
            f"problem '{problem.name}' has no closed-form solution to score against"
        )
    exact = problem.exact_at(result.t)
    y = np.atleast_2d(result.y)
    return AccuracyReport(
        problem=problem.name,
        method=result.method,
        n_points=int(result.t.size),
        h=h,
        max_abs_error=max_abs_error(y, exact),
        max_rel_error=max_relative_error(y, exact),
        rms_rel_error=rms_relative_error(y, exact),
        final_rel_error=final_relative_error(y, exact),
        n_rhs=result.stats.n_rhs,
        elapsed_s=result.stats.elapsed_s,
        success=result.success,
    )


def conservation_drift(result, problem) -> float | None:
    """Relative drift of a conserved quantity, or ``None`` if none is defined.

    Energy drift is the honest way to judge a long-horizon integration: a
    method can track the reference for a while and still be systematically
    pumping energy into the system.
    """
    if problem.conserved is None:
        return None
    q = np.asarray(problem.conserved(np.atleast_2d(result.y)), dtype=float)
    q0 = q[0]
    if abs(q0) < 1e-300:
        return float(np.max(np.abs(q - q0)))
    return float(np.max(np.abs(q - q0)) / abs(q0))
