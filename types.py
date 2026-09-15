"""Core data containers shared by every solver in the engine.

The containers are deliberately plain (`dataclass` + numpy arrays) so that
results can be serialised to JSON for the HTTP API without an extra mapping
layer, while still being cheap to pass around inside tight numerical loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SolverStats:
    """Work counters for a single integration.

    These are the numbers that actually matter when comparing an explicit
    method against an implicit one: wall-clock time alone hides the fact that
    a stiff solver trades many cheap right-hand-side evaluations for a few
    expensive Jacobian factorisations.
    """

    n_steps: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_rhs: int = 0
    n_jac: int = 0
    n_lu: int = 0
    n_newton: int = 0
    n_newton_failed: int = 0
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_rhs": self.n_rhs,
            "n_jac": self.n_jac,
            "n_lu": self.n_lu,
            "n_newton": self.n_newton,
            "n_newton_failed": self.n_newton_failed,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class SolverResult:
    """Output of an integration over ``t_span``.

    Attributes
    ----------
    t : (N,) array of time nodes, strictly increasing.
    y : (N, d) array of states, ``y[i]`` is the state at ``t[i]``.
    method : name of the method that produced the trajectory.
    stats : work counters.
    success : ``False`` if the step size collapsed or Newton failed to
        converge repeatedly, in which case ``t``/``y`` hold the partial
        trajectory computed before the failure.
    message : human readable termination reason.
    """

    t: np.ndarray
    y: np.ndarray
    method: str
    stats: SolverStats = field(default_factory=SolverStats)
    success: bool = True
    message: str = "integration completed"

    @property
    def y_final(self) -> np.ndarray:
        return self.y[-1]

    @property
    def n_points(self) -> int:
        return self.t.size

    def as_dict(self, max_points: int | None = None) -> dict[str, Any]:
        """JSON-ready view, optionally decimated to ``max_points`` samples."""
        t, y = self.t, self.y
        if max_points is not None and t.size > max_points:
            idx = np.unique(
                np.linspace(0, t.size - 1, max_points).round().astype(int)
            )
            t, y = t[idx], y[idx]
        return {
            "method": self.method,
            "t": t.tolist(),
            "y": y.T.tolist(),  # one list per state component
            "success": self.success,
            "message": self.message,
            "stats": self.stats.as_dict(),
        }
