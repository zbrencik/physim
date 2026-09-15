"""Problem definition: right-hand side, Jacobian, and reference solution.

A problem is the unit of validation in this engine. Where a closed-form
solution exists it is attached to the problem itself, so any solver can be
scored against truth rather than against another solver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]
Jac = Callable[[float, np.ndarray], np.ndarray]
Exact = Callable[[np.ndarray], np.ndarray]


@dataclass
class ODEProblem:
    """An initial value problem ``y' = f(t, y)``, ``y(t0) = y0``.

    Parameters
    ----------
    name, description
        Identifiers used by the CLI and the HTTP API.
    f
        Right-hand side. Must accept ``(t, y)`` with ``y`` of shape ``(d,)``
        and return shape ``(d,)``.
    y0
        Initial state.
    t_span
        Default integration interval ``(t0, tf)``.
    jac
        Analytic Jacobian ``df/dy``. Optional; implicit solvers fall back to
        forward differences when it is absent.
    exact
        Closed-form solution, vectorised over time: takes ``t`` of shape
        ``(N,)`` and returns ``(N, d)``. Present only for problems with a
        known analytical solution.
    stiffness_ratio
        |Re lambda_max| / |Re lambda_min| at the initial condition, when the
        problem is linear or locally linearisable. Used to label problems in
        the reports.
    """

    name: str
    f: RHS
    y0: np.ndarray
    t_span: tuple[float, float]
    description: str = ""
    jac: Jac | None = None
    exact: Exact | None = None
    stiff: bool = False
    stiffness_ratio: float | None = None
    labels: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    conserved: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.y0 = np.atleast_1d(np.asarray(self.y0, dtype=float))
        if not self.labels:
            self.labels = tuple(f"y{i}" for i in range(self.y0.size))

    @property
    def dim(self) -> int:
        return self.y0.size

    @property
    def has_exact(self) -> bool:
        return self.exact is not None

    def exact_at(self, t: np.ndarray) -> np.ndarray:
        if self.exact is None:
            raise ValueError(f"problem '{self.name}' has no closed-form solution")
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.asarray(self.exact(t), dtype=float)
        if out.ndim == 1:
            out = out[:, None]
        return out

    def numerical_jacobian(self, t: float, y: np.ndarray) -> np.ndarray:
        from physim.core.linalg import finite_difference_jacobian

        jac, _ = finite_difference_jacobian(self.f, t, y)
        return jac

    def eigenvalues(self, t: float | None = None,
                    y: np.ndarray | None = None) -> np.ndarray:
        """Spectrum of the (possibly frozen) Jacobian, for stiffness reporting."""
        t = self.t_span[0] if t is None else t
        y = self.y0 if y is None else y
        jac = self.jac(t, y) if self.jac is not None else self.numerical_jacobian(t, y)
        return np.linalg.eigvals(np.atleast_2d(jac))

    def meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "dim": self.dim,
            "t_span": list(self.t_span),
            "y0": self.y0.tolist(),
            "labels": list(self.labels),
            "stiff": self.stiff,
            "stiffness_ratio": self.stiffness_ratio,
            "has_exact": self.has_exact,
            "has_analytic_jacobian": self.jac is not None,
            "params": {k: v for k, v in self.params.items()
                       if isinstance(v, (int, float, str, bool))},
        }
