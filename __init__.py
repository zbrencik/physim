"""physim: numerical analysis and physical-layer simulation engine.

Two halves that share one validation philosophy: every result is scored
against a closed-form reference wherever one exists.

* ``physim.solvers`` -- explicit and implicit Runge-Kutta integrators.
* ``physim.problems`` -- test problems, most with analytical solutions.
* ``physim.phy``      -- modulation, channels, pulse shaping, BER simulation.
* ``physim.validation`` -- error metrics, order measurement, reports.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]


def __getattr__(name: str):
    # Lazy re-exports keep `import physim` cheap for the CLI.
    if name in ("get_solver", "list_solvers", "SOLVERS"):
        from physim import solvers
        return getattr(solvers, name)
    if name in ("get_problem", "list_problems", "PROBLEMS"):
        from physim import problems
        return getattr(problems, name)
    if name in ("get_modulation", "ber_sweep", "theoretical_ber"):
        from physim import phy
        return getattr(phy, name)
    raise AttributeError(f"module 'physim' has no attribute '{name}'")
