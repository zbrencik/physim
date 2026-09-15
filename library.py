"""Reference problems: non-stiff, stiff, and physically motivated.

Every problem in the first group has a closed-form solution, which is what
makes the accuracy claims in ``docs/VALIDATION.md`` checkable rather than
self-reported. The stiff group exists to separate methods that are merely
accurate from methods that are also stable at useful step sizes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import expm

from physim.problems.base import ODEProblem


def spectral_ratio(A: np.ndarray) -> float | None:
    """Ratio of the fastest to the slowest mode, by eigenvalue modulus.

    Using ``|lambda|`` rather than ``|Re lambda|`` matters for oscillatory
    systems: an explicit method's step size is limited by the distance of the
    eigenvalue from the origin, so a lightly damped high-frequency mode
    constrains the step exactly as a strongly damped one does, even though
    its real part is small.
    """
    eig = np.linalg.eigvals(np.atleast_2d(A))
    mag = np.abs(eig)
    mag = mag[mag > 1e-14]
    return float(mag.max() / mag.min()) if mag.size else None


# --------------------------------------------------------------------------
# Generic linear system: y' = A y with exact solution y(t) = expm(A t) y0
# --------------------------------------------------------------------------
def linear_system(A: np.ndarray, y0: np.ndarray, t_span=(0.0, 1.0),
                  name: str = "linear", description: str = "",
                  labels: tuple[str, ...] = ()) -> ODEProblem:
    """Build a constant-coefficient linear IVP with a matrix-exponential reference.

    ``expm`` is used rather than an eigendecomposition so the reference stays
    valid for defective (non-diagonalisable) ``A``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))

    def f(t: float, y: np.ndarray) -> np.ndarray:
        return A @ y

    def jac(t: float, y: np.ndarray) -> np.ndarray:
        return A

    def exact(t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t)
        out = np.empty((t.size, y0.size))
        for i, ti in enumerate(t):
            out[i] = expm(A * float(ti)) @ y0
        return out

    ratio = spectral_ratio(A)

    return ODEProblem(
        name=name, f=f, y0=y0, t_span=t_span, description=description,
        jac=jac, exact=exact, stiff=(ratio is not None and ratio > 1e2),
        stiffness_ratio=ratio, labels=labels,
        params={"dim": int(y0.size)},
    )


# --------------------------------------------------------------------------
# Non-stiff problems with closed-form solutions
# --------------------------------------------------------------------------
def exponential_decay(lam: float = 2.0, y0: float = 1.0,
                      t_span=(0.0, 5.0)) -> ODEProblem:
    """``y' = -lambda y``. The simplest possible order/stability check."""

    def f(t, y):
        return -lam * y

    def jac(t, y):
        return np.array([[-lam]])

    def exact(t):
        return (y0 * np.exp(-lam * np.atleast_1d(t)))[:, None]

    return ODEProblem(
        name="exponential_decay",
        description="y' = -lambda*y, exact y = y0*exp(-lambda*t)",
        f=f, y0=np.array([y0]), t_span=t_span, jac=jac, exact=exact,
        labels=("y",), params={"lambda": lam, "y0": y0},
    )


def logistic(r: float = 1.5, K: float = 10.0, y0: float = 0.5,
             t_span=(0.0, 10.0)) -> ODEProblem:
    """Logistic growth: nonlinear, but still analytically solvable."""

    def f(t, y):
        return r * y * (1.0 - y / K)

    def jac(t, y):
        return np.array([[r * (1.0 - 2.0 * y[0] / K)]])

    def exact(t):
        t = np.atleast_1d(t)
        return (K * y0 / (y0 + (K - y0) * np.exp(-r * t)))[:, None]

    return ODEProblem(
        name="logistic",
        description="y' = r*y*(1 - y/K), exact logistic curve",
        f=f, y0=np.array([y0]), t_span=t_span, jac=jac, exact=exact,
        labels=("population",), params={"r": r, "K": K, "y0": y0},
    )


def harmonic_oscillator(omega: float = 2.0, t_span=(0.0, 20.0)) -> ODEProblem:
    """Undamped oscillator ``x'' + omega^2 x = 0``.

    Long-horizon integration exposes phase error, which is invisible in a
    one-step local error test but dominates any real simulation.
    """
    A = np.array([[0.0, 1.0], [-omega**2, 0.0]])
    y0 = np.array([1.0, 0.0])

    def exact(t):
        t = np.atleast_1d(t)
        return np.column_stack([np.cos(omega * t), -omega * np.sin(omega * t)])

    prob = linear_system(A, y0, t_span, name="harmonic_oscillator",
                         description="x'' + omega^2 x = 0 (energy conserving)",
                         labels=("x", "v"))
    prob.exact = exact
    prob.params = {"omega": omega}
    prob.conserved = lambda y: 0.5 * (y[..., 1] ** 2 + omega**2 * y[..., 0] ** 2)
    return prob


def damped_oscillator(omega: float = 4.0, zeta: float = 0.15,
                      t_span=(0.0, 12.0)) -> ODEProblem:
    """Underdamped second-order system ``x'' + 2*zeta*omega x' + omega^2 x = 0``."""
    if not 0.0 < zeta < 1.0:
        raise ValueError("closed form below assumes underdamped 0 < zeta < 1")
    A = np.array([[0.0, 1.0], [-omega**2, -2.0 * zeta * omega]])
    y0 = np.array([1.0, 0.0])
    wd = omega * np.sqrt(1.0 - zeta**2)

    def exact(t):
        t = np.atleast_1d(t)
        env = np.exp(-zeta * omega * t)
        x = env * (np.cos(wd * t) + (zeta * omega / wd) * np.sin(wd * t))
        v = -env * (omega**2 / wd) * np.sin(wd * t)
        return np.column_stack([x, v])

    prob = linear_system(A, y0, t_span, name="damped_oscillator",
                         description="x'' + 2*zeta*omega*x' + omega^2*x = 0",
                         labels=("x", "v"))
    prob.exact = exact
    prob.params = {"omega": omega, "zeta": zeta}
    return prob


def rlc_step(R: float = 50.0, L: float = 1e-6, C: float = 1e-9,
             v_in: float = 1.0, t_span=(0.0, 2e-7)) -> ODEProblem:
    """Series RLC driven by a voltage step: the receiver front-end model.

    State is ``[v_C, i_L]``. The step response is the canonical second-order
    system, so the closed form is exact in the underdamped regime and the
    problem doubles as a link between the ODE half of the engine and the
    analogue front-end assumed by the PHY half.
    """
    omega0 = 1.0 / np.sqrt(L * C)
    zeta = (R / 2.0) * np.sqrt(C / L)
    if zeta >= 1.0:
        raise ValueError("closed form below assumes an underdamped RLC")
    wd = omega0 * np.sqrt(1.0 - zeta**2)

    A = np.array([[0.0, 1.0 / C], [-1.0 / L, -R / L]])
    b = np.array([0.0, v_in / L])
    y0 = np.array([0.0, 0.0])

    def f(t, y):
        return A @ y + b

    def jac(t, y):
        return A

    def exact(t):
        t = np.atleast_1d(t)
        env = np.exp(-zeta * omega0 * t)
        vc = v_in * (1.0 - env * (np.cos(wd * t)
                                  + (zeta * omega0 / wd) * np.sin(wd * t)))
        il = v_in * C * env * (omega0**2 / wd) * np.sin(wd * t)
        return np.column_stack([vc, il])

    return ODEProblem(
        name="rlc_step",
        description="Series RLC step response (front-end transient)",
        f=f, y0=y0, t_span=t_span, jac=jac, exact=exact,
        labels=("v_C", "i_L"), stiffness_ratio=spectral_ratio(A),
        params={"R": R, "L": L, "C": C, "omega0": float(omega0),
                "zeta": float(zeta)},
    )


def pll_frequency_step(omega_n: float = 2 * np.pi * 1e4, zeta: float = 0.707,
                       delta_f: float = 1e3, t_span=(0.0, 1e-3)) -> ODEProblem:
    """Linearised second-order PLL responding to a carrier frequency step.

    State is ``[phase_error, d(phase_error)/dt]``. Carrier recovery settling
    time is what sets the preamble length in a burst modem, so this is the
    ODE the PHY side actually cares about.
    """
    A = np.array([[0.0, 1.0], [-omega_n**2, -2.0 * zeta * omega_n]])
    dw = 2.0 * np.pi * delta_f
    y0 = np.array([0.0, dw])
    wd = omega_n * np.sqrt(1.0 - zeta**2)

    def exact(t):
        t = np.atleast_1d(t)
        env = np.exp(-zeta * omega_n * t)
        e = (dw / wd) * env * np.sin(wd * t)
        de = dw * env * (np.cos(wd * t) - (zeta * omega_n / wd) * np.sin(wd * t))
        return np.column_stack([e, de])

    prob = linear_system(A, y0, t_span, name="pll_frequency_step",
                         description="Second-order PLL phase error after a "
                                     "frequency step",
                         labels=("phase_error_rad", "d_phase_error"))
    prob.exact = exact
    prob.params = {"omega_n": float(omega_n), "zeta": zeta, "delta_f": delta_f}
    return prob


# --------------------------------------------------------------------------
# Stiff problems
# --------------------------------------------------------------------------
def stiff_linear(lam_fast: float = -1000.0, lam_slow: float = -1.0,
                 t_span=(0.0, 2.0)) -> ODEProblem:
    """Two-timescale linear system with a 1000:1 stiffness ratio.

    ``A`` is built by similarity transform so it is not diagonal: a solver
    cannot get the right answer by accidentally decoupling the components.
    """
    V = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    A = V @ np.diag([lam_fast, lam_slow]) @ np.linalg.inv(V)
    y0 = np.array([1.0, 0.0])
    return linear_system(
        A, y0, t_span, name="stiff_linear",
        description=f"Linear system, eigenvalues {lam_fast} and {lam_slow}",
        labels=("y0", "y1"),
    )


def prothero_robinson(lam: float = -1e4, t_span=(0.0, 2.0)) -> ODEProblem:
    """``y' = lambda (y - phi(t)) + phi'(t)`` with ``phi(t) = sin(t)``.

    The exact solution is ``phi`` itself for a consistent initial condition,
    independent of ``lambda``. Making ``lambda`` arbitrarily negative makes
    the problem arbitrarily stiff without changing the answer, which isolates
    order reduction: an A-stable method with poor stage order visibly loses
    accuracy here while the reference stays fixed.
    """

    def phi(t):
        return np.sin(t)

    def dphi(t):
        return np.cos(t)

    def f(t, y):
        return lam * (y - phi(t)) + dphi(t)

    def jac(t, y):
        return np.array([[lam]])

    def exact(t):
        return phi(np.atleast_1d(t))[:, None]

    return ODEProblem(
        name="prothero_robinson",
        description="Prothero-Robinson stiff scalar test, exact y = sin(t)",
        f=f, y0=np.array([phi(t_span[0])]), t_span=t_span, jac=jac, exact=exact,
        stiff=True, stiffness_ratio=abs(lam), labels=("y",),
        params={"lambda": lam},
    )


def van_der_pol(mu: float = 100.0, t_span: tuple[float, float] | None = None
                ) -> ODEProblem:
    """Van der Pol oscillator in the relaxation regime.

    No closed form exists; validation is against a tightly converged implicit
    reference and against the period asymptotics of the limit cycle.
    """
    t_span = t_span or (0.0, 3.0 * mu)

    def f(t, y):
        return np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])

    def jac(t, y):
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] ** 2)],
        ])

    return ODEProblem(
        name="van_der_pol",
        description=f"Van der Pol relaxation oscillator, mu = {mu:g}",
        f=f, y0=np.array([2.0, 0.0]), t_span=t_span, jac=jac,
        stiff=True, stiffness_ratio=float(mu**2), labels=("x", "v"),
        params={"mu": mu},
    )


def robertson(t_span=(0.0, 1e4)) -> ODEProblem:
    """Robertson's autocatalytic kinetics: the standard stiff benchmark.

    Concentrations span nine orders of magnitude and must sum to one, which
    gives a conservation invariant to check even though no closed form exists.
    """
    k1, k2, k3 = 0.04, 3.0e7, 1.0e4

    def f(t, y):
        y1, y2, y3 = y
        return np.array([
            -k1 * y1 + k3 * y2 * y3,
            k1 * y1 - k3 * y2 * y3 - k2 * y2 ** 2,
            k2 * y2 ** 2,
        ])

    def jac(t, y):
        y1, y2, y3 = y
        return np.array([
            [-k1, k3 * y3, k3 * y2],
            [k1, -k3 * y3 - 2.0 * k2 * y2, -k3 * y2],
            [0.0, 2.0 * k2 * y2, 0.0],
        ])

    return ODEProblem(
        name="robertson",
        description="Robertson chemical kinetics, invariant sum(y) = 1",
        f=f, y0=np.array([1.0, 0.0, 0.0]), t_span=t_span, jac=jac,
        stiff=True, stiffness_ratio=1e9, labels=("A", "B", "C"),
        params={"k1": k1, "k2": k2, "k3": k3},
        conserved=lambda y: np.sum(y, axis=-1),
    )


def transmission_line(n_sections: int = 20, length_m: float = 10.0,
                      R_per_m: float = 0.5, L_per_m: float = 250e-9,
                      G_per_m: float = 1e-6, C_per_m: float = 100e-12,
                      v_source: float = 1.0, z_load: float = 50.0,
                      t_span=(0.0, 4e-7)) -> ODEProblem:
    """Lumped RLGC transmission line: the telegrapher's equations, discretised.

    Splitting a line of length ``length_m`` into ``n_sections`` LC sections
    turns the PDE into a ``2n``-dimensional linear ODE. The spatial
    discretisation is what creates the stiffness: the fastest mode scales
    like ``n`` while the transit time stays fixed, so refining the mesh makes
    an explicit solver progressively more expensive for no extra accuracy.

    State ordering is ``[i_1, v_1, i_2, v_2, ...]``; the source is a step
    through ``R_per_m`` and the far end is terminated in ``z_load``.
    """
    n = int(n_sections)
    dx = length_m / n
    R, L, G, C = R_per_m * dx, L_per_m * dx, G_per_m * dx, C_per_m * dx

    dim = 2 * n
    A = np.zeros((dim, dim))
    b = np.zeros(dim)

    for k in range(n):
        i_idx, v_idx = 2 * k, 2 * k + 1
        # Inductor: L di/dt = v_upstream - v_node - R i
        A[i_idx, i_idx] = -R / L
        A[i_idx, v_idx] = -1.0 / L
        if k == 0:
            b[i_idx] = v_source / L
        else:
            A[i_idx, v_idx - 2] = 1.0 / L
        # Capacitor: C dv/dt = i_in - i_out - G v
        A[v_idx, i_idx] = 1.0 / C
        A[v_idx, v_idx] = -G / C
        if k < n - 1:
            A[v_idx, v_idx + 1] = -1.0 / C
        else:
            A[v_idx, v_idx] += -1.0 / (z_load * C)

    y0 = np.zeros(dim)

    def f(t, y):
        return A @ y + b

    def jac(t, y):
        return A

    # Affine system: shift to the steady state so expm gives the exact answer.
    y_ss = np.linalg.solve(A, -b)

    def exact(t):
        t = np.atleast_1d(t)
        out = np.empty((t.size, dim))
        d0 = y0 - y_ss
        for i, ti in enumerate(t):
            out[i] = expm(A * float(ti)) @ d0 + y_ss
        return out

    ratio = spectral_ratio(A)

    labels = tuple(
        f"{'i' if j % 2 == 0 else 'v'}{j // 2 + 1}" for j in range(dim)
    )
    # The difficulty here is not a wide eigenvalue *ratio* but the absolute
    # size of the spectrum: an explicit method needs h < ~2.78/|lambda|_max
    # for stability, and |lambda|_max grows linearly with the section count.
    lam_max = float(np.abs(np.linalg.eigvals(A)).max())
    h_limit = 2.78 / lam_max
    span = float(t_span[1] - t_span[0])
    series_r = R_per_m * length_m
    return ODEProblem(
        name="transmission_line",
        description=f"RLGC line, {n} sections, {length_m:g} m, "
                    f"{z_load:g} ohm load, |lambda|max = {lam_max:.3g} rad/s",
        f=f, y0=y0, t_span=t_span, jac=jac, exact=exact,
        # Flagged stiff on the operational test rather than the spectral
        # ratio: what makes this problem hard is that an explicit method
        # needs ~115 steps per transit time for *stability* alone, long
        # before accuracy asks for anything.
        stiff=span / h_limit > 1e2, stiffness_ratio=ratio, labels=labels,
        params={"n_sections": n, "length_m": length_m, "z_load": z_load,
                "series_resistance_ohm": series_r,
                "far_end_index": dim - 1,
                "dc_far_end_v": v_source * z_load / (z_load + series_r),
                "lambda_max": lam_max,
                "explicit_h_limit": h_limit,
                "explicit_steps_required": span / h_limit},
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
PROBLEMS: dict[str, Callable[..., ODEProblem]] = {
    "exponential_decay": exponential_decay,
    "logistic": logistic,
    "harmonic_oscillator": harmonic_oscillator,
    "damped_oscillator": damped_oscillator,
    "rlc_step": rlc_step,
    "pll_frequency_step": pll_frequency_step,
    "stiff_linear": stiff_linear,
    "prothero_robinson": prothero_robinson,
    "van_der_pol": van_der_pol,
    "robertson": robertson,
    "transmission_line": transmission_line,
}

#: Problems carrying a closed-form solution, used by the accuracy harness.
ANALYTIC_PROBLEMS = (
    "exponential_decay", "logistic", "harmonic_oscillator",
    "damped_oscillator", "rlc_step", "pll_frequency_step",
    "stiff_linear", "prothero_robinson", "transmission_line",
)


def get_problem(name: str, **kwargs) -> ODEProblem:
    """Instantiate a problem by name, forwarding keyword overrides."""
    try:
        factory = PROBLEMS[name]
    except KeyError:
        raise KeyError(
            f"unknown problem '{name}'. Available: {sorted(PROBLEMS)}"
        ) from None
    return factory(**kwargs)


def list_problems() -> list[dict]:
    return [get_problem(name).meta() for name in PROBLEMS]
