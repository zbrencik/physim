"""Solver correctness.

The Butcher tableaux are verified against their order conditions rather than
against a transcription of the literature: a typo in a coefficient shows up
here as a failed algebraic identity, not as a slightly-wrong answer three
modules downstream.
"""

import numpy as np
import pytest

from physim.problems import get_problem
from physim.solvers import ADAPTIVE, FIXED_STEP, SOLVERS, get_solver
from physim.solvers.explicit import RK45
from physim.solvers.implicit import TRBDF2, RadauIIA5
from physim.validation import convergence_study
from physim.validation.metrics import max_relative_error


# ------------------------------------------------------------------ tableaux
def test_radau_row_sums_match_nodes():
    assert np.allclose(RadauIIA5.A.sum(axis=1), RadauIIA5.C)


def test_radau_is_stiffly_accurate():
    """The last stage must equal the output weights, i.e. Y_s = y_{n+1}."""
    assert np.allclose(RadauIIA5.A[-1], RadauIIA5.B)
    assert np.isclose(RadauIIA5.C[-1], 1.0)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_radau_quadrature_conditions_B(k):
    """B(5): sum_i b_i c_i^(k-1) = 1/k, which gives order 5."""
    lhs = float(RadauIIA5.B @ RadauIIA5.C ** (k - 1))
    assert lhs == pytest.approx(1.0 / k, rel=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_radau_collocation_conditions_C(k):
    """C(3): sum_j a_ij c_j^(k-1) = c_i^k / k, the collocation property."""
    lhs = RadauIIA5.A @ RadauIIA5.C ** (k - 1)
    assert np.allclose(lhs, RadauIIA5.C**k / k, rtol=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_dormand_prince_order_conditions(k):
    assert float(RK45.B @ RK45.C ** (k - 1)) == pytest.approx(1.0 / k, rel=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_dormand_prince_embedded_is_fourth_order(k):
    assert float(RK45.B_HAT @ RK45.C ** (k - 1)) == pytest.approx(1.0 / k, rel=1e-12)


def test_dormand_prince_fsal_row_matches_weights():
    """Stage 7 uses the b-weights, which is what makes FSAL reuse valid."""
    assert np.allclose(RK45.A[6], RK45.B[:6])


def test_trbdf2_tableau_is_consistent():
    d, w, g = TRBDF2.D, TRBDF2.W, TRBDF2.GAMMA
    assert 2 * d == pytest.approx(g)             # stage 2 abscissa
    assert 2 * w + d == pytest.approx(1.0)       # stage 3 abscissa
    b_hat = np.array([(1 - w) / 3, (3 * w + 1) / 3, d / 3])
    assert b_hat.sum() == pytest.approx(1.0)     # embedded is consistent
    c = np.array([0.0, g, 1.0])
    assert float(b_hat @ c) == pytest.approx(0.5)  # and second order


# -------------------------------------------------------------- correctness
@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_every_solver_integrates_a_smooth_problem(name):
    problem = get_problem("exponential_decay")
    solver = get_solver(name, rtol=1e-8, atol=1e-10)
    kwargs = {} if solver.adaptive else {"n_steps": 2000}
    result = solver.solve(problem, **kwargs)
    assert result.success
    assert result.t[0] == pytest.approx(problem.t_span[0])
    assert result.t[-1] == pytest.approx(problem.t_span[1])
    assert np.all(np.diff(result.t) > 0)
    assert result.y.shape == (result.t.size, problem.dim)
    assert result.stats.n_rhs > 0


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_first_sample_is_the_initial_condition(name):
    problem = get_problem("damped_oscillator")
    solver = get_solver(name)
    kwargs = {} if solver.adaptive else {"n_steps": 500}
    result = solver.solve(problem, **kwargs)
    assert np.allclose(result.y[0], problem.y0)


def test_rk4_is_exact_on_a_cubic():
    """A fourth-order method integrates a cubic polynomial exactly."""
    from physim.problems.base import ODEProblem

    problem = ODEProblem(
        name="cubic", f=lambda t, y: np.array([3.0 * t**2]),
        y0=np.array([0.0]), t_span=(0.0, 2.0),
        exact=lambda t: (np.atleast_1d(t) ** 3)[:, None],
    )
    result = get_solver("rk4").solve(problem, n_steps=8)
    assert max_relative_error(result.y, problem.exact_at(result.t)) < 1e-14


@pytest.mark.parametrize("name,expected", [
    ("forward_euler", 1), ("heun", 2), ("rk4", 4),
    ("backward_euler", 1), ("trapezoidal", 2), ("radau_iia5", 5),
])
def test_observed_order_matches_theory(name, expected):
    study = convergence_study(name, get_problem("exponential_decay"))
    assert study.n_fitted >= 3
    assert study.observed_order == pytest.approx(expected, rel=0.12)


@pytest.mark.parametrize("name", ADAPTIVE)
def test_adaptive_methods_respect_tolerance_scaling(name):
    """Tightening the tolerance by 100x must reduce the error, not just the step."""
    problem = get_problem("damped_oscillator")
    loose = get_solver(name, rtol=1e-4, atol=1e-7).solve(problem)
    tight = get_solver(name, rtol=1e-8, atol=1e-11).solve(problem)
    e_loose = max_relative_error(loose.y, problem.exact_at(loose.t))
    e_tight = max_relative_error(tight.y, problem.exact_at(tight.t))
    assert e_tight < e_loose
    assert tight.stats.n_steps > loose.stats.n_steps


@pytest.mark.parametrize("name", FIXED_STEP)
def test_fixed_step_methods_require_a_step_size(name):
    with pytest.raises(ValueError, match="fixed-step"):
        get_solver(name).solve(get_problem("logistic"))


def test_rk45_fsal_does_not_leak_between_runs():
    """A reused solver instance must not seed stage 1 from the previous run."""
    problem = get_problem("damped_oscillator")
    solver = get_solver("rk45", rtol=1e-9, atol=1e-12)
    first = solver.solve(problem)
    second = solver.solve(problem)
    assert np.allclose(first.y[-1], second.y[-1], rtol=0, atol=0)


def test_solver_rejects_a_backwards_interval():
    with pytest.raises(ValueError, match="tf > t0"):
        get_solver("rk45").solve(get_problem("logistic"), t_span=(1.0, 0.0))


def test_unknown_solver_name_is_reported():
    with pytest.raises(KeyError, match="unknown solver"):
        get_solver("does_not_exist")


def test_energy_drift_is_bounded_for_a_high_order_method():
    """Long-horizon phase and amplitude behaviour, not just endpoint error."""
    problem = get_problem("harmonic_oscillator", t_span=(0.0, 200.0))
    result = get_solver("rk45", rtol=1e-10, atol=1e-13).solve(problem)
    energy = problem.conserved(result.y)
    drift = np.max(np.abs(energy - energy[0])) / energy[0]
    assert drift < 1e-6


def test_work_counters_are_consistent():
    result = get_solver("rk45", rtol=1e-6).solve(get_problem("logistic"))
    s = result.stats
    assert s.n_steps == s.n_accepted + s.n_rejected
    assert s.n_rhs >= 6 * s.n_steps  # six stages per step under FSAL
    assert s.elapsed_s > 0


def test_implicit_solver_counts_factorisations():
    result = get_solver("backward_euler").solve(get_problem("stiff_linear"),
                                                n_steps=100)
    assert result.stats.n_lu == 100
    assert result.stats.n_jac == 100
    assert result.stats.n_newton > 100  # at least one iteration per step


def test_analytic_and_numerical_jacobians_agree():
    problem = get_problem("robertson")
    y = np.array([0.7, 1e-5, 0.3])
    analytic = problem.jac(0.0, y)
    numerical = problem.numerical_jacobian(0.0, y)
    # Forward differences are first-order accurate, so the tolerance is set
    # relative to the largest entry rather than element by element.
    assert np.max(np.abs(analytic - numerical)) < 1e-3 * np.abs(analytic).max()


def test_missing_analytic_jacobian_falls_back_to_finite_differences():
    """An implicit method must still work when no Jacobian is supplied, and
    must land in essentially the same place as the analytic-Jacobian run:
    the Jacobian only steers the Newton iteration, it does not define the
    answer."""
    exact_jac = get_solver("backward_euler").solve(
        get_problem("stiff_linear"), n_steps=500)
    problem = get_problem("stiff_linear")
    problem.jac = None
    approx_jac = get_solver("backward_euler").solve(problem, n_steps=500)
    assert approx_jac.success
    assert approx_jac.stats.n_rhs > exact_jac.stats.n_rhs  # differencing costs
    assert max_relative_error(approx_jac.y, exact_jac.y) < 1e-8
