"""Stiff behaviour.

The point of an implicit method is not that it is more accurate per step, it
is that its stability does not depend on the step size. These tests assert
that difference rather than assuming it.
"""

import numpy as np
import pytest

from physim.problems import get_problem
from physim.solvers import get_solver
from physim.validation.metrics import max_relative_error, score


def test_stiff_problems_report_a_large_spectral_ratio():
    assert get_problem("stiff_linear").stiffness_ratio >= 1e3
    assert get_problem("prothero_robinson").stiffness_ratio >= 1e3
    assert get_problem("robertson").stiffness_ratio >= 1e6


def test_transmission_line_is_stiff_by_the_operational_test():
    """Its eigenvalues are all of similar magnitude, so the usual spectral
    ratio says 'not stiff' while an explicit method still needs a hundred
    steps per transit time purely to stay stable. Stiffness is about the
    step being set by stability rather than accuracy, and the flag follows
    that definition."""
    problem = get_problem("transmission_line")
    assert problem.stiffness_ratio < 100          # ratio test says nothing
    assert problem.params["explicit_steps_required"] > 100
    assert problem.stiff


def test_explicit_method_diverges_beyond_its_stability_limit():
    """Forward Euler with h > 2/|lambda| must blow up, and must say so."""
    problem = get_problem("prothero_robinson")
    result = get_solver("forward_euler").solve(problem, n_steps=200)
    assert not result.success
    assert "not finite" in result.message or "diverged" in result.message


def test_explicit_method_is_stable_inside_its_limit():
    problem = get_problem("prothero_robinson")
    lam = abs(problem.params["lambda"])
    span = problem.t_span[1] - problem.t_span[0]
    n_steps = int(np.ceil(span * lam / 1.5))  # h = 1.5/|lambda| < 2/|lambda|
    result = get_solver("forward_euler").solve(problem, n_steps=n_steps)
    assert result.success


@pytest.mark.parametrize("name", ["backward_euler", "trapezoidal", "trbdf2",
                                  "radau_iia5"])
def test_implicit_methods_survive_the_same_step_size(name):
    """Same problem, same 200 steps, no stability constraint."""
    problem = get_problem("prothero_robinson")
    solver = get_solver(name, rtol=1e-8, atol=1e-10)
    kwargs = {} if solver.adaptive else {"n_steps": 200}
    result = solver.solve(problem, **kwargs)
    assert result.success
    assert np.all(np.isfinite(result.y))


def test_implicit_accuracy_ranks_by_order_on_a_stiff_problem():
    """Higher order should mean smaller error at equal step count."""
    problem = get_problem("prothero_robinson")
    errors = {}
    for name in ("backward_euler", "trapezoidal", "radau_iia5"):
        result = get_solver(name).solve(problem, n_steps=200)
        errors[name] = score(result, problem).max_rel_error_pct
    assert errors["trapezoidal"] < errors["backward_euler"]
    assert errors["radau_iia5"] < errors["trapezoidal"]
    assert errors["radau_iia5"] < 1e-6


def test_l_stability_damps_a_stiff_transient_faster_than_a_stability():
    """The trapezoidal rule is A-stable but not L-stable: at h|lambda| >> 1 its
    amplification factor tends to -1, so a stiff transient rings instead of
    vanishing. Backward Euler and Radau IIA damp it in one step."""
    problem = get_problem("stiff_linear")
    lam = float(np.min(problem.eigenvalues().real))
    h = 20.0 / abs(lam)  # deep in the stiff regime
    n_steps = max(2, int(np.ceil((problem.t_span[1] - problem.t_span[0]) / h)))

    def early_oscillation(name):
        y = get_solver(name).solve(problem, n_steps=n_steps).y
        d = np.diff(y[:6, 0])
        return np.sum(np.diff(np.sign(d)) != 0)

    assert early_oscillation("trapezoidal") > early_oscillation("backward_euler")
    assert early_oscillation("radau_iia5") == 0


def test_robertson_conserves_total_mass():
    """A three-species kinetics system over nine decades of time: the sum of
    concentrations is an exact invariant, so its drift is a pure measure of
    solver quality. The step must be adaptive here, because the initial
    transient is a million times faster than the final approach."""
    problem = get_problem("robertson")
    result = get_solver("trbdf2", rtol=1e-10, atol=1e-14).solve(problem)
    assert result.success
    total = problem.conserved(result.y)
    assert np.max(np.abs(total - 1.0)) < 1e-10


def test_robertson_defeats_an_explicit_method():
    """The stability limit is set by a mode with a 1e-9 s time constant while
    the run covers 1e4 s, so an explicit solver exhausts its step budget."""
    problem = get_problem("robertson")
    result = get_solver("rk45", rtol=1e-8, atol=1e-12,
                        max_steps=200_000).solve(problem)
    assert not result.success


def test_van_der_pol_stiff_limit_cycle_is_reached():
    """No closed form here, so the check is physical: the relaxation
    oscillator must settle onto a limit cycle with amplitude near 2."""
    problem = get_problem("van_der_pol")
    result = get_solver("trbdf2", rtol=1e-10, atol=1e-12).solve(problem)
    assert result.success
    x = result.y[:, 0]
    # Amplitude: 2 + O(mu^(-4/3)), so 2.001 at mu = 100.
    assert np.max(x) == pytest.approx(2.0, rel=0.01)
    assert np.min(x) == pytest.approx(-2.0, rel=0.01)

    # Period: the relaxation limit gives T -> (3 - 2*ln2)*mu. Measured as
    # the first return to the initial state rather than by counting zero
    # crossings, which is ambiguous when the run covers under two cycles.
    mu = problem.params["mu"]
    distance = np.linalg.norm(result.y - problem.y0, axis=1)
    late = result.t > 0.25 * (3 - 2 * np.log(2)) * mu
    period = result.t[late][np.argmin(distance[late])]
    assert period == pytest.approx((3 - 2 * np.log(2)) * mu, rel=0.02)


def test_transmission_line_matches_its_matrix_exponential():
    """A 40-state RLGC ladder solved by Radau against expm of the same system."""
    problem = get_problem("transmission_line")
    result = get_solver("radau_iia5", rtol=1e-10, atol=1e-14).solve(
        problem, n_steps=3000)
    assert result.success
    assert max_relative_error(result.y, problem.exact_at(result.t)) < 1e-6


def test_transmission_line_settles_towards_the_resistive_divider():
    """Physics check: the far end heads for the DC divider set by the load
    and the line's own series resistance, 50/(50+5) = 0.909. The window ends
    after eight transit times, so some ringing is still present."""
    problem = get_problem("transmission_line")
    result = get_solver("radau_iia5", rtol=1e-10, atol=1e-14).solve(
        problem, n_steps=3000)
    far_end = result.y[:, problem.params["far_end_index"]]
    assert far_end[-1] == pytest.approx(problem.params["dc_far_end_v"], rel=0.02)
    # The step has to arrive, not appear instantly: nothing moves at the far
    # end until roughly one transit time has passed.
    transit_s = 1e-8
    assert np.max(np.abs(far_end[result.t < 0.5 * transit_s])) < 0.05


def test_fixed_step_implicit_reports_a_newton_failure_instead_of_guessing():
    """At mu = 100 the van der Pol switch is nearly vertical. A fixed-step
    Newton cannot resolve it at any practical step size; the solver must say
    so rather than return the last iterate as if it were a solution."""
    problem = get_problem("van_der_pol")
    result = get_solver("radau_iia5").solve(problem, n_steps=6000)
    assert not result.success
    assert "Newton" in result.message
    assert result.stats.n_newton_failed > 0


def test_explicit_step_limit_is_reported_and_honest():
    """The documented explicit limit must actually be the edge of stability."""
    problem = get_problem("transmission_line")
    h_limit = problem.params["explicit_h_limit"]
    span = problem.t_span[1] - problem.t_span[0]
    safe = get_solver("rk4").solve(problem, n_steps=int(span / (0.5 * h_limit)))
    unsafe = get_solver("rk4").solve(problem, n_steps=int(span / (3.0 * h_limit)))
    assert safe.success
    assert not unsafe.success
