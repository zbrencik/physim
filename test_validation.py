"""The validation harness itself.

If the scoring code is wrong, every number in the documentation is wrong,
so the metrics are tested against cases with a known answer.
"""

import json

import numpy as np
import pytest

from physim.problems import get_problem
from physim.solvers import get_solver
from physim.validation import convergence_study, work_precision
from physim.validation.metrics import (
    conservation_drift,
    max_relative_error,
    relative_error,
    rms_relative_error,
    score,
)
from physim.validation.report import run_all


def test_relative_error_is_normalised_by_amplitude_not_pointwise():
    """A trajectory that crosses zero would give an infinite pointwise
    relative error; normalising by the amplitude of the reference keeps the
    metric meaningful for oscillators."""
    reference = np.sin(np.linspace(0, 2 * np.pi, 101))[:, None]
    numeric = reference + 1e-3
    err = relative_error(numeric, reference)
    assert np.all(np.isfinite(err))
    assert np.max(err) == pytest.approx(1e-3, rel=1e-9)


def test_error_metrics_are_ordered():
    rng = np.random.default_rng(1)
    ref = np.ones((500, 2))
    num = ref + rng.normal(scale=1e-3, size=ref.shape)
    assert rms_relative_error(num, ref) <= max_relative_error(num, ref)


def test_exact_agreement_scores_zero():
    ref = np.linspace(1, 2, 50)[:, None]
    assert max_relative_error(ref, ref) == 0.0


def test_score_reports_percentages_and_applies_the_tolerance():
    problem = get_problem("exponential_decay")
    result = get_solver("rk45", rtol=1e-10, atol=1e-12).solve(problem)
    report = score(result, problem)
    assert report.max_rel_error_pct == pytest.approx(
        100 * report.max_rel_error, rel=1e-12)
    assert report.passes(0.01)
    assert not report.passes(1e-12)


def test_score_fails_a_deliberately_coarse_run():
    problem = get_problem("damped_oscillator")
    result = get_solver("forward_euler").solve(problem, n_steps=60)
    assert not score(result, problem).passes(0.01)


def test_convergence_drops_points_in_the_round_off_floor():
    """Once the error reaches machine precision the slope carries no
    information about the method's order, so those points must be excluded
    rather than allowed to flatten the fit."""
    study = convergence_study("radau_iia5", get_problem("exponential_decay"),
                              step_counts=(50, 100, 200, 400, 800, 1600, 3200))
    assert study.n_fitted < 7
    assert study.observed_order == pytest.approx(5, rel=0.1)


def test_convergence_reports_nan_when_no_asymptotic_range_exists():
    """Forward Euler at these step sizes never leaves the pre-asymptotic
    regime on this problem; inventing a slope would be worse than saying so."""
    study = convergence_study("forward_euler", get_problem("prothero_robinson"),
                              step_counts=(10, 20, 40))
    assert np.isnan(study.observed_order) or study.n_fitted == 0


def test_convergence_errors_shrink_monotonically():
    study = convergence_study("rk4", get_problem("damped_oscillator"))
    assert np.all(np.diff(study.errors) < 0)
    assert np.all(np.diff(study.step_sizes) < 0)


def test_halving_the_step_divides_the_error_by_two_to_the_order():
    """The definition of order p, checked directly rather than through a fit."""
    study = convergence_study("rk4", get_problem("exponential_decay"),
                              step_counts=(100, 200))
    ratio = study.errors[0] / study.errors[1]
    assert ratio == pytest.approx(2 ** 4, rel=0.1)


def test_work_precision_trades_cost_against_accuracy():
    """A higher-order method should reach a given accuracy with fewer
    right-hand-side evaluations, which is the only comparison that matters."""
    curves = work_precision(["heun", "rk4"], get_problem("damped_oscillator"),
                            step_counts=(100, 200, 400, 800, 1600))

    def error_at(points, n_rhs):
        """Accuracy reached for a given number of right-hand-side calls.

        Compared at equal work rather than equal error: both curves cover
        the same range of work, so no extrapolation is involved, whereas a
        fixed error target can fall outside one method's reachable range.
        """
        work = np.log([p["n_rhs"] for p in points])
        err = np.log([p["error"] for p in points])
        return float(np.exp(np.interp(np.log(n_rhs), work, err)))

    budget = 4000  # inside the measured range for both methods
    assert error_at(curves["rk4"], budget) < error_at(curves["heun"], budget)
    # And the cheapest run of each must still be the least accurate one.
    for points in curves.values():
        assert points[0]["n_rhs"] < points[-1]["n_rhs"]
        assert points[0]["error"] > points[-1]["error"]


def test_conservation_drift_is_none_without_an_invariant():
    problem = get_problem("logistic")
    result = get_solver("rk4").solve(problem, n_steps=200)
    assert conservation_drift(result, problem) is None


def test_conservation_drift_is_small_for_a_good_method():
    problem = get_problem("harmonic_oscillator")
    result = get_solver("rk45", rtol=1e-10, atol=1e-13).solve(problem)
    assert conservation_drift(result, problem) < 1e-8


def test_quick_report_runs_end_to_end_and_serialises(tmp_path):
    report = run_all(quick=True)
    assert report.accuracy
    assert report.convergence
    assert report.stiff
    assert report.phy
    assert 0.0 <= report.accuracy_pass_rate <= 1.0

    path = report.write(tmp_path / "validation.json")
    payload = json.loads(path.read_text())
    assert payload["environment"]["numpy"]
    assert payload["summary"]["accuracy_pass_rate"] == pytest.approx(
        report.accuracy_pass_rate)
    # Every recorded number must survive a JSON round trip unchanged.
    assert isinstance(payload["accuracy"][0]["max_rel_error_pct"], float)


def test_quick_report_meets_the_headline_accuracy_claim():
    """The project claims sub-0.01% agreement against closed forms; the
    harness must be able to demonstrate it, not merely measure it."""
    report = run_all(quick=True)
    assert report.accuracy_pass_rate == 1.0
