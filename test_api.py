"""HTTP surface: contracts, limits and error handling."""

import pytest


def test_health_reports_a_version(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["version"].count(".") == 2


def test_meta_lists_everything_the_dashboard_needs(client):
    meta = client.get("/api/meta").json()
    assert {"solvers", "problems", "modulations", "channels", "version"} <= set(meta)
    assert any(s["adaptive"] for s in meta["solvers"])
    assert any(p["stiff"] for p in meta["problems"])
    assert all("bits_per_symbol" in m for m in meta["modulations"])


def test_index_and_static_assets_are_served(client):
    assert client.get("/").status_code == 200
    for asset in ("app.js", "plot.js", "style.css"):
        assert client.get(f"/static/{asset}").status_code == 200


def test_solve_returns_a_trajectory_and_an_accuracy_report(client):
    body = client.post("/api/ode/solve", json={
        "problem": "damped_oscillator", "solver": "rk45", "rtol": 1e-8,
    }).json()
    assert body["success"]
    assert len(body["t"]) == len(body["y"][0])
    assert body["accuracy"]["passes_001pct"] is True
    assert body["stats"]["n_rhs"] > 0


def test_solve_omits_accuracy_when_no_closed_form_exists(client):
    body = client.post("/api/ode/solve", json={
        "problem": "van_der_pol", "solver": "trbdf2", "t_end": 20.0,
    }).json()
    assert body["success"]
    assert body["accuracy"] is None
    assert body["exact"] is None


def test_solve_reports_divergence_without_raising(client):
    """An unstable configuration is a result, not a server error."""
    body = client.post("/api/ode/solve", json={
        "problem": "prothero_robinson", "solver": "forward_euler", "steps": 100,
    }).json()
    assert body["success"] is False
    assert body["message"]


def test_convergence_returns_one_study_per_solver(client):
    body = client.post("/api/ode/convergence", json={
        "problem": "exponential_decay",
        "solvers": ["heun", "rk4"],
        "counts": [50, 100, 200, 400],
    }).json()
    orders = {s["method"]: s["observed_order"] for s in body["studies"]}
    assert orders["heun"] == pytest.approx(2, rel=0.15)
    assert orders["rk4"] == pytest.approx(4, rel=0.15)


def test_convergence_rejects_adaptive_methods(client):
    """A refinement study needs a step size to refine."""
    r = client.post("/api/ode/convergence", json={
        "problem": "logistic", "solvers": ["rk45"]})
    assert r.status_code == 422


def test_convergence_rejects_problems_without_a_reference(client):
    r = client.post("/api/ode/convergence", json={
        "problem": "van_der_pol", "solvers": ["rk4"]})
    assert r.status_code == 422


def test_ber_sweep_returns_points_with_confidence_intervals(client):
    body = client.post("/api/phy/ber", json={
        "scheme": "qpsk", "ebn0_start": 0, "ebn0_stop": 6, "ebn0_step": 3,
        "target_errors": 200, "max_symbols": 200_000,
    }).json()
    points = body["sweep"]["points"]
    assert len(points) == 3
    for p in points:
        assert p["ci_low"] <= p["ber"] <= p["ci_high"]
        assert p["theory_within_ci"] is True
    assert len(body["theory_curve"]["ber"]) > len(points)


def test_ber_sweep_is_seeded_and_repeatable(client):
    payload = {"scheme": "bpsk", "ebn0_start": 4, "ebn0_stop": 4,
               "ebn0_step": 1, "target_errors": 100, "max_symbols": 100_000}
    a = client.post("/api/phy/ber", json=payload).json()
    b = client.post("/api/phy/ber", json=payload).json()
    assert a["sweep"]["points"][0]["ber"] == b["sweep"]["points"][0]["ber"]


def test_constellation_returns_ideal_and_received_points(client):
    body = client.post("/api/phy/constellation", json={
        "scheme": "qam16", "ebn0_db": 18.0, "n_symbols": 2000}).json()
    assert len(body["ideal"]) == 16
    assert len(body["received"]) == 2000
    assert body["evm_pct"] > 0


def test_pathloss_orders_the_models_as_physics_requires(client):
    body = client.post("/api/phy/pathloss", json={
        "frequency_hz": 2.4e9, "exponent": 3.0}).json()
    assert body["breakpoint_m"] > 0
    # An urban exponent of 3 must cost more than free space at long range.
    assert body["log_distance_db"][-1] > body["free_space_db"][-1]


def test_link_budget_closes_at_short_range_and_fails_at_long_range(client):
    near = client.post("/api/phy/link_budget", json={
        "distance_m": 20.0, "scheme": "qpsk", "target_ber": 1e-6}).json()
    far = client.post("/api/phy/link_budget", json={
        "distance_m": 20_000.0, "scheme": "qpsk", "target_ber": 1e-6}).json()
    assert near["closes"] is True
    assert far["closes"] is False
    assert far["path_loss_db"] > near["path_loss_db"]


def test_pulse_endpoint_reports_isi_and_papr(client):
    body = client.post("/api/phy/pulse", json={"beta": 0.35, "sps": 8}).json()
    assert body["isi_residual"] < 0.01
    assert body["papr_tx_db"] > 0


@pytest.mark.parametrize("path,payload", [
    ("/api/ode/solve", {"problem": "no_such_problem"}),
    ("/api/ode/solve", {"problem": "logistic", "solver": "no_such_solver"}),
])
def test_unknown_names_return_404(client, path, payload):
    assert client.post(path, json=payload).status_code == 404


@pytest.mark.parametrize("payload", [
    {"scheme": "qpsk", "ebn0_start": 0, "ebn0_stop": 60, "ebn0_step": 0.5},
    {"scheme": "qpsk", "max_symbols": 10**9},
    {"scheme": "not_a_scheme"},
])
def test_ber_requests_outside_the_limits_are_rejected(client, payload):
    assert client.post("/api/phy/ber", json=payload).status_code in (404, 422)


def test_step_count_is_capped(client):
    """The server must not let a browser request an unbounded integration."""
    r = client.post("/api/ode/solve", json={
        "problem": "logistic", "solver": "rk4", "steps": 10**9})
    assert r.status_code == 422
