"""Channel models: noise conventions, fading statistics, path loss."""

import numpy as np
import pytest
from scipy.special import j0

from physim.phy import channel as ch


def test_noise_variance_follows_the_ebn0_definition():
    """N0 = Es/(k * 10^(EbN0/10)) with Es = 1. A factor-of-two slip here is
    the single most common bug in a link simulator, so it is pinned."""
    assert ch.noise_variance(0.0, 1) == pytest.approx(1.0)
    assert ch.noise_variance(10.0, 1) == pytest.approx(0.1)
    assert ch.noise_variance(0.0, 4) == pytest.approx(0.25)


def test_esn0_and_ebn0_are_inverses():
    for k in (1, 2, 4, 6):
        assert ch.esn0_to_ebn0_db(ch.ebn0_to_esn0_db(7.5, k), k) == pytest.approx(7.5)


def test_complex_noise_splits_power_evenly_across_dimensions():
    """Each quadrature carries N0/2. Measured, not assumed."""
    rng = np.random.default_rng(1)
    n0 = 0.4
    z = ch.awgn(np.zeros(400_000, dtype=complex), n0, rng)
    assert np.var(z.real) == pytest.approx(n0 / 2, rel=0.02)
    assert np.var(z.imag) == pytest.approx(n0 / 2, rel=0.02)
    assert np.mean(np.abs(z) ** 2) == pytest.approx(n0, rel=0.02)


def test_real_noise_carries_the_full_variance():
    """For a one-dimensional constellation there is no quadrature to hide in."""
    rng = np.random.default_rng(2)
    z = ch.awgn(np.zeros(400_000), 0.4, rng)
    assert np.var(z) == pytest.approx(0.4, rel=0.02)


def test_rayleigh_coefficients_have_unit_average_gain():
    rng = np.random.default_rng(3)
    h = ch.rayleigh_coefficients(500_000, rng)
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.01)
    # Envelope follows Rayleigh: E|h| = sqrt(pi)/2 for unit power.
    assert np.mean(np.abs(h)) == pytest.approx(np.sqrt(np.pi) / 2, rel=0.01)


@pytest.mark.parametrize("k_db", [0.0, 6.0, 20.0])
def test_rician_power_is_normalised_and_k_factor_is_recovered(k_db):
    rng = np.random.default_rng(4)
    h = ch.rician_coefficients(400_000, k_db, rng)
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.02)
    # Specular power over scattered power must return the requested K.
    los = np.abs(np.mean(h)) ** 2
    scatter = np.mean(np.abs(h - np.mean(h)) ** 2)
    assert 10 * np.log10(los / scatter) == pytest.approx(k_db, abs=0.3)


def test_rician_at_high_k_approaches_an_unfaded_channel():
    rng = np.random.default_rng(5)
    h = ch.rician_coefficients(100_000, 30.0, rng)
    assert np.std(np.abs(h)) < 0.05


def test_jakes_autocorrelation_matches_the_bessel_model():
    """Sum-of-sinusoids fading must reproduce R(tau) = J0(2*pi*fd*tau);
    otherwise burst lengths, and any interleaving conclusion, are wrong."""
    rng = np.random.default_rng(6)
    fd_ts = 0.005
    h = ch.jakes_fading(200_000, fd_ts, rng)
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.05)
    lags = np.arange(0, 60, 10)
    for lag in lags:
        measured = np.mean(h[: h.size - lag] * np.conj(h[lag:])).real
        assert measured == pytest.approx(j0(2 * np.pi * fd_ts * lag), abs=0.06)


def test_free_space_loss_matches_the_friis_equation():
    """2.4 GHz at 1 m is 40.05 dB; the standard sanity value."""
    assert ch.free_space_path_loss_db(1.0, 2.4e9) == pytest.approx(40.05, abs=0.02)
    # Doubling the distance costs 6.02 dB in free space.
    delta = (ch.free_space_path_loss_db(200.0, 2.4e9)
             - ch.free_space_path_loss_db(100.0, 2.4e9))
    assert delta == pytest.approx(6.0206, abs=1e-3)


def test_log_distance_reduces_to_free_space_at_exponent_two():
    d = np.array([1.0, 10.0, 100.0, 1000.0])
    assert np.allclose(ch.log_distance_path_loss_db(d, 2.4e9, exponent=2.0),
                       ch.free_space_path_loss_db(d, 2.4e9))


def test_log_distance_slope_follows_the_exponent():
    for n in (2.0, 3.0, 4.0):
        delta = (ch.log_distance_path_loss_db(1000.0, 2.4e9, exponent=n)
                 - ch.log_distance_path_loss_db(100.0, 2.4e9, exponent=n))
        assert delta == pytest.approx(10 * n, abs=1e-6)


def test_two_ray_decays_as_fourth_power_past_the_breakpoint():
    """Beyond the breakpoint the ground reflection turns 20 dB/decade into
    40 dB/decade; this is the whole reason the model exists."""
    f, ht, hr = 2.4e9, 10.0, 1.5
    d_b = ch.breakpoint_distance_m(f, ht, hr)
    d1, d2 = 4 * d_b, 40 * d_b
    delta = (ch.two_ray_path_loss_db(d2, f, h_tx_m=ht, h_rx_m=hr)
             - ch.two_ray_path_loss_db(d1, f, h_tx_m=ht, h_rx_m=hr))
    assert delta == pytest.approx(40.0, abs=1.0)


def test_breakpoint_matches_four_pi_ht_hr_over_lambda():
    f, ht, hr = 2.4e9, 10.0, 1.5
    lam = ch.SPEED_OF_LIGHT / f
    assert ch.breakpoint_distance_m(f, ht, hr) == pytest.approx(
        4 * np.pi * ht * hr / lam, rel=1e-9)


def test_shadowing_has_the_requested_spread():
    rng = np.random.default_rng(7)
    s = ch.log_normal_shadowing_db(200_000, sigma_db=8.0, rng=rng)
    assert np.std(s) == pytest.approx(8.0, rel=0.02)
    assert np.mean(s) == pytest.approx(0.0, abs=0.1)


def test_correlated_shadowing_decorrelates_over_its_length_scale():
    """Gudmundson's model: correlation decays to 1/e at the decorrelation
    distance, so neighbouring positions are not independent draws."""
    rng = np.random.default_rng(8)
    s = ch.correlated_shadowing_db(200_000, sigma_db=8.0, decorrelation_m=50.0,
                                   step_m=1.0, rng=rng)
    assert np.std(s) == pytest.approx(8.0, rel=0.05)
    # One decorrelation distance of travel costs a factor of e in correlation.
    r = np.corrcoef(s[:-50], s[50:])[0, 1]
    assert r == pytest.approx(np.exp(-1.0), abs=0.08)


def test_thermal_noise_floor_is_minus_174_dbm_per_hz():
    budget = ch.link_budget(100.0, 2.4e9, bandwidth_hz=1.0, noise_figure_db=0.0)
    assert budget.noise_power_dbm == pytest.approx(-173.98, abs=0.02)


def test_link_budget_margin_is_snr_minus_requirement():
    b = ch.link_budget(500.0, 2.4e9, tx_power_dbm=20.0, required_snr_db=12.0)
    assert b.margin_db == pytest.approx(b.snr_db - 12.0)
    assert b.as_dict()["closes"] is (b.margin_db >= 0)


def test_link_budget_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown path loss model"):
        ch.link_budget(100.0, 2.4e9, model="ray_tracing")
