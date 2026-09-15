"""End-to-end link simulation against the closed-form references.

This is the headline claim of the project, so the tolerances are stated in
units of the Monte Carlo standard error rather than as round numbers.
"""

import numpy as np
import pytest

from physim.phy.link import (
    ber_sweep,
    constellation_samples,
    simulate_ber_point,
    wilson_interval,
)
from physim.phy.theory import ber_qam_exact, theoretical_ber


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(50, 10_000)
    assert lo < 0.005 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_interval_handles_zero_errors():
    """A run with no errors still bounds the BER from above; a naive
    normal interval would collapse to zero width and claim certainty."""
    lo, hi = wilson_interval(0, 100_000)
    assert lo == 0.0
    assert 0 < hi < 1e-4


def test_wilson_interval_narrows_with_more_trials():
    _, hi_few = wilson_interval(10, 1_000)
    _, hi_many = wilson_interval(1_000, 100_000)
    assert hi_many < hi_few


@pytest.mark.parametrize("scheme,ebn0", [("bpsk", 6.0), ("qpsk", 6.0),
                                         ("qam16", 12.0), ("psk8", 12.0)])
def test_simulated_ber_is_within_three_sigma_of_theory(scheme, ebn0):
    """The deviation is judged against the binomial standard error of the
    run itself, so the test does not silently weaken as the run gets short."""
    rng = np.random.default_rng(101)
    point = simulate_ber_point(scheme, ebn0, rng=rng, target_errors=800,
                               max_symbols=1_500_000)
    p = point.theory_ber
    sigma = np.sqrt(p * (1 - p) / point.n_bits)
    assert abs(point.ber - p) < 3 * sigma
    assert point.theory_within_ci


@pytest.mark.parametrize("scheme", ["bpsk", "qpsk", "qam16"])
def test_symbol_error_rate_and_bit_error_rate_are_consistent(scheme):
    """Gray labelling means most symbol errors flip exactly one bit, so
    BER * log2(M) should sit just below SER at moderate SNR."""
    from physim.phy.modulation import get_modulation

    k = get_modulation(scheme).bits_per_symbol
    point = simulate_ber_point(scheme, 10.0, rng=np.random.default_rng(3),
                               target_errors=500, max_symbols=1_000_000)
    ratio = point.ber * k / point.ser
    assert 0.8 <= ratio <= 1.05


def test_sweep_is_bit_for_bit_reproducible():
    """A curve that cannot be regenerated is not a measurement."""
    a = ber_sweep("qpsk", [4.0, 6.0], seed=999, target_errors=200,
                  max_symbols=200_000)
    b = ber_sweep("qpsk", [4.0, 6.0], seed=999, target_errors=200,
                  max_symbols=200_000)
    assert [p.ber for p in a.points] == [p.ber for p in b.points]
    assert [p.n_bits for p in a.points] == [p.n_bits for p in b.points]


def test_different_seeds_give_different_realisations():
    a = ber_sweep("qpsk", [4.0], seed=1, target_errors=200, max_symbols=200_000)
    b = ber_sweep("qpsk", [4.0], seed=2, target_errors=200, max_symbols=200_000)
    assert a.points[0].n_bit_errors != b.points[0].n_bit_errors


def test_sweep_ber_decreases_with_snr():
    sweep = ber_sweep("qpsk", [0.0, 3.0, 6.0, 9.0], seed=42, target_errors=300,
                      max_symbols=500_000)
    assert np.all(np.diff(sweep.ber) < 0)


def test_worst_deviation_ignores_points_with_too_few_errors():
    """A point that stopped at three errors carries no information; letting
    it set the headline number would make the figure meaningless."""
    sweep = ber_sweep("qpsk", [2.0, 4.0, 14.0], seed=7, target_errors=200,
                      max_symbols=100_000)
    assert sweep.points[-1].n_bit_errors < 50          # ran out of symbols
    assert sweep.worst_relative_deviation(min_errors=50) < 0.25


def test_implementation_loss_is_small_for_an_ideal_receiver():
    """With perfect synchronisation the simulated curve should sit within a
    small fraction of a dB of theory."""
    sweep = ber_sweep("qpsk", [2.0, 4.0, 6.0, 8.0], seed=5, target_errors=500,
                      max_symbols=2_000_000)
    assert abs(sweep.implementation_loss_db()) < 0.15


def test_rayleigh_channel_is_markedly_worse_than_awgn():
    awgn = simulate_ber_point("qpsk", 12.0, channel="awgn",
                              rng=np.random.default_rng(1),
                              target_errors=200, max_symbols=2_000_000)
    fading = simulate_ber_point("qpsk", 12.0, channel="rayleigh",
                                rng=np.random.default_rng(1),
                                target_errors=200, max_symbols=2_000_000)
    assert fading.ber > 20 * awgn.ber


def test_rayleigh_matches_its_own_closed_form():
    point = simulate_ber_point("qpsk", 15.0, channel="rayleigh",
                               rng=np.random.default_rng(2),
                               target_errors=1000, max_symbols=3_000_000)
    expected = float(np.ravel(theoretical_ber(15.0, "qpsk", channel="rayleigh"))[0])
    assert point.ber == pytest.approx(expected, rel=0.15)


def test_rician_sits_between_rayleigh_and_awgn():
    kw = dict(target_errors=400, max_symbols=2_000_000)
    r = simulate_ber_point("qpsk", 12.0, channel="rayleigh",
                           rng=np.random.default_rng(4), **kw).ber
    k = simulate_ber_point("qpsk", 12.0, channel="rician", rician_k_db=8.0,
                           rng=np.random.default_rng(4), **kw).ber
    a = simulate_ber_point("qpsk", 12.0, channel="awgn",
                           rng=np.random.default_rng(4), **kw).ber
    assert a < k < r


def test_run_stops_once_the_error_target_is_met():
    """Effort follows the error count, not a fixed bit budget."""
    point = simulate_ber_point("bpsk", 0.0, rng=np.random.default_rng(6),
                               target_errors=200, max_symbols=5_000_000)
    assert point.n_bit_errors >= 200
    # Errors arrive in the first chunk at 0 dB, so the run stops there
    # instead of spending its five-million-symbol budget.
    assert point.n_bits <= 100_000


def test_unknown_channel_is_reported():
    with pytest.raises(ValueError, match="unknown channel"):
        simulate_ber_point("qpsk", 5.0, channel="underwater")


def test_constellation_samples_report_consistent_evm():
    """EVM and Eb/N0 are two views of the same noise power: for 16-QAM,
    Es/N0 = Eb/N0 + 10log10(4), and EVM should recover it."""
    out = constellation_samples("qam16", ebn0_db=18.0, n_symbols=20_000,
                                seed=9)
    assert out["evm_snr_db"] == pytest.approx(18.0 + 10 * np.log10(4), abs=0.3)
    assert len(out["ideal"]) == 16
    assert len(out["received"]) == 20_000


def test_high_order_scheme_tracks_theory_at_high_snr():
    point = simulate_ber_point("qam64", 18.0, rng=np.random.default_rng(8),
                               target_errors=600, max_symbols=4_000_000)
    p = float(np.ravel(ber_qam_exact(18.0, 64))[0])
    sigma = np.sqrt(p * (1 - p) / point.n_bits)
    assert abs(point.ber - p) < 3 * sigma
