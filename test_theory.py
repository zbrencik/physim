"""Closed-form references.

These are the yardsticks the Monte Carlo results are measured against, so
they are themselves checked against independent derivations and known values.
"""

import numpy as np
import pytest

from physim.phy import theory as th


def test_q_function_at_known_points():
    assert th.qfunc(0.0) == pytest.approx(0.5)
    assert th.qfunc(1.0) == pytest.approx(0.158655, rel=1e-5)
    assert th.qfunc(2.0) == pytest.approx(0.0227501, rel=1e-5)
    assert th.qfunc(6.0) == pytest.approx(9.8659e-10, rel=1e-4)


def test_q_function_inverse_round_trips():
    x = np.array([0.5, 1.0, 3.0, 5.0, 7.0])
    assert np.allclose(th.qfunc_inv(th.qfunc(x)), x, rtol=1e-9)


def test_qpsk_and_bpsk_share_a_ber_curve():
    """Gray-coded QPSK is two orthogonal BPSK channels, so per-bit
    performance is identical even though the symbol rate differs."""
    ebn0 = np.linspace(0, 12, 25)
    assert np.allclose(th.ber_bpsk(ebn0), th.ber_psk(ebn0, 4), rtol=1e-12)


def test_exact_qam_formula_reduces_to_bpsk_for_m_equals_four():
    """The per-dimension PAM derivation must collapse to Q(sqrt(2 Eb/N0))
    at M = 4; agreement to machine precision means the general machinery
    is not quietly wrong."""
    ebn0 = np.linspace(0, 14, 29)
    assert np.allclose(th.ber_qam_exact(ebn0, 4), th.ber_bpsk(ebn0), rtol=1e-12)


@pytest.mark.parametrize("M", [4, 16, 64, 256])
def test_ber_is_monotone_in_snr(M):
    ebn0 = np.linspace(0, 25, 60)
    ber = th.ber_qam_exact(ebn0, M)
    assert np.all(np.diff(ber) < 0)
    assert np.all((ber > 0) & (ber < 0.5))


def _gap(ebn0, M):
    exact = float(np.ravel(th.ber_qam_exact(ebn0, M))[0])
    approx = float(np.ravel(th.ber_qam_nearest_neighbour(ebn0, M))[0])
    return abs(approx / exact - 1.0)


@pytest.mark.parametrize("M", [64, 256])
def test_nearest_neighbour_approximation_tightens_at_high_snr(M):
    """The textbook approximation counts only nearest-neighbour errors, so it
    is optimistic when the noise is large enough to reach past them. The gap
    must close as SNR rises. Quantifying it is why both are implemented."""
    assert _gap(4.0, M) > 1e-3
    assert _gap(16.0, M) < 1e-5


@pytest.mark.parametrize("M", [16, 64, 256])
def test_the_two_qam_formulas_agree_across_the_operating_range(M):
    """Anywhere a real link lives, the two derivations must give the same
    answer; a disagreement would mean one of them is wrong."""
    for ebn0 in np.linspace(8, 24, 17):
        exact = float(np.ravel(th.ber_qam_exact(ebn0, M))[0])
        # Only where a link would actually be operated: at a BER of 0.1 the
        # noise routinely reaches past the nearest neighbours, and the
        # approximation is not meant to hold there.
        if not (th.QAM_EXACT_PRECISION_FLOOR < exact < 1e-2):
            continue
        approx = float(np.ravel(th.ber_qam_nearest_neighbour(ebn0, M))[0])
        assert approx == pytest.approx(exact, rel=1e-3)


def test_exact_formula_documents_where_it_stops_being_trustworthy():
    """Past the floor the exact sum is cancellation noise, and the two
    formulas visibly diverge. The constant exists so that is a documented
    limit rather than a surprise."""
    assert _gap(28.0, 16) > 0.5
    assert float(np.ravel(th.ber_qam_exact(28.0, 16))[0]) < th.QAM_EXACT_PRECISION_FLOOR


def test_larger_constellations_need_more_energy_per_bit():
    ebn0 = 14.0
    assert (th.ber_qam_exact(ebn0, 4) < th.ber_qam_exact(ebn0, 16)
            < th.ber_qam_exact(ebn0, 64) < th.ber_qam_exact(ebn0, 256))


@pytest.mark.parametrize("M,expected", [(4, 10.5), (16, 14.4), (64, 18.8)])
def test_required_ebn0_matches_published_values(M, expected):
    """Textbook Eb/N0 for 1e-6 BER on Gray-coded square QAM."""
    assert th.required_ebn0_db(1e-6, M, "qam") == pytest.approx(expected, abs=0.2)


def test_required_ebn0_inverts_the_ber_curve():
    for M in (4, 16, 64):
        x = th.required_ebn0_db(1e-5, M, "qam")
        assert th.ber_qam_exact(x, M) == pytest.approx(1e-5, rel=1e-3)


def test_rayleigh_fading_costs_an_order_of_magnitude_in_snr():
    """Without diversity, fading turns exponential decay into 1/SNR decay."""
    assert th.ber_rayleigh(20.0, 4) > th.ber_psk(20.0, 4) * 100


def test_rayleigh_diversity_order_is_one():
    """A single-antenna Rayleigh link has slope -1 on a log-log BER curve."""
    ebn0 = np.linspace(25, 45, 21)
    ber = th.ber_rayleigh(ebn0, 4)
    assert th.diversity_order_rayleigh(ebn0, ber) == pytest.approx(1.0, abs=0.05)


def test_capacity_matches_shannon():
    assert th.awgn_capacity(0.0) == pytest.approx(1.0)
    assert th.awgn_capacity(10 * np.log10(3.0)) == pytest.approx(2.0)


def test_dispatcher_selects_the_right_family():
    ebn0 = np.linspace(0, 10, 11)
    assert np.allclose(th.theoretical_ber(ebn0, "bpsk"), th.ber_bpsk(ebn0))
    assert np.allclose(th.theoretical_ber(ebn0, "qam16"), th.ber_qam_exact(ebn0, 16))
    assert np.allclose(th.theoretical_ber(ebn0, "psk8"), th.ber_psk(ebn0, 8))
