"""Pulse shaping: the Nyquist criterion, bandwidth and peak-to-average power."""

import numpy as np
import pytest

from physim.phy import pulse as ps


@pytest.mark.parametrize("beta", [0.0, 0.22, 0.35, 0.5, 1.0])
def test_raised_cosine_has_zero_intersymbol_interference(beta):
    """A raised-cosine pulse must vanish at every non-zero symbol instant.
    This is the defining property, so it is checked directly on the taps."""
    sps = 8
    taps = ps.rc_taps(beta, span_symbols=12, sps=sps)
    centre = taps.size // 2
    offsets = np.arange(-12, 13) * sps
    offsets = offsets[(offsets != 0) & (np.abs(offsets) <= centre)]
    assert np.max(np.abs(taps[centre + offsets])) < 1e-9 * taps[centre]


@pytest.mark.parametrize("beta", [0.22, 0.35, 0.5])
def test_matched_root_raised_cosine_pair_is_nyquist(beta):
    """Two RRC filters in cascade give a raised cosine; residual ISI is a
    truncation artefact and must stay small at a realistic span."""
    assert ps.nyquist_isi_residual(beta, span_symbols=10, sps=8) < 0.01
    assert ps.nyquist_isi_residual(beta, span_symbols=20, sps=8) < 0.002


def test_longer_filters_reduce_truncation_isi():
    short = ps.nyquist_isi_residual(0.35, span_symbols=4, sps=8)
    long = ps.nyquist_isi_residual(0.35, span_symbols=16, sps=8)
    assert long < short / 5


def test_rrc_taps_are_symmetric_and_unit_energy():
    taps = ps.rrc_taps(0.35, span_symbols=10, sps=8)
    assert taps.size % 2 == 1
    assert np.allclose(taps, taps[::-1], atol=1e-12)
    assert np.sum(taps ** 2) == pytest.approx(1.0, rel=1e-9)


def test_rrc_handles_the_removable_singularities():
    """beta != 0 puts poles at t = 0 and t = +-T/(4*beta); the closed-form
    limits must be used there instead of a divide by zero."""
    for beta in (0.25, 0.5):
        taps = ps.rrc_taps(beta, span_symbols=10, sps=4)
        assert np.all(np.isfinite(taps))


def test_upsampling_inserts_the_right_number_of_zeros():
    x = np.array([1.0, -1.0, 1.0])
    up = ps.upsample(x, 4)
    assert up.size == 12
    assert np.array_equal(up[::4], x)
    assert np.count_nonzero(up) == 3


def test_shaped_link_recovers_its_symbols_without_noise():
    out = ps.shaped_link_demo("qpsk", beta=0.35, sps=8, n_symbols=2000,
                              ebn0_db=None, seed=3)
    assert out["bit_errors"] == 0


def test_shaping_reduces_occupied_bandwidth_as_rolloff_falls():
    narrow = ps.shaped_link_demo("qpsk", beta=0.15, sps=8, n_symbols=4000, seed=1)
    wide = ps.shaped_link_demo("qpsk", beta=0.9, sps=8, n_symbols=4000, seed=1)
    assert narrow["occupied_bw_hz"] < wide["occupied_bw_hz"]


def test_occupied_bandwidth_brackets_the_nyquist_minimum():
    """For symbol rate Rs and rolloff beta the 99% bandwidth sits near
    Rs*(1+beta); a pulse shaper that claims less is not band-limiting."""
    out = ps.shaped_link_demo("qpsk", beta=0.35, sps=8, n_symbols=8000, seed=2)
    rs = out["symbol_rate_hz"]
    assert 0.9 * rs <= out["occupied_bw_hz"] <= 1.6 * rs


def test_papr_rises_when_the_pulse_is_shaped():
    """An unshaped constant-modulus stream has 0 dB PAPR; shaping creates
    peaks, which is what drives power-amplifier backoff."""
    out = ps.shaped_link_demo("qpsk", beta=0.35, sps=8, n_symbols=4000, seed=4)
    assert out["papr_tx_db"] > 2.0
    assert ps.papr_db(np.exp(1j * np.linspace(0, 10, 1000))) == pytest.approx(
        0.0, abs=1e-9)


def test_eye_diagram_traces_are_aligned_and_open():
    """Every trace must span the same two symbols, and at the sampling
    instant the traces must separate into the transmitted levels: that
    separation is the eye opening."""
    out = ps.shaped_link_demo("qpsk", beta=0.35, sps=8, n_symbols=1000,
                              ebn0_db=None, seed=5, return_waveforms=True)
    eye = ps.eye_diagram(out["rx_waveform"].real, sps=8, n_traces=50,
                         span_symbols=2)
    traces = np.array(eye["traces"])
    assert traces.shape == (50, 16)
    assert len(eye["t"]) == 16

    centre = traces[:, 8]                      # one symbol in, a sampling instant
    assert np.min(np.abs(centre)) > 0.5 * np.max(np.abs(centre))
