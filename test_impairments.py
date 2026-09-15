"""Front-end impairments: phase noise, offsets, imbalance and quantisation."""

import numpy as np
import pytest

from physim.phy import impairments as imp
from physim.phy.modulation import get_modulation


def test_phase_noise_increment_variance_matches_the_lorentzian_model():
    """A free-running oscillator is a Wiener process whose per-sample
    increment variance is 2*pi*linewidth/fs. Estimated from the increments,
    where there are a million samples, rather than from the endpoint spread,
    where there are only as many samples as there are runs."""
    rng = np.random.default_rng(1)
    fs, lw = 1e6, 1e3
    phi = imp.phase_noise(1_000_000, lw, fs, rng)
    assert np.var(np.diff(phi)) == pytest.approx(2 * np.pi * lw / fs, rel=0.01)


def test_phase_noise_variance_grows_linearly_with_time():
    """var(phi(t)) = 2*pi*linewidth*t: doubling the elapsed time must double
    the spread. Tested as a ratio, which is insensitive to the estimator's
    own noise in a way an absolute comparison is not."""
    rng = np.random.default_rng(1)
    fs, lw, n = 1e6, 1e3, 4000
    runs = np.array([imp.phase_noise(n, lw, fs, rng) for _ in range(2000)])
    variances = np.array([np.var(runs[:, k - 1]) for k in (500, 1000, 2000, 4000)])
    assert np.all(np.diff(variances) > 0)
    for ratio in variances[1:] / variances[:-1]:
        assert ratio == pytest.approx(2.0, rel=0.12)


def test_phase_noise_increments_are_independent():
    rng = np.random.default_rng(2)
    phi = imp.phase_noise(200_000, 1e4, 1e6, rng)
    d = np.diff(phi)
    assert abs(np.corrcoef(d[:-1], d[1:])[0, 1]) < 0.02


def test_phase_noise_preserves_amplitude():
    rng = np.random.default_rng(3)
    s = get_modulation("qam16").modulate(
        get_modulation("qam16").random_bits(1000, rng))
    out = imp.apply_phase_noise(s, 1e3, 1e6, rng)
    assert np.allclose(np.abs(out), np.abs(s))


def test_frequency_offset_rotates_at_the_requested_rate():
    s = np.ones(1000, dtype=complex)
    out = imp.apply_frequency_offset(s, offset_hz=100.0, symbol_rate_hz=10_000.0)
    # One full turn every 100 samples.
    assert np.angle(out[100]) == pytest.approx(0.0, abs=1e-9)
    assert np.angle(out[25]) == pytest.approx(np.pi / 2, abs=1e-9)


def test_frequency_offset_is_undone_by_its_inverse():
    rng = np.random.default_rng(4)
    s = rng.normal(size=500) + 1j * rng.normal(size=500)
    out = imp.apply_frequency_offset(
        imp.apply_frequency_offset(s, 250.0, 1e5), -250.0, 1e5)
    assert np.allclose(out, s, atol=1e-12)


def test_iq_imbalance_creates_a_conjugate_image():
    """Gain and phase mismatch leak a mirrored copy of the signal; the
    leakage level must match the standard image-rejection formula."""
    n = 200_000
    s = np.exp(1j * 2 * np.pi * 0.1 * np.arange(n))
    out = imp.apply_iq_imbalance(s, gain_db=1.0, phase_deg=5.0)
    spectrum = np.abs(np.fft.fft(out)) ** 2
    wanted = spectrum[int(0.1 * n)]
    image = spectrum[int(0.9 * n)]
    measured_db = 10 * np.log10(wanted / image)
    assert measured_db == pytest.approx(imp.image_rejection_db(1.0, 5.0), abs=0.5)


def test_perfect_iq_path_is_transparent():
    rng = np.random.default_rng(6)
    s = rng.normal(size=100) + 1j * rng.normal(size=100)
    assert np.allclose(imp.apply_iq_imbalance(s, 0.0, 0.0), s, atol=1e-12)


@pytest.mark.parametrize("n_bits", [6, 8, 10, 12])
def test_quantisation_noise_matches_six_db_per_bit_for_a_sine(n_bits):
    """SQNR = 6.02B + 1.76 dB is the *sinusoidal* full-scale figure, which is
    what a converter datasheet quotes, so it is measured with a sine. The
    formula assumes quantisation error is uniform and uncorrelated with the
    signal, which stops holding at very low resolutions, so the check starts
    at six bits."""
    t = np.linspace(0, 997 * 2 * np.pi, 400_000, endpoint=False)
    x = np.sin(t)
    err = imp.quantize(x, n_bits, full_scale=1.0) - x
    measured = 10 * np.log10(np.var(x) / np.var(err))
    assert measured == pytest.approx(imp.sqnr_db(n_bits), abs=0.5)


@pytest.mark.parametrize("n_bits", [6, 10])
def test_uniform_input_loses_the_crest_factor_term(n_bits):
    """The same quantiser fed a uniform full-scale signal gives 6.02B with no
    1.76 dB bonus: that term comes from a sine's crest factor, not from the
    converter. Keeping both cases separate stops the constant being treated
    as universal."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=400_000)
    err = imp.quantize(x, n_bits, full_scale=1.0) - x
    measured = 10 * np.log10(np.var(x) / np.var(err))
    assert measured == pytest.approx(6.02 * n_bits, abs=0.3)


def test_quantiser_uses_the_full_code_range():
    x = np.linspace(-1, 1, 10_000)
    q = imp.quantize(x, 4, full_scale=1.0)
    assert np.unique(q).size <= 16
    assert np.max(np.abs(q)) <= 1.0 + 1e-12


def test_quantiser_clips_beyond_full_scale():
    q = imp.quantize(np.array([5.0, -5.0]), 8, full_scale=1.0)
    assert np.all(np.abs(q) <= 1.0 + 1e-12)


def test_impulsive_noise_is_rare_and_strong():
    """Middleton class-A style bursts: a small fraction of samples carry
    most of the disturbance energy."""
    rng = np.random.default_rng(8)
    z = imp.impulsive_noise(500_000, prob=0.01, power_ratio_db=30.0,
                            base_var=0.01, rng=rng)
    hit = np.abs(z) ** 2 > 10 * 0.01
    assert 0.005 < hit.mean() < 0.02
    assert np.sum(np.abs(z[hit]) ** 2) > np.sum(np.abs(z[~hit]) ** 2)


def test_ornstein_uhlenbeck_has_the_requested_spread_and_memory():
    rng = np.random.default_rng(9)
    x = imp.ou_process(400_000, sigma=2.0, tau_samples=100.0, rng=rng)
    assert np.std(x) == pytest.approx(2.0, rel=0.08)
    assert np.corrcoef(x[:-100], x[100:])[0, 1] == pytest.approx(
        np.exp(-1.0), abs=0.08)


def test_evm_is_zero_without_impairment_and_grows_with_noise():
    rng = np.random.default_rng(10)
    mod = get_modulation("qam16")
    s = mod.modulate(mod.random_bits(5000, rng))
    assert imp.evm_pct(s, s) == pytest.approx(0.0, abs=1e-12)
    noisy = s + 0.05 * (rng.normal(size=s.size) + 1j * rng.normal(size=s.size))
    assert imp.evm_pct(noisy, s) > 5.0


def test_evm_and_snr_are_inverse_views_of_the_same_quantity():
    """EVM = 1/sqrt(SNR) for a unit-energy constellation."""
    rng = np.random.default_rng(11)
    mod = get_modulation("qpsk")
    s = mod.modulate(mod.random_bits(200_000, rng))
    snr_db = 20.0
    sigma2 = 10 ** (-snr_db / 10)
    noisy = s + np.sqrt(sigma2 / 2) * (rng.normal(size=s.size)
                                       + 1j * rng.normal(size=s.size))
    assert imp.evm_to_snr_db(imp.evm_pct(noisy, s)) == pytest.approx(
        snr_db, abs=0.2)
