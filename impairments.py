"""Stochastic impairments beyond thermal noise.

Real links are rarely AWGN-limited. Oscillator phase noise, a residual
carrier frequency offset, IQ imbalance in the analogue front end, finite ADC
resolution and impulsive interference each degrade the link in a distinct
way, and each has a closed-form signature that can be checked:

* Free-running phase noise is a Wiener process, so its variance grows
  linearly with time at rate ``2*pi*linewidth*Ts``.
* Uniform quantisation gives ``SQNR = 6.02 B + 1.76 dB`` for a full-scale
  sine, the standard result that makes ADC bit depth a link budget line item.
* A constant frequency offset rotates the constellation at a constant rate,
  which is why the EVM it causes grows without bound over a burst.

``tests/test_impairments.py`` checks all three against their closed forms.
"""

from __future__ import annotations

import numpy as np


def phase_noise(n_samples: int, linewidth_hz: float, sample_rate_hz: float,
                rng: np.random.Generator) -> np.ndarray:
    """Wiener (random-walk) phase noise from a free-running oscillator.

    Returns the phase in radians. The per-sample increment variance is
    ``2*pi*linewidth/fs``, which follows from the Lorentzian lineshape of a
    free-running source.
    """
    step_var = 2.0 * np.pi * linewidth_hz / sample_rate_hz
    increments = rng.normal(0.0, np.sqrt(step_var), n_samples)
    return np.cumsum(increments)


def apply_phase_noise(symbols: np.ndarray, linewidth_hz: float,
                      symbol_rate_hz: float,
                      rng: np.random.Generator) -> np.ndarray:
    phi = phase_noise(np.size(symbols), linewidth_hz, symbol_rate_hz, rng)
    return symbols * np.exp(1j * phi.reshape(np.shape(symbols)))


def apply_frequency_offset(symbols: np.ndarray, offset_hz: float,
                           symbol_rate_hz: float,
                           initial_phase: float = 0.0) -> np.ndarray:
    """Rotate by a constant carrier frequency offset."""
    n = np.arange(np.size(symbols))
    phase = 2.0 * np.pi * offset_hz * n / symbol_rate_hz + initial_phase
    return symbols * np.exp(1j * phase.reshape(np.shape(symbols)))


def apply_iq_imbalance(symbols: np.ndarray, gain_db: float = 0.5,
                       phase_deg: float = 3.0) -> np.ndarray:
    """Amplitude and quadrature-phase mismatch between the I and Q paths.

    Produces an image of the signal at the mirror frequency; the image
    rejection ratio it implies is what an analogue front end is specified on.
    """
    g = 10.0 ** (gain_db / 20.0)
    theta = np.deg2rad(phase_deg)
    # The I path takes the gain error; the phase error leaks I into Q.
    i_out = np.real(symbols) * g
    q_out = (np.imag(symbols) * np.cos(theta)
             + np.real(symbols) * np.sin(theta))
    return i_out + 1j * q_out


def image_rejection_db(gain_db: float, phase_deg: float) -> float:
    """Closed-form IRR for a given gain/phase mismatch."""
    g = 10.0 ** (gain_db / 20.0)
    theta = np.deg2rad(phase_deg)
    num = 1.0 + 2.0 * g * np.cos(theta) + g**2
    den = 1.0 - 2.0 * g * np.cos(theta) + g**2
    return float(10.0 * np.log10(num / den))


def quantize(x: np.ndarray, n_bits: int, full_scale: float | None = None
             ) -> np.ndarray:
    """Mid-rise uniform quantiser with clipping at ``+/- full_scale``."""
    x = np.asarray(x)
    if full_scale is None:
        full_scale = float(np.max(np.abs(np.concatenate(
            [np.real(x).ravel(), np.imag(x).ravel()]
        )))) or 1.0
    levels = 2 ** n_bits
    step = 2.0 * full_scale / levels

    def q(v):
        out = np.floor(v / step) * step + step / 2.0
        return np.clip(out, -full_scale + step / 2.0, full_scale - step / 2.0)

    if np.iscomplexobj(x):
        return q(np.real(x)) + 1j * q(np.imag(x))
    return q(x)


def sqnr_db(n_bits: int) -> float:
    """Ideal signal-to-quantisation-noise ratio for a full-scale sine."""
    return 6.02 * n_bits + 1.76


def impulsive_noise(shape, prob: float, power_ratio_db: float,
                    base_var: float, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli-Gaussian (Middleton Class A style) impulsive interference.

    A fraction ``prob`` of samples are hit by a burst that is
    ``power_ratio_db`` stronger than the background. Unlike AWGN the damage
    is concentrated: the same average noise power produces a very different
    BER depending on whether it arrives smoothly or in bursts, which is the
    argument for interleaving.
    """
    ratio = 10.0 ** (power_ratio_db / 10.0)
    hits = rng.random(shape) < prob
    sigma_bg = np.sqrt(base_var / 2.0)
    sigma_imp = np.sqrt(base_var * ratio / 2.0)
    sigma = np.where(hits, sigma_imp, sigma_bg)
    return (rng.normal(0.0, 1.0, shape) * sigma
            + 1j * rng.normal(0.0, 1.0, shape) * sigma)


def ou_process(n_samples: int, sigma: float, tau_samples: float,
               rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck process: Gaussian, stationary, exponentially
    correlated with time constant ``tau_samples``.

    The discrete update is exact for the continuous process (not an Euler
    approximation), so the stationary variance is ``sigma^2`` regardless of
    the sampling interval.
    """
    rho = float(np.exp(-1.0 / tau_samples))
    out = np.empty(n_samples)
    out[0] = rng.normal(0.0, sigma)
    innov = rng.normal(0.0, sigma * np.sqrt(1.0 - rho**2), n_samples)
    for i in range(1, n_samples):
        out[i] = rho * out[i - 1] + innov[i]
    return out


def evm_pct(received: np.ndarray, reference: np.ndarray) -> float:
    """Error vector magnitude as a percentage of the reference RMS."""
    err = np.asarray(received) - np.asarray(reference)
    return float(100.0 * np.sqrt(np.mean(np.abs(err) ** 2))
                 / np.sqrt(np.mean(np.abs(reference) ** 2)))


def evm_to_snr_db(evm_percent: float) -> float:
    """``SNR = -20 log10(EVM)``, the usual test-equipment conversion."""
    return float(-20.0 * np.log10(evm_percent / 100.0))
