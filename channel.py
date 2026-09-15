"""Propagation and noise models.

Everything here is vectorised over an arbitrary leading shape, so a Monte
Carlo run can generate ``(n_trials, n_symbols)`` realisations in one call
rather than looping in Python.

Conventions
-----------
* Complex baseband throughout. A "symbol" is one complex sample.
* ``E[|s|^2] = 1`` for every constellation, so ``Es/N0 = k * Eb/N0`` with
  ``k = log2(M)``.
* Complex AWGN has total variance ``N0`` split evenly between the in-phase
  and quadrature components, i.e. ``N0/2`` per real dimension. Getting this
  factor wrong is the classic 3 dB simulation bug, so the convention is
  asserted directly in ``tests/test_channel.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
BOLTZMANN = 1.380649e-23


# ------------------------------------------------------------------- SNR maths
def ebn0_to_esn0_db(ebn0_db: float | np.ndarray, bits_per_symbol: int
                    ) -> np.ndarray:
    """``Es/N0 [dB] = Eb/N0 [dB] + 10 log10(k)``."""
    return np.asarray(ebn0_db, dtype=float) + 10.0 * np.log10(bits_per_symbol)


def esn0_to_ebn0_db(esn0_db: float | np.ndarray, bits_per_symbol: int
                    ) -> np.ndarray:
    return np.asarray(esn0_db, dtype=float) - 10.0 * np.log10(bits_per_symbol)


def noise_variance(ebn0_db: float | np.ndarray, bits_per_symbol: int,
                   es: float = 1.0) -> np.ndarray:
    """Total complex noise variance ``N0`` for a target Eb/N0."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return es / (bits_per_symbol * ebn0)


# ----------------------------------------------------------------------- noise
def awgn(symbols: np.ndarray, noise_var: float,
         rng: np.random.Generator) -> np.ndarray:
    """Add circularly symmetric complex Gaussian noise of variance ``noise_var``."""
    sigma = np.sqrt(noise_var / 2.0)
    shape = np.shape(symbols)
    noise = rng.normal(0.0, sigma, shape) + 1j * rng.normal(0.0, sigma, shape)
    return symbols + noise


def awgn_at_ebn0(symbols: np.ndarray, ebn0_db: float, bits_per_symbol: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, float]:
    nv = float(noise_variance(ebn0_db, bits_per_symbol))
    return awgn(symbols, nv, rng), nv


# ---------------------------------------------------------------------- fading
def rayleigh_coefficients(shape, rng: np.random.Generator) -> np.ndarray:
    """Flat Rayleigh fading with ``E[|h|^2] = 1`` (unit average power)."""
    return (rng.normal(0.0, np.sqrt(0.5), shape)
            + 1j * rng.normal(0.0, np.sqrt(0.5), shape))


def rician_coefficients(shape, k_db: float, rng: np.random.Generator
                        ) -> np.ndarray:
    """Rician fading with K-factor ``k_db``; ``E[|h|^2] = 1``.

    ``K`` is the power ratio of the deterministic line-of-sight component to
    the diffuse component. ``K -> 0`` recovers Rayleigh, ``K -> inf`` a pure
    AWGN channel, which makes it the natural knob for sweeping between the
    two in a link study.
    """
    k = 10.0 ** (k_db / 10.0)
    los = np.sqrt(k / (k + 1.0))
    diffuse = np.sqrt(1.0 / (k + 1.0))
    scattered = (rng.normal(0.0, np.sqrt(0.5), shape)
                 + 1j * rng.normal(0.0, np.sqrt(0.5), shape))
    return los + diffuse * scattered


def jakes_fading(n_samples: int, fd_ts: float, rng: np.random.Generator,
                 n_sinusoids: int = 32) -> np.ndarray:
    """Time-correlated Rayleigh fading by the sum-of-sinusoids method.

    ``fd_ts`` is the maximum Doppler shift normalised by the sample rate.
    Independent draws per symbol (``rayleigh_coefficients``) model a channel
    that changes infinitely fast; a real link's fade duration is what decides
    whether an interleaver helps, and that requires the correct autocorrelation

        ``R(tau) = J0(2*pi*f_d*tau)``

    which this generator reproduces (verified in ``tests/test_channel.py``).
    """
    n = np.arange(n_samples)
    m = np.arange(1, n_sinusoids + 1)
    # Clarke's model: uniformly distributed angles of arrival, randomised per
    # realisation so that repeated calls are statistically independent.
    alpha = (2.0 * np.pi * m - np.pi + rng.uniform(-np.pi, np.pi)) / (4.0 * n_sinusoids)
    phi = rng.uniform(-np.pi, np.pi, n_sinusoids)
    psi = rng.uniform(-np.pi, np.pi, n_sinusoids)

    arg = 2.0 * np.pi * fd_ts * np.cos(alpha)[:, None] * n[None, :]
    real = np.cos(arg + phi[:, None]).sum(axis=0)
    imag = np.cos(arg + psi[:, None]).sum(axis=0)
    h = (real + 1j * imag) / np.sqrt(n_sinusoids)
    return h / np.sqrt(np.mean(np.abs(h) ** 2))


# ------------------------------------------------------------------- path loss
def free_space_path_loss_db(distance_m: np.ndarray | float,
                            frequency_hz: float) -> np.ndarray:
    """Friis free-space loss. Exact only in the far field, ``d >> lambda``."""
    d = np.maximum(np.asarray(distance_m, dtype=float), 1e-6)
    lam = SPEED_OF_LIGHT / frequency_hz
    return 20.0 * np.log10(4.0 * np.pi * d / lam)


def log_distance_path_loss_db(distance_m: np.ndarray | float,
                              frequency_hz: float, exponent: float = 3.0,
                              d0_m: float = 1.0) -> np.ndarray:
    """Log-distance model: free-space to ``d0``, then a fitted exponent.

    ``exponent = 2`` is free space; 2.7-3.5 is typical urban; 4-6 indoor
    with obstructions. This is the model behind most coverage predictions,
    and the exponent is the single parameter that matters most.
    """
    d = np.maximum(np.asarray(distance_m, dtype=float), d0_m)
    pl0 = free_space_path_loss_db(d0_m, frequency_hz)
    return pl0 + 10.0 * exponent * np.log10(d / d0_m)


def two_ray_path_loss_db(distance_m: np.ndarray | float, frequency_hz: float,
                         h_tx_m: float = 10.0, h_rx_m: float = 1.5,
                         reflection_coeff: float = -1.0) -> np.ndarray:
    """Two-ray ground reflection with explicit phase interference.

    Summing the direct and reflected rays coherently reproduces the
    near-field interference nulls and the transition to a fourth-power decay
    beyond the breakpoint distance ``4*pi*h_tx*h_rx/lambda`` -- structure
    that a pure log-distance fit smooths away entirely.
    """
    d = np.maximum(np.asarray(distance_m, dtype=float), 1e-6)
    lam = SPEED_OF_LIGHT / frequency_hz
    d_los = np.sqrt(d**2 + (h_tx_m - h_rx_m) ** 2)
    d_ref = np.sqrt(d**2 + (h_tx_m + h_rx_m) ** 2)
    phase = 2.0 * np.pi * (d_ref - d_los) / lam
    e_field = (1.0 / d_los) + reflection_coeff * np.exp(-1j * phase) / d_ref
    gain = (lam / (4.0 * np.pi)) * np.abs(e_field)
    return -20.0 * np.log10(np.maximum(gain, 1e-30))


def breakpoint_distance_m(frequency_hz: float, h_tx_m: float, h_rx_m: float
                          ) -> float:
    """Distance beyond which two-ray loss rolls off as ``d^-4``."""
    lam = SPEED_OF_LIGHT / frequency_hz
    return float(4.0 * np.pi * h_tx_m * h_rx_m / lam)


def log_normal_shadowing_db(shape, sigma_db: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Uncorrelated log-normal shadowing, 4-12 dB standard deviation typical."""
    return rng.normal(0.0, sigma_db, shape)


def correlated_shadowing_db(n_samples: int, sigma_db: float,
                            decorrelation_m: float, step_m: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Gudmundson shadowing: an Ornstein-Uhlenbeck process along a route.

    Shadowing from a given building persists while the receiver stays behind
    it, so successive samples along a track are correlated with
    ``R(d) = sigma^2 * exp(-d / d_corr)``. Drawing independent samples instead
    makes outage look far less bursty than it is.
    """
    rho = float(np.exp(-step_m / decorrelation_m))
    innovation = sigma_db * np.sqrt(1.0 - rho**2)
    out = np.empty(n_samples)
    out[0] = rng.normal(0.0, sigma_db)
    noise = rng.normal(0.0, innovation, n_samples)
    for i in range(1, n_samples):
        out[i] = rho * out[i - 1] + noise[i]
    return out


# ----------------------------------------------------------------- link budget
@dataclass
class LinkBudget:
    """Received power and margin for a point-to-point link."""

    tx_power_dbm: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    path_loss_db: float
    other_losses_db: float
    bandwidth_hz: float
    noise_figure_db: float
    temperature_k: float
    required_snr_db: float

    @property
    def rx_power_dbm(self) -> float:
        return float(self.tx_power_dbm + self.tx_gain_dbi + self.rx_gain_dbi
                     - self.path_loss_db - self.other_losses_db)

    @property
    def noise_power_dbm(self) -> float:
        """Thermal floor ``kTB`` plus the receiver noise figure."""
        thermal_w = BOLTZMANN * self.temperature_k * self.bandwidth_hz
        return float(10.0 * np.log10(thermal_w * 1e3) + self.noise_figure_db)

    @property
    def snr_db(self) -> float:
        return float(self.rx_power_dbm - self.noise_power_dbm)

    @property
    def margin_db(self) -> float:
        return float(self.snr_db - self.required_snr_db)

    def as_dict(self) -> dict:
        # Plain Python scalars: numpy types do not survive JSON serialisation.
        return {
            "rx_power_dbm": self.rx_power_dbm,
            "noise_power_dbm": self.noise_power_dbm,
            "snr_db": self.snr_db,
            "margin_db": self.margin_db,
            "closes": bool(self.margin_db >= 0.0),
        }


def link_budget(distance_m: float, frequency_hz: float,
                tx_power_dbm: float = 20.0, tx_gain_dbi: float = 2.0,
                rx_gain_dbi: float = 2.0, bandwidth_hz: float = 20e6,
                noise_figure_db: float = 6.0, temperature_k: float = 290.0,
                required_snr_db: float = 10.0, other_losses_db: float = 3.0,
                model: str = "log_distance", exponent: float = 3.0,
                **model_kwargs) -> LinkBudget:
    """Assemble a link budget using one of the path-loss models above."""
    if model == "free_space":
        pl = float(free_space_path_loss_db(distance_m, frequency_hz))
    elif model == "two_ray":
        pl = float(two_ray_path_loss_db(distance_m, frequency_hz, **model_kwargs))
    elif model == "log_distance":
        pl = float(log_distance_path_loss_db(distance_m, frequency_hz,
                                             exponent=exponent, **model_kwargs))
    else:
        raise ValueError(f"unknown path loss model '{model}'")
    return LinkBudget(
        tx_power_dbm=tx_power_dbm, tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=rx_gain_dbi, path_loss_db=pl,
        other_losses_db=other_losses_db, bandwidth_hz=bandwidth_hz,
        noise_figure_db=noise_figure_db, temperature_k=temperature_k,
        required_snr_db=required_snr_db,
    )
