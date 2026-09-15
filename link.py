"""End-to-end Monte Carlo link simulation.

The loop is chunked rather than fully vectorised over the whole run: a
10^8-bit sweep does not fit in memory as one array, but 10^6-symbol chunks
keep every inner operation in numpy while bounding peak memory to a few tens
of megabytes.

Two details make the results trustworthy rather than merely plausible:

* **Early stopping on error count, not sample count.** A BER estimate is a
  binomial proportion; its relative precision depends on the number of
  *errors* observed, not the number of bits sent. Stopping at a fixed 200
  errors gives roughly +/-7% relative precision at every point on the curve,
  so the high-SNR tail is as trustworthy as the low-SNR head.
* **Wilson score intervals.** The normal approximation to a binomial
  interval breaks down exactly where BER simulations live -- tiny ``p``,
  huge ``n`` -- and can produce negative lower bounds. The Wilson interval
  stays inside ``[0, 1]`` and remains valid for small error counts.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from physim.phy import channel as ch
from physim.phy.modulation import Modulation, get_modulation
from physim.phy.theory import theoretical_ber

CHANNELS = ("awgn", "rayleigh", "rician")


@dataclass
class BERPoint:
    ebn0_db: float
    ber: float
    ser: float
    n_bits: int
    n_bit_errors: int
    n_symbols: int
    n_symbol_errors: int
    theory_ber: float
    ci_low: float
    ci_high: float
    elapsed_s: float

    @property
    def rel_deviation(self) -> float:
        """Simulated/theoretical - 1, the headline agreement figure."""
        if self.theory_ber <= 0:
            return float("nan")
        return self.ber / self.theory_ber - 1.0

    @property
    def theory_within_ci(self) -> bool:
        return self.ci_low <= self.theory_ber <= self.ci_high

    def as_dict(self) -> dict[str, Any]:
        return {
            "ebn0_db": self.ebn0_db, "ber": self.ber, "ser": self.ser,
            "n_bits": self.n_bits, "n_bit_errors": self.n_bit_errors,
            "n_symbols": self.n_symbols,
            "n_symbol_errors": self.n_symbol_errors,
            "theory_ber": self.theory_ber,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "rel_deviation": self.rel_deviation,
            "theory_within_ci": self.theory_within_ci,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class BERSweep:
    scheme: str
    channel: str
    points: list[BERPoint] = field(default_factory=list)
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def ebn0_db(self) -> np.ndarray:
        return np.array([p.ebn0_db for p in self.points])

    @property
    def ber(self) -> np.ndarray:
        return np.array([p.ber for p in self.points])

    @property
    def theory(self) -> np.ndarray:
        return np.array([p.theory_ber for p in self.points])

    def worst_relative_deviation(self, min_errors: int = 50) -> float:
        """Largest sim/theory gap among statistically meaningful points."""
        devs = [abs(p.rel_deviation) for p in self.points
                if p.n_bit_errors >= min_errors and p.theory_ber > 0]
        return float(max(devs)) if devs else float("nan")

    def implementation_loss_db(self) -> float:
        """Horizontal gap between the simulated and theoretical curves, in dB.

        Reported because it is the number a radio engineer reads off a BER
        plot: an ideal simulated link should show ~0 dB, and anything else
        points at a modelling or normalisation error.
        """
        from physim.phy.modulation import get_modulation
        from physim.phy.theory import required_ebn0_db

        mod = get_modulation(self.scheme)
        gaps = []
        for p in self.points:
            if p.n_bit_errors < 50 or not (1e-6 < p.ber < 0.2):
                continue
            need = required_ebn0_db(p.ber, mod.M, mod.family)
            if np.isfinite(need):
                gaps.append(p.ebn0_db - need)
        return float(np.mean(gaps)) if gaps else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme, "channel": self.channel, "seed": self.seed,
            "params": self.params,
            "points": [p.as_dict() for p in self.points],
            "worst_relative_deviation": self.worst_relative_deviation(),
            "implementation_loss_db": self.implementation_loss_db(),
        }


def wilson_interval(errors: int, trials: int, z: float = 1.96
                    ) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if trials == 0:
        return 0.0, 1.0
    p = errors / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / trials + z2 / (4 * trials**2))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _apply_channel(symbols: np.ndarray, channel: str, noise_var: float,
                   rng: np.random.Generator, rician_k_db: float,
                   fd_ts: float) -> np.ndarray:
    """Transmit, fade, add noise, and equalise with perfect CSI."""
    if channel == "awgn":
        return ch.awgn(symbols, noise_var, rng)

    if channel == "rayleigh":
        if fd_ts > 0:
            h = ch.jakes_fading(symbols.size, fd_ts, rng)
        else:
            h = ch.rayleigh_coefficients(symbols.shape, rng)
    elif channel == "rician":
        h = ch.rician_coefficients(symbols.shape, rician_k_db, rng)
    else:
        raise ValueError(f"unknown channel '{channel}', expected one of {CHANNELS}")

    received = h * symbols + ch.awgn(np.zeros_like(symbols), noise_var, rng)
    # Zero-forcing with perfect channel knowledge: the noise is scaled by
    # 1/h, which is precisely why deep fades dominate the error rate.
    return received / h


def simulate_ber_point(
    mod: Modulation | str,
    ebn0_db: float,
    channel: str = "awgn",
    rng: np.random.Generator | None = None,
    target_errors: int = 200,
    max_symbols: int = 2_000_000,
    min_symbols: int = 2_000,
    chunk_symbols: int = 100_000,
    rician_k_db: float = 6.0,
    fd_ts: float = 0.0,
) -> BERPoint:
    """Measure BER and SER at one Eb/N0 by chunked Monte Carlo."""
    if isinstance(mod, str):
        mod = get_modulation(mod)
    rng = rng or np.random.default_rng()

    k = mod.bits_per_symbol
    noise_var = float(ch.noise_variance(ebn0_db, k))

    n_bits = n_bit_err = n_sym = n_sym_err = 0
    started = time.perf_counter()

    while n_sym < max_symbols and (n_bit_err < target_errors
                                   or n_sym < min_symbols):
        size = min(chunk_symbols, max_symbols - n_sym)
        tx_bits = mod.random_bits(size, rng)
        symbols = mod.modulate(tx_bits)
        r = _apply_channel(symbols, channel, noise_var, rng, rician_k_db, fd_ts)

        rx_idx = mod.demodulate_indices(r)
        tx_idx = _bits_to_idx(tx_bits, k)
        rx_bits = _idx_to_bits(rx_idx, k)

        n_bit_err += int(np.count_nonzero(tx_bits != rx_bits))
        n_sym_err += int(np.count_nonzero(tx_idx != rx_idx))
        n_bits += tx_bits.size
        n_sym += size

    ber = n_bit_err / n_bits if n_bits else 0.0
    ser = n_sym_err / n_sym if n_sym else 0.0
    lo, hi = wilson_interval(n_bit_err, n_bits)
    theory = float(np.atleast_1d(
        theoretical_ber(np.array([ebn0_db]), mod.name, channel)
    )[0])

    return BERPoint(
        ebn0_db=float(ebn0_db), ber=ber, ser=ser, n_bits=n_bits,
        n_bit_errors=n_bit_err, n_symbols=n_sym, n_symbol_errors=n_sym_err,
        theory_ber=theory, ci_low=lo, ci_high=hi,
        elapsed_s=time.perf_counter() - started,
    )


def _bits_to_idx(bits: np.ndarray, k: int) -> np.ndarray:
    from physim.phy.modulation import bits_to_ints

    return bits_to_ints(bits, k)


def _idx_to_bits(idx: np.ndarray, k: int) -> np.ndarray:
    from physim.phy.modulation import ints_to_bits

    return ints_to_bits(idx, k)


def ber_sweep(
    scheme: str = "qpsk",
    ebn0_db: Sequence[float] | np.ndarray = tuple(range(0, 13)),
    channel: str = "awgn",
    seed: int | None = 12345,
    target_errors: int = 200,
    max_symbols: int = 1_000_000,
    **kwargs,
) -> BERSweep:
    """Sweep Eb/N0 for one scheme, reproducibly.

    The seed is part of the returned record: a BER curve that cannot be
    regenerated bit-for-bit is not a measurement, it is an anecdote.
    """
    mod = get_modulation(scheme)
    rng = np.random.default_rng(seed)
    sweep = BERSweep(scheme=scheme, channel=channel, seed=seed,
                     params={"target_errors": target_errors,
                             "max_symbols": max_symbols, **kwargs})
    for value in np.atleast_1d(np.asarray(ebn0_db, dtype=float)):
        sweep.points.append(simulate_ber_point(
            mod, float(value), channel=channel, rng=rng,
            target_errors=target_errors, max_symbols=max_symbols, **kwargs
        ))
    return sweep


def constellation_samples(scheme: str = "qam16", ebn0_db: float = 15.0,
                          n_symbols: int = 2000, channel: str = "awgn",
                          seed: int | None = 7, rician_k_db: float = 6.0,
                          ) -> dict[str, Any]:
    """Received scatter at a given Eb/N0, for the constellation view."""
    mod = get_modulation(scheme)
    rng = np.random.default_rng(seed)
    noise_var = float(ch.noise_variance(ebn0_db, mod.bits_per_symbol))
    bits = mod.random_bits(n_symbols, rng)
    tx = mod.modulate(bits)
    rx = _apply_channel(tx, channel, noise_var, rng, rician_k_db, 0.0)
    # EVM as a percentage of the reference RMS amplitude (which is 1 by
    # construction, but the ratio is written out so the definition is clear).
    evm = float(100.0 * np.sqrt(np.mean(np.abs(rx - tx) ** 2))
                / np.sqrt(np.mean(np.abs(tx) ** 2)))
    return {
        "scheme": scheme,
        "ebn0_db": ebn0_db,
        "channel": channel,
        "ideal": [[float(c.real), float(c.imag)] for c in mod.constellation],
        "received": [[float(z.real), float(z.imag)] for z in rx[:n_symbols]],
        "evm_pct": evm,
        # EVM and SNR are two views of the same noise power, so reporting
        # both makes the constellation view checkable against the Eb/N0 it
        # was asked for: Es/N0 = Eb/N0 + 10*log10(bits per symbol).
        "evm_snr_db": float(-20.0 * np.log10(evm / 100.0)),
        "esn0_db": float(ch.ebn0_to_esn0_db(ebn0_db, mod.bits_per_symbol)),
        "noise_var": noise_var,
    }
