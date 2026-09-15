"""Pulse shaping and the Nyquist ISI criterion.

A root-raised-cosine filter is not itself ISI-free -- the *cascade* of the
transmit RRC and the receive matched filter is, because it forms a raised
cosine, which has exact zero crossings at every non-zero multiple of the
symbol period. That property is the whole reason RRC is split across the two
ends of the link, and :func:`nyquist_isi_residual` measures it directly
instead of taking it on faith.
"""

from __future__ import annotations

import numpy as np


def evm_pct(received: np.ndarray, reference: np.ndarray) -> float:
    """Error vector magnitude as a percentage of the reference RMS."""
    err = np.mean(np.abs(received - reference) ** 2)
    return float(100.0 * np.sqrt(err / np.mean(np.abs(reference) ** 2)))


def rrc_taps(beta: float, span_symbols: int = 10, sps: int = 8) -> np.ndarray:
    """Root-raised-cosine impulse response, normalised to unit energy.

    ``beta`` is the excess-bandwidth (roll-off) factor: 0 gives the ideal
    brick-wall sinc with the tightest spectrum and the worst time-domain
    tails, 1 gives the gentlest tails and twice the Nyquist bandwidth.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("roll-off beta must lie in [0, 1]")
    n = span_symbols * sps
    if n % 2:
        n += 1
    t = np.arange(-n // 2, n // 2 + 1, dtype=float) / sps  # in symbol periods
    h = np.empty_like(t)

    eps = 1e-12
    # Three regions: the t = 0 singularity, the t = +/-1/(4*beta)
    # singularity, and the general case.
    at_zero = np.abs(t) < eps
    if beta > 0:
        at_quarter = np.abs(np.abs(t) - 1.0 / (4.0 * beta)) < eps
    else:
        at_quarter = np.zeros_like(t, dtype=bool)
    general = ~(at_zero | at_quarter)

    h[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
    if np.any(at_quarter):
        h[at_quarter] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
    tg = t[general]
    num = (np.sin(np.pi * tg * (1.0 - beta))
           + 4.0 * beta * tg * np.cos(np.pi * tg * (1.0 + beta)))
    den = np.pi * tg * (1.0 - (4.0 * beta * tg) ** 2)
    h[general] = num / den

    return h / np.linalg.norm(h)


def rc_taps(beta: float, span_symbols: int = 10, sps: int = 8) -> np.ndarray:
    """Raised-cosine response: the Nyquist pulse the RRC pair adds up to."""
    n = span_symbols * sps
    if n % 2:
        n += 1
    t = np.arange(-n // 2, n // 2 + 1, dtype=float) / sps
    sinc = np.sinc(t)
    denom = 1.0 - (2.0 * beta * t) ** 2
    cos_term = np.cos(np.pi * beta * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = sinc * cos_term / denom
    singular = np.abs(denom) < 1e-12
    if np.any(singular) and beta > 0:
        h[singular] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta))
    return h


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Zero-stuff to ``sps`` samples per symbol."""
    out = np.zeros(symbols.size * sps, dtype=complex)
    out[::sps] = symbols
    return out


def pulse_shape(symbols: np.ndarray, taps: np.ndarray, sps: int) -> np.ndarray:
    """Upsample and convolve with the transmit filter."""
    return np.convolve(upsample(symbols, sps), taps, mode="full")


def matched_filter(waveform: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Correlate with the time-reversed conjugate transmit pulse."""
    return np.convolve(waveform, np.conj(taps[::-1]), mode="full")


def nyquist_isi_residual(beta: float, span_symbols: int = 10, sps: int = 8
                         ) -> float:
    """Worst residual of the RRC-pair cascade at non-zero symbol instants.

    Zero would mean perfectly ISI-free. It is not exactly zero here because
    the filter is truncated to a finite span; the residual falls as the span
    grows, which is the engineering trade behind choosing a tap count.
    """
    taps = rrc_taps(beta, span_symbols, sps)
    cascade = np.convolve(taps, taps)
    peak = int(np.argmax(np.abs(cascade)))
    offsets = np.arange(-(span_symbols - 1), span_symbols) * sps
    idx = peak + offsets
    idx = idx[(idx >= 0) & (idx < cascade.size)]
    samples = cascade[idx] / cascade[peak]
    nonzero_instants = np.abs(offsets[(idx >= 0) & (idx < cascade.size)]) > 0
    return float(np.max(np.abs(samples[nonzero_instants])))


def eye_diagram(waveform: np.ndarray, sps: int, n_traces: int = 100,
                span_symbols: int = 2, offset: int = 0) -> dict:
    """Slice a waveform into overlapping traces for an eye diagram.

    The eye opening is the timing margin the symbol clock has to work with;
    a closed eye means no sampling instant recovers the symbols cleanly.
    """
    width = span_symbols * sps
    start = offset
    traces = []
    while len(traces) < n_traces and start + width < waveform.size:
        traces.append(np.real(waveform[start:start + width]).tolist())
        start += sps
    return {
        "sps": sps,
        "span_symbols": span_symbols,
        "t": (np.arange(width) / sps).tolist(),
        "traces": traces,
    }


def papr_db(waveform: np.ndarray) -> float:
    """Peak-to-average power ratio in dB.

    Sets how far a power amplifier must back off from saturation to stay
    linear, which is a direct hit to range: every dB of PAPR is a dB of
    transmit power the link does not get.
    """
    p = np.abs(waveform) ** 2
    mean = float(np.mean(p))
    if mean <= 0:
        return float("nan")
    return float(10.0 * np.log10(np.max(p) / mean))


def occupied_bandwidth(waveform: np.ndarray, sample_rate: float,
                       fraction: float = 0.99) -> float:
    """Bandwidth containing ``fraction`` of the total power."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(waveform))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(waveform.size, 1.0 / sample_rate))
    total = spectrum.sum()
    order = np.argsort(np.abs(freqs))
    cumulative = np.cumsum(spectrum[order]) / total
    idx = int(np.searchsorted(cumulative, fraction))
    idx = min(idx, order.size - 1)
    return float(2.0 * np.abs(freqs[order][idx]))


def shaped_link_demo(scheme: str = "qpsk", beta: float = 0.35, sps: int = 8,
                     n_symbols: int = 400, ebn0_db: float | None = 20.0,
                     span_symbols: int = 10, seed: int | None = 3,
                     symbol_rate_hz: float = 1e6,
                     return_waveforms: bool = False) -> dict:
    """Shape, transmit through AWGN, matched-filter, and report the result.

    ``ebn0_db=None`` runs the chain noiseless, which is the only way to tell
    a pulse-shaping bug from a noise-floor effect: with no noise the bit
    error count must be exactly zero.

    Scalar outputs are JSON-safe so the HTTP layer can return the dict
    unchanged; pass ``return_waveforms=True`` for the raw arrays.
    """
    from physim.phy import channel as ch
    from physim.phy.modulation import get_modulation

    rng = np.random.default_rng(seed)
    mod = get_modulation(scheme)
    taps = rrc_taps(beta, span_symbols, sps)

    bits = mod.random_bits(n_symbols, rng)
    symbols = mod.modulate(bits)
    tx = pulse_shape(symbols, taps, sps)

    # Noise is added at the waveform rate, so scale by sps to keep Eb/N0
    # referred to the symbol rate rather than the sample rate.
    if ebn0_db is None:
        noise_var = 0.0
        rx = tx
    else:
        noise_var = float(ch.noise_variance(ebn0_db, mod.bits_per_symbol)) * sps
        rx = ch.awgn(tx, noise_var, rng)
    mf = matched_filter(rx, taps)

    delay = taps.size - 1  # group delay of the two cascaded filters
    sampled = mf[delay:delay + n_symbols * sps:sps]
    n_recovered = sampled.size
    rx_bits = mod.demodulate(sampled)
    bit_errors = int(np.count_nonzero(
        rx_bits != bits[: n_recovered * mod.bits_per_symbol]))

    out = {
        "scheme": scheme, "beta": beta, "sps": sps, "ebn0_db": ebn0_db,
        "taps": taps.tolist(),
        "papr_tx_db": papr_db(tx),
        "isi_residual": nyquist_isi_residual(beta, span_symbols, sps),
        "eye": eye_diagram(mf[delay:], sps, n_traces=80, span_symbols=2),
        "sampled": [[float(z.real), float(z.imag)] for z in sampled[:400]],
        "n_symbols": int(n_recovered),
        "bit_errors": bit_errors,
        "ber": bit_errors / max(n_recovered * mod.bits_per_symbol, 1),
        "evm_pct": float(evm_pct(sampled, symbols[:n_recovered])),
        "symbol_rate_hz": float(symbol_rate_hz),
        "occupied_bw_hz": float(occupied_bandwidth(tx, symbol_rate_hz * sps)),
        "nyquist_bw_hz": float(symbol_rate_hz * (1.0 + beta)),
    }
    if return_waveforms:
        out.update(tx_waveform=tx, rx_waveform=mf[delay:], symbols=symbols)
    return out
