"""Analytical error probabilities: the reference the simulator is scored against.

Two levels of reference are provided for square QAM:

``ber_qam_exact``
    Built from the exact per-dimension PAM transition probabilities. Square
    QAM with Gray labelling separates into two independent PAM channels, so
    summing ``P(m -> m') * hamming(gray(m), gray(m'))`` over every level pair
    gives the exact BER with no high-SNR assumption. This is what the Monte
    Carlo results are compared against.

``ber_qam_nearest_neighbour``
    The textbook approximation that counts only adjacent-symbol errors. It is
    accurate above roughly 10 dB and included so the gap to the exact value
    is visible rather than assumed away.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import erfc

from physim.phy.modulation import gray_encode


def qfunc(x: np.ndarray | float) -> np.ndarray:
    """Gaussian tail probability ``Q(x) = P(N(0,1) > x)``."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def qfunc_inv(p: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`qfunc`, by bisection on the monotone tail."""
    from scipy.special import erfcinv

    return np.sqrt(2.0) * erfcinv(2.0 * np.asarray(p, dtype=float))


# ----------------------------------------------------------------------- AWGN
def ber_bpsk(ebn0_db: np.ndarray | float) -> np.ndarray:
    """``Q(sqrt(2 Eb/N0))`` -- exact for BPSK and, per bit, for Gray QPSK."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return qfunc(np.sqrt(2.0 * ebn0))


ber_qpsk = ber_bpsk


def ser_psk(ebn0_db: np.ndarray | float, M: int) -> np.ndarray:
    """Symbol error rate for coherent M-PSK (tight approximation for M >= 4)."""
    k = np.log2(M)
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return 2.0 * qfunc(np.sqrt(2.0 * k * ebn0) * np.sin(np.pi / M))


def ber_psk(ebn0_db: np.ndarray | float, M: int) -> np.ndarray:
    """Gray-coded M-PSK BER: one bit error per nearest-neighbour symbol error."""
    if M == 2:
        return ber_bpsk(ebn0_db)
    if M == 4:
        return ber_qpsk(ebn0_db)
    return ser_psk(ebn0_db, M) / np.log2(M)


@lru_cache(maxsize=32)
def _pam_bit_labels(m_side: int) -> tuple[np.ndarray, int]:
    k_half = int(np.log2(m_side))
    labels = gray_encode(np.arange(m_side))
    return labels, k_half


def _hamming_matrix(labels: np.ndarray, n_bits: int) -> np.ndarray:
    x = labels[:, None] ^ labels[None, :]
    out = np.zeros_like(x)
    for b in range(n_bits):
        out += (x >> b) & 1
    return out


#: Below this BER the exact QAM sum loses significance to cancellation
#: between nearly equal Q-function terms. Nothing operates down here (1e-20
#: is one error per 3 million years at 10 Gb/s), but the limit is recorded
#: so nobody reads meaning into the digits.
QAM_EXACT_PRECISION_FLOOR = 1e-20


def ber_qam_exact(ebn0_db: np.ndarray | float, M: int) -> np.ndarray:
    """Exact BER for Gray-coded square M-QAM in AWGN.

    Square QAM factors into two independent ``sqrt(M)``-PAM channels. For
    each pair of levels the transition probability is the Gaussian mass in
    the receiving decision interval, and each transition costs the Hamming
    distance between the two Gray labels. Summing over all pairs is exact --
    no nearest-neighbour assumption, valid at any SNR.
    """
    ebn0_db = np.atleast_1d(np.asarray(ebn0_db, dtype=float))
    k = int(np.log2(M))
    if M == 2:
        return ber_bpsk(ebn0_db)

    m_side = int(round(np.sqrt(M)))
    if m_side * m_side != M:
        raise ValueError(f"{M}-QAM is not square")
    labels, k_half = _pam_bit_labels(m_side)
    ham = _hamming_matrix(labels, k_half)

    scale = np.sqrt(3.0 / (2.0 * (M - 1)))          # unit mean symbol energy
    levels = scale * (2.0 * np.arange(m_side) - (m_side - 1))

    ebn0 = 10.0 ** (ebn0_db / 10.0)
    n0 = 1.0 / (k * ebn0)                            # Es = 1
    sigma = np.sqrt(n0 / 2.0)                        # per real dimension

    # Decision boundaries midway between adjacent levels.
    edges = np.concatenate(([-np.inf], (levels[:-1] + levels[1:]) / 2.0,
                            [np.inf]))

    out = np.empty(ebn0_db.shape)
    for idx, sig in enumerate(np.atleast_1d(sigma)):
        z = (edges[None, :] - levels[:, None]) / sig
        cdf = 0.5 * erfc(-z / np.sqrt(2.0))          # Phi(z)
        p_trans = cdf[:, 1:] - cdf[:, :-1]           # P(sent m -> decided m')
        out[idx] = (p_trans * ham).sum() / (m_side * k_half)
    return out


def ber_qam_nearest_neighbour(ebn0_db: np.ndarray | float, M: int) -> np.ndarray:
    """Standard nearest-neighbour QAM approximation (accurate above ~10 dB)."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    k = np.log2(M)
    m_side = np.sqrt(M)
    return (4.0 / k) * (1.0 - 1.0 / m_side) * qfunc(
        np.sqrt(3.0 * k * ebn0 / (M - 1.0))
    )


def ser_qam(ebn0_db: np.ndarray | float, M: int) -> np.ndarray:
    """Exact SER for square M-QAM (product form of the two PAM channels)."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    k = np.log2(M)
    q = qfunc(np.sqrt(3.0 * k * ebn0 / (M - 1.0)))
    a = 2.0 * (1.0 - 1.0 / np.sqrt(M))
    return 2.0 * a * q - a**2 * q**2


# --------------------------------------------------------------- flat fading
def ber_rayleigh(ebn0_db: np.ndarray | float, M: int,
                 family: str = "qam") -> np.ndarray:
    """BER over flat Rayleigh fading with coherent detection and perfect CSI.

    Averaging ``Q(sqrt(c*gamma))`` over an exponential SNR distribution has
    the closed form ``0.5*(1 - sqrt(c*gbar/(2 + c*gbar)))``. The result decays
    like ``1/SNR`` rather than exponentially: the deep-fade tail, not the
    average SNR, is what dominates the error rate. That is the entire
    argument for diversity, and it is visible directly in the BER plot.
    """
    gbar = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    k = np.log2(M)
    if M == 2 or (M == 4 and family in ("qam", "psk")):
        c = 2.0  # antipodal / Gray QPSK per-bit
        return 0.5 * (1.0 - np.sqrt(c * gbar / (2.0 + c * gbar)))
    c = 3.0 * k / (M - 1.0)
    base = 0.5 * (1.0 - np.sqrt(c * gbar / (2.0 + c * gbar)))
    return (4.0 / k) * (1.0 - 1.0 / np.sqrt(M)) * base


def diversity_order_rayleigh(ebn0_db: np.ndarray, ber: np.ndarray) -> float:
    """Slope of log10(BER) against log10(SNR): 1 for flat Rayleigh, L for L-fold
    diversity. Fitted over points with a measurable error rate."""
    ebn0_db = np.asarray(ebn0_db, dtype=float)
    ber = np.asarray(ber, dtype=float)
    ok = (ber > 0) & np.isfinite(ber)
    if ok.sum() < 2:
        return float("nan")
    x = ebn0_db[ok] / 10.0                  # log10 of linear SNR
    return float(-np.polyfit(x, np.log10(ber[ok]), 1)[0])


# ------------------------------------------------------------------- capacity
def awgn_capacity(snr_db: np.ndarray | float) -> np.ndarray:
    """Shannon capacity in bits per complex symbol: ``log2(1 + SNR)``."""
    snr = 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)
    return np.log2(1.0 + snr)


def required_ebn0_db(target_ber: float, M: int, family: str = "qam",
                     lo: float = -10.0, hi: float = 60.0,
                     tol: float = 1e-4) -> float:
    """Eb/N0 needed to hit ``target_ber``, by bisection on the exact curve.

    This is the number a link budget actually needs: it converts a BER
    requirement into the SNR margin the radio has to deliver.
    """
    def ber_at(x: float) -> float:
        if family == "psk" and M > 4:
            return float(ber_psk(x, M))
        return float(np.atleast_1d(ber_qam_exact(x, M))[0])

    if ber_at(hi) > target_ber:
        return float("inf")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ber_at(mid) > target_ber:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def theoretical_ber(ebn0_db: np.ndarray, scheme: str,
                    channel: str = "awgn") -> np.ndarray:
    """Dispatch to the right closed form for a (scheme, channel) pair."""
    from physim.phy.modulation import get_modulation

    mod = get_modulation(scheme)
    M, family = mod.M, mod.family
    if channel == "rayleigh":
        return np.atleast_1d(ber_rayleigh(ebn0_db, M, family))
    if family == "psk" and M > 4:
        return np.atleast_1d(ber_psk(ebn0_db, M))
    return np.atleast_1d(ber_qam_exact(ebn0_db, M))
