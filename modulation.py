"""Gray-coded linear modulation: mapping, hard demapping and soft LLRs.

Design notes
------------
* Every constellation is normalised to unit average symbol energy, so an
  Es/N0 in decibels means the same thing regardless of order.
* Square QAM is demapped by independent per-dimension slicing rather than by
  a nearest-neighbour search over all ``M`` points. Slicing is ``O(n)`` where
  the search is ``O(n*M)``, which is the difference between a 64-QAM sweep
  finishing in seconds and finishing in minutes.
* Gray labelling is not decoration: it is what makes the nearest-neighbour
  symbol error contribute exactly one bit error, and therefore what makes
  ``BER ~ SER / log2(M)`` hold at high SNR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ Gray code
def gray_encode(m: np.ndarray | int) -> np.ndarray:
    """Binary-reflected Gray code: ``g = m XOR (m >> 1)``."""
    m = np.asarray(m)
    return m ^ (m >> 1)


def gray_decode(g: np.ndarray | int, n_bits: int) -> np.ndarray:
    """Inverse Gray code by prefix XOR (doubling shifts)."""
    m = np.asarray(g).copy()
    shift = 1
    while shift < n_bits:
        m = m ^ (m >> shift)
        shift <<= 1
    return m


def bits_to_ints(bits: np.ndarray, k: int) -> np.ndarray:
    """Pack a bit stream into k-bit integers, most significant bit first."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size % k:
        raise ValueError(f"bit count {bits.size} is not a multiple of k={k}")
    groups = bits.reshape(-1, k)
    weights = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)
    return groups @ weights


def ints_to_bits(values: np.ndarray, k: int) -> np.ndarray:
    """Unpack k-bit integers into a bit stream, most significant bit first."""
    values = np.asarray(values, dtype=np.int64).ravel()
    shifts = np.arange(k - 1, -1, -1)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).ravel()


# --------------------------------------------------------------- constellation
@dataclass
class Modulation:
    """A Gray-labelled constellation with unit average symbol energy.

    ``constellation[i]`` is the complex point carrying bit label ``i``.
    """

    name: str
    M: int
    family: str  # "pam" (real), "psk", or "qam"
    constellation: np.ndarray

    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.M))

    @property
    def is_real(self) -> bool:
        return self.family == "pam"

    @property
    def mean_energy(self) -> float:
        return float(np.mean(np.abs(self.constellation) ** 2))

    @property
    def min_distance(self) -> float:
        """Minimum Euclidean distance, which sets the high-SNR error floor."""
        c = self.constellation
        d = np.abs(c[:, None] - c[None, :])
        np.fill_diagonal(d, np.inf)
        return float(d.min())

    # -------------------------------------------------------------- transmit
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Map a bit stream to complex symbols."""
        idx = bits_to_ints(bits, self.bits_per_symbol)
        return self.constellation[idx]

    def random_bits(self, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=n_symbols * self.bits_per_symbol,
                            dtype=np.uint8)

    # --------------------------------------------------------------- receive
    def demodulate(self, r: np.ndarray) -> np.ndarray:
        """Hard-decision demapping to a bit stream."""
        return ints_to_bits(self.demodulate_indices(r), self.bits_per_symbol)

    def demodulate_indices(self, r: np.ndarray) -> np.ndarray:
        """Hard-decision demapping to bit-label integers."""
        r = np.asarray(r)
        if self.family == "qam":
            return _slice_square_qam(r, self.M)
        if self.family == "pam":
            return _slice_pam(r.real, self.M)
        if self.family == "psk":
            return _slice_psk(r, self.M)
        return _nearest_neighbour(r, self.constellation)

    def llr(self, r: np.ndarray, noise_var: float,
            chunk: int = 200_000) -> np.ndarray:
        """Max-log soft bit LLRs, ordered like :meth:`demodulate`.

        Positive LLR favours a transmitted ``0``. Computed by the max-log
        approximation, which replaces the log-sum-exp over each bit subset by
        its dominant term -- within about 0.1 dB of exact at the SNRs where a
        soft-decision decoder operates, for a fraction of the cost.
        """
        r = np.asarray(r).ravel()
        k = self.bits_per_symbol
        labels = np.arange(self.M)
        bit_is_one = ((labels[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(bool)
        out = np.empty((r.size, k))
        for start in range(0, r.size, chunk):
            block = r[start:start + chunk]
            d2 = np.abs(block[:, None] - self.constellation[None, :]) ** 2
            for b in range(k):
                d0 = d2[:, ~bit_is_one[:, b]].min(axis=1)
                d1 = d2[:, bit_is_one[:, b]].min(axis=1)
                out[start:start + block.size, b] = (d1 - d0) / noise_var
        return out.ravel()

    def meta(self) -> dict:
        return {
            "name": self.name,
            "M": self.M,
            "family": self.family,
            "bits_per_symbol": self.bits_per_symbol,
            "mean_energy": self.mean_energy,
            "min_distance": self.min_distance,
            "points": [[float(c.real), float(c.imag)]
                       for c in self.constellation],
        }


# ------------------------------------------------------------------- slicers
def _pam_levels(m_side: int) -> np.ndarray:
    """Unnormalised PAM levels ``-(m-1), ..., -1, 1, ..., (m-1)``."""
    return 2.0 * np.arange(m_side) - (m_side - 1)


def _pam_scale(M: int, family: str) -> float:
    """Normalisation giving unit *average symbol* energy."""
    if family == "pam":
        return float(np.sqrt(3.0 / (M**2 - 1))) if M > 2 else 1.0
    return float(np.sqrt(3.0 / (2.0 * (M - 1))))  # square QAM, both dimensions


def _slice_pam(x: np.ndarray, M: int) -> np.ndarray:
    scale = _pam_scale(M, "pam")
    idx = np.rint((x / scale + (M - 1)) / 2.0).astype(np.int64)
    np.clip(idx, 0, M - 1, out=idx)
    return gray_encode(idx)


def _slice_square_qam(r: np.ndarray, M: int) -> np.ndarray:
    m_side = int(np.sqrt(M))
    k_half = int(np.log2(m_side))
    scale = _pam_scale(M, "qam")
    i_idx = np.rint((r.real / scale + (m_side - 1)) / 2.0).astype(np.int64)
    q_idx = np.rint((r.imag / scale + (m_side - 1)) / 2.0).astype(np.int64)
    np.clip(i_idx, 0, m_side - 1, out=i_idx)
    np.clip(q_idx, 0, m_side - 1, out=q_idx)
    return (gray_encode(i_idx) << k_half) | gray_encode(q_idx)


def _slice_psk(r: np.ndarray, M: int) -> np.ndarray:
    idx = np.rint(np.angle(r) * M / (2.0 * np.pi)).astype(np.int64) % M
    return gray_encode(idx)


def _nearest_neighbour(r: np.ndarray, constellation: np.ndarray,
                       chunk: int = 200_000) -> np.ndarray:
    r = np.asarray(r).ravel()
    out = np.empty(r.size, dtype=np.int64)
    for start in range(0, r.size, chunk):
        block = r[start:start + chunk]
        d = np.abs(block[:, None] - constellation[None, :])
        out[start:start + block.size] = np.argmin(d, axis=1)
    return out


# ------------------------------------------------------------------ factories
def _square_qam(M: int, name: str) -> Modulation:
    m_side = int(round(np.sqrt(M)))
    if m_side * m_side != M or (m_side & (m_side - 1)):
        raise ValueError(f"{M}-QAM is not a square constellation")
    k_half = int(np.log2(m_side))
    scale = _pam_scale(M, "qam")
    levels = _pam_levels(m_side) * scale

    points = np.empty(M, dtype=complex)
    for label in range(M):
        i_label = label >> k_half
        q_label = label & (m_side - 1)
        i_idx = gray_decode(np.array(i_label), k_half)
        q_idx = gray_decode(np.array(q_label), k_half)
        points[label] = levels[int(i_idx)] + 1j * levels[int(q_idx)]
    return Modulation(name=name, M=M, family="qam", constellation=points)


def _psk(M: int, name: str) -> Modulation:
    labels = np.arange(M)
    idx = gray_decode(labels, int(np.log2(M)))
    points = np.exp(2j * np.pi * idx / M)
    return Modulation(name=name, M=M, family="psk", constellation=points)


def _bpsk() -> Modulation:
    # Antipodal real signalling; the quadrature branch carries no information
    # and is discarded at the slicer, which is why BPSK and QPSK share a BER.
    return Modulation(name="bpsk", M=2, family="pam",
                      constellation=np.array([-1.0 + 0j, 1.0 + 0j]))


_FACTORIES = {
    "bpsk": _bpsk,
    "qpsk": lambda: _square_qam(4, "qpsk"),
    "psk8": lambda: _psk(8, "psk8"),
    "qam16": lambda: _square_qam(16, "qam16"),
    "qam64": lambda: _square_qam(64, "qam64"),
    "qam256": lambda: _square_qam(256, "qam256"),
}

SCHEMES: tuple[str, ...] = tuple(_FACTORIES)


def get_modulation(name: str) -> Modulation:
    key = name.lower()
    if key not in _FACTORIES:
        raise KeyError(f"unknown scheme '{name}'. Available: {list(SCHEMES)}")
    return _FACTORIES[key]()


def list_modulations() -> list[dict]:
    return [get_modulation(n).meta() for n in SCHEMES]


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance between two integer arrays."""
    x = np.bitwise_xor(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    count = np.zeros_like(x)
    while np.any(x):
        count += x & 1
        x >>= 1
    return count


def count_bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(tx_bits) != np.asarray(rx_bits)))
