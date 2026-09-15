"""Constellation mappings: energy, Gray labelling, and demodulation."""

import numpy as np
import pytest

from physim.phy.modulation import (
    SCHEMES,
    bits_to_ints,
    count_bit_errors,
    get_modulation,
    gray_decode,
    gray_encode,
    hamming_distance,
    ints_to_bits,
)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constellation_has_unit_average_energy(scheme):
    """Every scheme is normalised so Eb/N0 comparisons are like for like."""
    assert get_modulation(scheme).mean_energy == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constellation_size_matches_bits_per_symbol(scheme):
    mod = get_modulation(scheme)
    assert mod.constellation.size == mod.M == 2 ** mod.bits_per_symbol


@pytest.mark.parametrize("scheme", SCHEMES)
def test_noiseless_round_trip_is_exact(scheme):
    mod = get_modulation(scheme)
    rng = np.random.default_rng(7)
    bits = mod.random_bits(5000, rng)
    assert np.array_equal(mod.demodulate(mod.modulate(bits)), bits)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_neighbours_differ_by_one_bit(scheme):
    """Gray labelling: the nearest neighbour of any point in the
    constellation must differ in exactly one bit, which is what makes
    BER ~ SER / log2(M) at high SNR."""
    mod = get_modulation(scheme)
    pts = mod.constellation
    d = np.abs(pts[:, None] - pts[None, :])
    np.fill_diagonal(d, np.inf)
    labels = np.arange(mod.M)
    bits = ints_to_bits(labels, mod.bits_per_symbol).reshape(
        mod.M, mod.bits_per_symbol)
    dmin = d.min()
    for i in range(mod.M):
        for j in np.flatnonzero(d[i] <= dmin * 1.001):
            assert int(np.sum(bits[i] != bits[j])) == 1


def test_gray_code_is_invertible():
    values = np.arange(256)
    assert np.array_equal(gray_decode(gray_encode(values), 8), values)


def test_gray_code_successors_differ_by_one_bit():
    g = gray_encode(np.arange(64))
    transitions = np.bitwise_xor(g[:-1], g[1:])
    assert np.all(np.isin(transitions, [1, 2, 4, 8, 16, 32]))  # one bit set


def test_bit_packing_round_trip():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=6000, dtype=np.uint8)
    assert np.array_equal(ints_to_bits(bits_to_ints(bits, 6), 6).ravel(), bits)


def test_bit_packing_rejects_a_ragged_stream():
    with pytest.raises(ValueError):
        bits_to_ints(np.zeros(7, dtype=np.uint8), 4)


@pytest.mark.parametrize("scheme", ["qpsk", "qam16", "qam64", "psk8"])
def test_slicer_agrees_with_brute_force_nearest_neighbour(scheme):
    """The fast analytic slicers must give the same answer as an O(M) search
    over the whole constellation, including deep in the noise."""
    mod = get_modulation(scheme)
    rng = np.random.default_rng(11)
    r = (rng.normal(size=4000) + 1j * rng.normal(size=4000)) * 0.8
    brute = np.argmin(np.abs(r[:, None] - mod.constellation[None, :]), axis=1)
    assert np.array_equal(mod.demodulate_indices(r), brute)


@pytest.mark.parametrize("scheme", ["bpsk", "qpsk", "qam16"])
def test_llr_sign_agrees_with_the_hard_decision(scheme):
    """A soft decision that disagrees with the hard decision would silently
    break any downstream decoder."""
    mod = get_modulation(scheme)
    rng = np.random.default_rng(5)
    bits = mod.random_bits(2000, rng)
    r = mod.modulate(bits) + 0.05 * (rng.normal(size=2000) +
                                     1j * rng.normal(size=2000))
    llr = mod.llr(r, noise_var=0.01)
    assert np.array_equal((llr < 0).astype(np.uint8).ravel(),
                          mod.demodulate(r).reshape(llr.shape).ravel())


def test_llr_magnitude_grows_as_noise_falls():
    mod = get_modulation("qpsk")
    rng = np.random.default_rng(5)
    r = mod.modulate(mod.random_bits(1000, rng))
    assert (np.mean(np.abs(mod.llr(r, 0.001))) >
            np.mean(np.abs(mod.llr(r, 0.1))))


def test_hamming_distance_and_error_count_agree():
    rng = np.random.default_rng(13)
    a = rng.integers(0, 16, size=500)
    b = rng.integers(0, 16, size=500)
    by_int = int(hamming_distance(a, b).sum())
    by_bits = count_bit_errors(ints_to_bits(a, 4), ints_to_bits(b, 4))
    assert by_int == by_bits


def test_unknown_scheme_is_reported():
    with pytest.raises(KeyError, match="unknown scheme"):
        get_modulation("qam1024")
