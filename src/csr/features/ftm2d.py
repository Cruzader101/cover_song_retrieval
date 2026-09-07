"""The 2D Fourier magnitude descriptor (Bertin-Mahieux & Ellis, 2012).

The idea is one trick applied twice. The magnitude of a 2-D DFT is unchanged when
you circularly shift either axis of the input. Rolling the pitch axis is exactly
what transposing to another key does, and rolling the time axis is what starting
the recording at a different point does. So taking |FFT2| of a chroma patch throws
away key and starting offset while keeping the harmonic-rhythmic texture, which is
what makes two covers look alike.

Tempo invariance is *not* free here. It comes from the rate the chroma was
downsampled to before this runs, which is why that rate is a config knob.

The output is one fixed-length vector per performance, so comparing two songs is a
dot product rather than an alignment. That is the whole appeal: it scales to the
full 15000-item collection, where alignment methods cannot.
"""

from __future__ import annotations

import numpy as np


def _patches(chroma: np.ndarray, patch_frames: int, patch_hop: int) -> np.ndarray:
    """Slice (n_frames, 12) into overlapping (n_patches, patch_frames, 12) blocks.

    Performances shorter than one patch are zero-padded rather than rejected; the
    collection contains some very short recordings and dropping them would quietly
    change what we are measuring.
    """
    if len(chroma) < patch_frames:
        padded = np.zeros((patch_frames, 12), dtype=np.float32)
        padded[: len(chroma)] = chroma
        return padded[None]

    starts = range(0, len(chroma) - patch_frames + 1, patch_hop)
    return np.stack([chroma[s : s + patch_frames] for s in starts])


def ftm2d(
    chroma: np.ndarray,
    patch_frames: int = 180,
    patch_hop: int = 60,
    n_time_coeffs: int = 32,
    compress: str = "sqrt",
    aggregate: str = "median",
    drop_dc: bool = False,
) -> np.ndarray:
    """Describe one performance as a fixed-length, key-invariant vector.

    Args:
        chroma: (n_frames, 12), already downsampled and frame-normalised.
        patch_frames: length of each analysis window, in downsampled frames.
        patch_hop: step between windows.
        n_time_coeffs: how many of the lowest time frequencies to keep.
        compress: 'sqrt', 'log' or 'none' amplitude compression before the FFT.
        aggregate: 'median' or 'mean' over patches.
        drop_dc: zero the [0, 0] coefficient before normalising. That bin is the
            mean of the patch, and it carries about 97% of the descriptor's
            energy, which pins every pair of songs to a small angle and leaves
            the discriminative bins to fight over what is left. See the note
            below.

    Returns:
        (n_time_coeffs * 7,) float32 with unit L2 norm.

    Only pitch bins 0..6 are kept: for real input the 2-D FFT magnitude satisfies
    |F[u, v]| == |F[-u, -v]|, so bins 7..11 are mirror images of bins 1..5 and
    carry nothing new.

    On `drop_dc`: the DC bin is invariant in the same way as everything else, so
    keeping it costs nothing in principle. In practice it dominates: with it in,
    every cosine distance on the benchmark subsample falls in [0, 0.05], and
    dropping it raises MAP from 0.178 to 0.226 there and from 0.059 to 0.090 on
    the held-out split. It is off by default so that the descriptor stays the one
    Bertin-Mahieux & Ellis describe, and the gain is reported as a measured
    variant rather than folded silently into the baseline.

    Invariance is exact for transposition, and exact for time shift only *within*
    a patch -- shifting a whole performance moves frames across patch boundaries,
    so the aggregated descriptor drifts slightly. Measured drift on unit-norm
    descriptors is ~5e-3, against ~3e-9 for transposition. Both are asserted in
    the tests at those tolerances.
    """
    chroma = np.asarray(chroma, dtype=np.float32)
    if chroma.ndim != 2 or chroma.shape[1] != 12:
        raise ValueError(f"expected (n_frames, 12), got {chroma.shape}")

    if compress == "sqrt":
        chroma = np.sqrt(chroma)
    elif compress == "log":
        chroma = np.log1p(chroma)
    elif compress != "none":
        raise ValueError(f"unknown compression {compress!r}")

    patches = _patches(chroma, patch_frames, patch_hop)
    spectra = np.abs(np.fft.fft2(patches, axes=(1, 2)))
    kept = spectra[:, :n_time_coeffs, :7]

    reduce = {"median": np.median, "mean": np.mean}[aggregate]
    descriptor = reduce(kept, axis=0).ravel().astype(np.float32)

    if drop_dc:
        descriptor[0] = 0.0

    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm > 1e-9 else descriptor


def ftm2d_matrix(chromas, **kwargs) -> np.ndarray:
    """Stack descriptors for many performances. -> (n_perfs, d) float32."""
    return np.stack([ftm2d(c, **kwargs) for c in chromas])
