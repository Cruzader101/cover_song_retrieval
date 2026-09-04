"""Chroma preprocessing shared by every method.

Shapes are always (n_frames, 12): time down the rows, pitch class across the
columns, with column i one semitone above column i-1 and column 11 wrapping back
to column 0. Transposing a piece by k semitones is therefore a circular roll of
the pitch axis, which is what makes key invariance cheap.
"""

from __future__ import annotations

import numpy as np

from csr.data.datacos import FRAME_RATE_HZ


def normalize_frames(chroma: np.ndarray, kind: str = "max") -> np.ndarray:
    """Scale each frame independently. (n_frames, 12) -> (n_frames, 12).

    Silent frames stay at zero rather than being divided by ~0: a frame with no
    energy carries no pitch information, and inventing some would be worse than
    leaving it empty.
    """
    if kind == "none":
        return chroma.astype(np.float32, copy=False)
    if kind == "max":
        norm = chroma.max(axis=1, keepdims=True)
    elif kind == "l2":
        norm = np.linalg.norm(chroma, axis=1, keepdims=True)
    elif kind == "l1":
        norm = np.abs(chroma).sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown normalization {kind!r}")

    out = np.zeros_like(chroma, dtype=np.float32)
    loud = norm[:, 0] > 1e-9
    out[loud] = chroma[loud] / norm[loud]
    return out


def downsample(chroma: np.ndarray, factor: int, how: str = "mean") -> np.ndarray:
    """Average blocks of `factor` frames. (n_frames, 12) -> (n_frames // factor, 12).

    A trailing partial block is dropped. Factors below 2 are a no-op.
    """
    if factor <= 1:
        return chroma.astype(np.float32, copy=False)
    n = len(chroma) // factor
    if n == 0:  # too short to downsample; keep one averaged frame
        return chroma.mean(axis=0, keepdims=True).astype(np.float32)
    blocks = chroma[: n * factor].reshape(n, factor, 12)
    reduce = {"mean": np.mean, "median": np.median, "max": np.max}[how]
    return reduce(blocks, axis=1).astype(np.float32)


def downsample_to_rate(
    chroma: np.ndarray, target_hz: float, source_hz: float = FRAME_RATE_HZ, **kwargs
) -> np.ndarray:
    """Downsample to roughly `target_hz` frames per second.

    Analyses ask for a rate rather than a factor so that a wrong assumption about
    the hop size stays in one place instead of being baked into every config.
    """
    return downsample(chroma, max(int(round(source_hz / target_hz)), 1), **kwargs)


def transpose(chroma: np.ndarray, semitones: int) -> np.ndarray:
    """Shift the pitch axis by `semitones`, wrapping around."""
    return np.roll(chroma, semitones, axis=1)


def global_chroma(chroma: np.ndarray) -> np.ndarray:
    """Average pitch-class profile of a whole performance, L2-normalised. -> (12,)."""
    profile = chroma.mean(axis=0)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 1e-9 else profile


def optimal_transposition_index(a: np.ndarray, b: np.ndarray) -> int:
    """How far to roll `b` so its key matches `a`'s (Serra et al. 2008).

    Compares average pitch-class profiles at all 12 rotations and returns the best.
    Not symmetric as a value: oti(a, b) == -oti(b, a) mod 12.

    Args:
        a: (n_a, 12).  b: (n_b, 12).
    Returns:
        A shift in [0, 12) to pass to `transpose(b, shift)`.
    """
    ha, hb = global_chroma(a), global_chroma(b)
    scores = [ha @ np.roll(hb, s) for s in range(12)]
    return int(np.argmax(scores))
