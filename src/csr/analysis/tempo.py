"""How fast a performance moves, estimated from chroma alone.

Da-TACOS ships madmom's beat-tracker output, but the local archive is a truncated
fragment of the real one, so this derives a pulse from the HPCP we already have:
the flux of the chroma is a novelty curve, and its dominant periodicity is a pulse.

The name is `pulse_bpm`, not `tempo_bpm`, and the difference is real. Chroma
deliberately suppresses percussion, so what this tracks is harmonic rhythm -- the
rate at which chords change -- which need not be the beat, and is routinely some
simple fraction of it.

That is still enough for the only thing it is used for. Two performances of one work
play the same chord sequence, so the *ratio* of their harmonic rhythms is the ratio
of their tempos whatever the two absolute numbers mean. Nothing here compares a BPM
across different works, and `octave_fold` discards the factor-of-two ambiguity that
any periodicity estimate carries.
"""

from __future__ import annotations

import numpy as np

from csr.data.datacos import FRAME_RATE_HZ
from csr.features.chroma import normalize_frames

#: Searched periodicity range. Wide enough to hold both a beat and a slow harmonic
#: rhythm, narrow enough to exclude the phrase-length periodicities that would
#: otherwise win on music with a repeating progression.
BPM_RANGE = (40.0, 200.0)


def novelty(chroma: np.ndarray) -> np.ndarray:
    """Half-wave-rectified chroma flux. ``(n_frames, 12) -> (n_frames - 1,)``.

    Only increases count. A chord arriving is an event; the previous one decaying is
    the same event seen from behind, and counting both would put two peaks in the
    curve where the music has one.

    Frames are normalised first, so a loud master and a quiet one give comparable
    curves instead of tempo tracking recording level.
    """
    frames = normalize_frames(np.asarray(chroma, dtype=np.float32), "max")
    return np.maximum(np.diff(frames, axis=0), 0.0).sum(axis=1)


def pulse_period(
    curve: np.ndarray,
    source_hz: float = FRAME_RATE_HZ,
    bpm_range: tuple[float, float] = BPM_RANGE,
    tolerance: float = 0.05,
) -> float:
    """Dominant period of a novelty curve, in frames.

    Autocorrelation, normalised by the number of overlapping samples at each lag.
    Without that division the correlation decays with lag purely because fewer terms
    contribute to it, and the peak picker would drift toward the shortest lag in the
    window regardless of the music.

    Among lags that explain the curve about as well -- within `tolerance` of the
    best -- the shortest wins. Every multiple of a true period correlates nearly as
    well as the period itself, measured here at within 1% of each other, so a plain
    argmax chooses between a beat and three of them on noise. That is the metrical
    ambiguity every tempo estimator has, and left alone it would let two
    performances of one work resolve the same music to different levels and look
    re-timed when they are not.
    """
    curve = np.asarray(curve, dtype=np.float64)
    curve = curve - curve.mean()
    n = len(curve)

    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(curve, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    acf /= np.arange(n, 0, -1)

    # Round inward. A lag is a whole number of frames, so rounding outward would
    # report a BPM just past the range that was asked for.
    lo = int(np.ceil(60.0 * source_hz / bpm_range[1]))
    hi = min(int(np.floor(60.0 * source_hz / bpm_range[0])), n - 1)
    if hi <= lo:
        raise ValueError(
            f"novelty curve of {n} frames is too short to hold a period in "
            f"{bpm_range[0]}-{bpm_range[1]} BPM at {source_hz:.1f} Hz"
        )
    window = acf[lo : hi + 1]
    best = window.max()
    good = np.flatnonzero(window >= best - tolerance * abs(best))
    return float(lo + int(good[0]))


def pulse_bpm(
    chroma: np.ndarray,
    source_hz: float = FRAME_RATE_HZ,
    bpm_range: tuple[float, float] = BPM_RANGE,
) -> float:
    """Dominant periodicity of a performance's chroma flux, in beats per minute."""
    return 60.0 * source_hz / pulse_period(novelty(chroma), source_hz, bpm_range)


def octave_fold(ratio: float | np.ndarray) -> float | np.ndarray:
    """``|log2(ratio)|`` folded into ``[0, 0.5]``. Half time is no difference at all.

    Any periodicity estimate can land on a half or double of the rate a listener
    would tap, and two performances of one work can disagree about which. Folding
    means the factor measures how much the *music* was re-timed rather than how the
    two estimators happened to resolve that ambiguity.
    """
    exponent = np.log2(ratio)
    return np.abs(exponent - np.round(exponent))
