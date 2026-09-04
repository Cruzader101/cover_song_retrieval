"""A small fake corpus with real clique structure.

Frame counts are generated at the real native rate (~86 Hz, see
`datacos.FRAME_RATE_HZ`), so a synthetic performance is roughly as long as a real
one and every downsampling factor in the project behaves the same way here as it
does on the actual data.

Every method can be developed and tested against this without the 9.6 GB download,
and it is deliberately easy: performances of one work share a chord progression and
differ only by key, tempo, starting point, length and noise. A method that handles
transposition and tempo should score near-perfectly here. One that does not will
score near random, which makes this a fast check on whether an implementation does
what its docstring claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _progression(rng: np.random.Generator, n_chords: int = 8) -> np.ndarray:
    """(n_chords, 12) major triads on random roots -- one work's harmony."""
    chords = np.zeros((n_chords, 12), dtype=np.float32)
    for i, root in enumerate(rng.integers(0, 12, size=n_chords)):
        for interval in (0, 4, 7):
            chords[i, (root + interval) % 12] = 1.0
    return chords


def _render(
    rng: np.random.Generator,
    progression: np.ndarray,
    frames_per_chord: int,
    n_repeats: int,
    transpose: int,
    noise: float,
) -> np.ndarray:
    """One performance of a progression as (n_frames, 12) float32."""
    looped = np.tile(progression, (n_repeats, 1))
    frames = np.repeat(looped, frames_per_chord, axis=0)
    frames = np.roll(frames, transpose, axis=1)
    start = rng.integers(0, max(len(frames) // 4, 1))
    frames = frames[start:]
    frames = frames + rng.normal(0.0, noise, frames.shape)
    return np.clip(frames, 0.0, None).astype(np.float32)


def synthetic_corpus(
    n_cliques: int = 8,
    per_clique: int = 4,
    n_singletons: int = 6,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build a fake collection shaped like the real one.

    Mirrors Da-TACOS in the way that matters: singleton cliques are included as
    distractors, so the collection is larger than the query set.

    Returns:
        (manifest, chroma) where manifest has the same columns the real one does
        and chroma maps perf_id -> (n_frames, 12) float32.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    chroma: dict[str, np.ndarray] = {}

    def add(clique_id: str, perf_id: str, array: np.ndarray) -> None:
        chroma[perf_id] = array
        rows.append(
            {
                "perf_id": perf_id,
                "clique_id": clique_id,
                "path": f"<synthetic>/{clique_id}/{perf_id}",
                "work_title": clique_id,
                "perf_artist": perf_id,
                "n_frames": len(array),
            }
        )

    for c in range(n_cliques):
        progression = _progression(rng)
        for p in range(per_clique):
            add(
                f"W_{c}",
                f"P_{c}_{p}",
                _render(
                    rng,
                    progression,
                    frames_per_chord=int(rng.integers(120, 280)),
                    n_repeats=int(rng.integers(8, 16)),
                    transpose=int(rng.integers(0, 12)),
                    noise=noise,
                ),
            )

    for s in range(n_singletons):
        add(
            f"S_{s}",
            f"P_s_{s}",
            _render(
                rng,
                _progression(rng),
                frames_per_chord=int(rng.integers(120, 280)),
                n_repeats=int(rng.integers(8, 16)),
                transpose=int(rng.integers(0, 12)),
                noise=noise,
            ),
        )

    return pd.DataFrame(rows), chroma
