"""What changed between a query and one of its covers.

A pair table says where each true cover landed. This says what was done to it: how
far the key moved, whether the mode flipped, how much the timing was stretched, how
many years apart the two recordings are, whether one of them has no singer.

Every factor is deliberately *symmetric*, a property of the unordered pair rather
than of a direction through it. A cover being three semitones above the query is the
same transformation as the query being three semitones below the cover, and giving
those two rows different values would invent a difference the music does not have.
The consequence -- that each unordered pair appears twice with identical factors --
is exactly why standard errors are clustered on the clique.

Nothing here fills a gap with a plausible value. A pair missing a release year gets
NaN and is dropped from any model that uses the year, with the count reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from csr.analysis.tempo import octave_fold

#: The binary factors, kept as 0/1 so a coefficient reads as the cost of the switch
#: itself. The continuous ones are standardized instead, so theirs reads per
#: standard deviation, and the two are labelled differently in any table.
BINARY_FACTORS = ("mode_change", "instrumental_mismatch", "same_artist")
CONTINUOUS_FACTORS = ("pulse_ratio", "duration_ratio", "year_gap")


def semitone_distance(a, b):
    """Shortest way around the twelve pitch classes. -> 0-6.

    B to C is one semitone, not eleven: the pitch axis is a circle, which is the
    same fact that lets a key change be a roll of the chroma. Six is the maximum
    because past a tritone you are travelling back the other way.
    """
    diff = np.abs(np.asarray(a) - np.asarray(b)) % 12
    return np.minimum(diff, 12 - diff)


def oti_distance(profiles: np.ndarray, left, right):
    """Transposition between pairs, measured from chroma instead of a key label.

    The vectorised form of `chroma.optimal_transposition_index` over precomputed
    mean pitch-class profiles: rotate one profile against the other twelve ways and
    keep the best, then fold onto 0-6 the way `semitone_distance` does.

    This exists to check the key detector rather than to replace it. Essentia puts
    20% of the collection in C, which is a suspiciously popular key, and a factor
    built on a label that wrong in a systematic way would be measuring the detector.
    """
    left_profiles = profiles[left]
    scores = np.stack(
        [
            (left_profiles * np.roll(profiles, shift, axis=1)[right]).sum(axis=1)
            for shift in range(12)
        ]
    )
    best = scores.argmax(axis=0)
    return np.minimum(best, 12 - best)


def align(collection: pd.DataFrame, metadata: pd.DataFrame, profiles: np.ndarray):
    """Put the metadata and profiles into collection row order.

    The metadata table covers all 15000 performances; a collection is a subset in
    its own order, and that order is what indexes the distance matrix. Getting this
    wrong would silently describe one performance with another's key.
    """
    position = pd.Series(np.arange(len(metadata)), index=metadata["perf_id"])
    rows = position.reindex(collection["perf_id"])
    if rows.isna().any():
        missing = int(rows.isna().sum())
        raise ValueError(f"{missing} collection rows have no metadata row")
    rows = rows.to_numpy(dtype=int)
    return metadata.iloc[rows].reset_index(drop=True), profiles[rows]


def pair_factors(
    pairs: np.ndarray,
    collection: pd.DataFrame,
    metadata: pd.DataFrame,
    profiles: np.ndarray,
) -> pd.DataFrame:
    """Join the transformation between the two performances onto every pair.

    Args:
        pairs: `(n_pairs, 3)` of (query index, cover index, rank), as produced by
            `eval.metrics.pair_ranks`. Indices are into `collection`.
        collection: the rows the distance matrix was built over.
        metadata: per-performance table, already aligned by `align`.
        profiles: `(n_items, 12)` mean chroma, already aligned by `align`.

    Returns:
        One row per pair: the two ids, the clique, the rank and its transforms, and
        one column per factor.
    """
    query, cover, rank = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    n_items = len(collection)

    key_pc = metadata["key_pc"].to_numpy()
    minor = (metadata["scale"] == "minor").to_numpy()
    year = metadata["release_year"].to_numpy(dtype=float)
    instrumental = metadata["instrumental"].to_numpy(dtype=bool)
    bpm = metadata["pulse_bpm"].to_numpy(dtype=float)
    frames = collection["n_frames"].to_numpy(dtype=float)
    artist = collection["perf_artist"].to_numpy()

    return pd.DataFrame(
        {
            "query_id": collection["perf_id"].to_numpy()[query],
            "cover_id": collection["perf_id"].to_numpy()[cover],
            "clique_id": collection["clique_id"].to_numpy()[query],
            "rank": rank,
            # Normalised by how much room there was to be wrong in. A rank of 300
            # out of 15000 and a rank of 13 out of 650 are the same achievement, and
            # only the normalised form can be compared across collections.
            "log_rank": np.log10(rank / (n_items - 1)),
            "hit_at_10": (rank <= 10).astype(float),
            "key_shift": semitone_distance(key_pc[query], key_pc[cover]),
            "oti_shift": oti_distance(profiles, query, cover),
            "mode_change": (minor[query] != minor[cover]).astype(float),
            "pulse_ratio": octave_fold(bpm[query] / bpm[cover]),
            "duration_ratio": np.abs(np.log2(frames[query] / frames[cover])),
            "year_gap": np.abs(year[query] - year[cover]),
            "instrumental_mismatch": (
                instrumental[query] != instrumental[cover]
            ).astype(float),
            "same_artist": (artist[query] == artist[cover]).astype(float),
        }
    )


def design_matrix(
    frame: pd.DataFrame, key_column: str = "key_shift"
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the regressors. -> `(X, names)`, no intercept.

    Key distance enters as dummies for 1-6 semitones, with same-key as the omitted
    reference, so each coefficient reads as the cost of that shift against no shift
    at all. Entering it as a single number would assert that a tritone hurts six
    times as much as a semitone, which is a claim about music, not a measurement.

    Continuous factors are standardized so their coefficients compare directly.
    Binary ones are left alone: "the mode flipped" has no standard deviation worth
    dividing by, and 0-to-1 is already the interesting change.
    """
    columns, names = [], []

    shift = frame[key_column].to_numpy()
    for semitones in range(1, 7):
        columns.append((shift == semitones).astype(float))
        names.append(f"{key_column}_{semitones}")

    for name in CONTINUOUS_FACTORS:
        values = frame[name].to_numpy(dtype=float)
        spread = values.std()
        columns.append((values - values.mean()) / spread if spread else values * 0.0)
        names.append(f"{name}_sd")

    for name in BINARY_FACTORS:
        columns.append(frame[name].to_numpy(dtype=float))
        names.append(name)

    return np.column_stack(columns), tuple(names)
