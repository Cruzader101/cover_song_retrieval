"""Known-answer tests for the pair-level factors and the two control methods.

The controls are the reason a null result in this analysis can be believed, so the
property that separates them -- one is key invariant, the other is not -- is
asserted here rather than assumed from the docstrings.
"""

import numpy as np
import pandas as pd
import pytest

from csr.analysis.factors import (
    align,
    design_matrix,
    oti_distance,
    pair_factors,
    semitone_distance,
)
from csr.data.synthetic import _progression, _render
from csr.eval.metrics import evaluate
from csr.methods import build


def test_semitone_distance_goes_the_short_way_round():
    """B to C is one semitone. The pitch axis is a circle, not a line."""
    assert semitone_distance(11, 0) == 1
    assert semitone_distance(0, 11) == 1
    assert semitone_distance(1, 8) == 5
    assert semitone_distance(0, 6) == 6  # a tritone is as far as it gets
    assert semitone_distance(4, 4) == 0


def test_semitone_distance_never_exceeds_six():
    a, b = np.meshgrid(np.arange(12), np.arange(12))
    distances = semitone_distance(a, b)
    assert distances.max() == 6
    assert np.array_equal(distances, distances.T)  # symmetric


def test_oti_distance_recovers_a_known_rotation():
    """Roll a profile by 3 and the measured transposition must be 3."""
    rng = np.random.default_rng(0)
    base = rng.random(12)
    profiles = np.stack([base, np.roll(base, 3), np.roll(base, 6)])
    assert list(oti_distance(profiles, [0, 0, 1], [1, 2, 2])) == [3, 6, 3]


def test_align_reorders_metadata_to_match_the_collection():
    """A collection is a subset in its own order, and that order indexes distances."""
    collection = pd.DataFrame({"perf_id": ["c", "a"]})
    metadata = pd.DataFrame({"perf_id": ["a", "b", "c"], "key_pc": [0, 1, 2]})
    profiles = np.array([[1.0], [2.0], [3.0]])

    aligned, rolled = align(collection, metadata, profiles)
    assert list(aligned["key_pc"]) == [2, 0]
    assert list(rolled[:, 0]) == [3.0, 1.0]


def test_align_raises_on_a_performance_it_cannot_describe():
    collection = pd.DataFrame({"perf_id": ["a", "missing"]})
    metadata = pd.DataFrame({"perf_id": ["a"], "key_pc": [0]})
    with pytest.raises(ValueError, match="no metadata row"):
        align(collection, metadata, np.zeros((1, 1)))


def factor_fixture():
    """Two performances of one work, differing in every factor by a known amount."""
    collection = pd.DataFrame(
        {
            "perf_id": ["P0", "P1"],
            "clique_id": ["W0", "W0"],
            "n_frames": [1000.0, 2000.0],
            "perf_artist": ["Someone", "Someone Else"],
        }
    )
    metadata = pd.DataFrame(
        {
            "perf_id": ["P0", "P1"],
            "key_pc": [11, 0],
            "scale": ["major", "minor"],
            "release_year": [1970.0, 1995.0],
            "instrumental": [False, True],
            "pulse_bpm": [120.0, 240.0],
        }
    )
    profiles = np.eye(12)[[0, 1]]
    pairs = np.array([[0, 1, 4]], dtype=np.int32)
    return pairs, collection, metadata, profiles


def test_pair_factors_hand_computed():
    """One pair, every factor derived by hand.

    key 11 vs 0        -> 1 semitone the short way round
    major vs minor     -> mode changed
    120 vs 240 BPM     -> exactly double, folded to 0
    1000 vs 2000 frames-> |log2(0.5)| = 1
    1970 vs 1995       -> 25 years
    vocal vs instrument-> mismatch
    different artists  -> not the same artist
    rank 4 of 2 items  -> log10(4 / 1) = log10(4)
    """
    frame = pair_factors(*factor_fixture())
    row = frame.iloc[0]

    assert row["key_shift"] == 1
    assert row["mode_change"] == 1.0
    assert row["pulse_ratio"] == pytest.approx(0.0)
    assert row["duration_ratio"] == pytest.approx(1.0)
    assert row["year_gap"] == pytest.approx(25.0)
    assert row["instrumental_mismatch"] == 1.0
    assert row["same_artist"] == 0.0
    assert row["log_rank"] == pytest.approx(np.log10(4.0))
    assert row["hit_at_10"] == 1.0
    assert row["clique_id"] == "W0"


def test_a_missing_year_stays_missing():
    """No imputation. A pair with no year is dropped downstream, not guessed at."""
    pairs, collection, metadata, profiles = factor_fixture()
    metadata.loc[1, "release_year"] = np.nan
    frame = pair_factors(pairs, collection, metadata, profiles)
    assert np.isnan(frame["year_gap"].iloc[0])


def test_design_matrix_drops_the_same_key_reference_level():
    """Six dummies for 1-6 semitones; no shift is the baseline they are read against."""
    frame = pd.DataFrame(
        {
            "key_shift": [0, 1, 6],
            "pulse_ratio": [0.1, 0.2, 0.3],
            "duration_ratio": [0.5, 0.6, 0.7],
            "year_gap": [1.0, 2.0, 3.0],
            "mode_change": [0.0, 1.0, 0.0],
            "instrumental_mismatch": [0.0, 0.0, 1.0],
            "same_artist": [1.0, 0.0, 0.0],
        }
    )
    X, names = design_matrix(frame)

    assert "key_shift_0" not in names
    assert names[:6] == tuple(f"key_shift_{s}" for s in range(1, 7))
    assert list(X[:, 0]) == [0.0, 1.0, 0.0]  # the 1-semitone dummy
    assert list(X[:, 5]) == [0.0, 0.0, 1.0]  # the 6-semitone dummy
    assert X[0, :6].sum() == 0.0  # the reference row fires no dummy


def test_design_matrix_standardizes_only_the_continuous_factors():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "key_shift": rng.integers(0, 7, 200),
            "pulse_ratio": rng.normal(0.2, 0.05, 200),
            "duration_ratio": rng.normal(0.4, 0.2, 200),
            "year_gap": rng.normal(20.0, 8.0, 200),
            "mode_change": rng.integers(0, 2, 200).astype(float),
            "instrumental_mismatch": rng.integers(0, 2, 200).astype(float),
            "same_artist": rng.integers(0, 2, 200).astype(float),
        }
    )
    X, names = design_matrix(frame)

    for name in ("pulse_ratio_sd", "duration_ratio_sd", "year_gap_sd"):
        column = X[:, names.index(name)]
        assert column.mean() == pytest.approx(0.0, abs=1e-12)
        assert column.std() == pytest.approx(1.0)

    # Binary factors keep their 0/1 scale, so their coefficient is the whole switch.
    assert set(np.unique(X[:, names.index("mode_change")])) == {0.0, 1.0}


def test_the_two_controls_differ_only_in_key_invariance(corpus):
    """The control pair, on a corpus whose covers are randomly transposed.

    synthetic_corpus rolls each performance by a random number of semitones, so a
    method that compares pitch-class profiles where they lie has nothing to go on,
    while the same comparison at its best rotation recovers the clique. If this ever
    stops holding, a null key effect elsewhere in the analysis stops being evidence.
    """
    manifest, chroma = corpus
    load = lambda pid: chroma[pid]  # noqa: E731
    cliques = manifest["clique_id"].to_list()
    queries = np.flatnonzero(manifest["clique_id"].duplicated(keep=False).to_numpy())

    plain = build("chroma_mean")(manifest, load, {}, None)
    rotated = build("chroma_mean_oti")(manifest, load, {}, None)

    plain_map = evaluate(plain[queries], cliques, queries).mean_average_precision
    rotated_map = evaluate(rotated[queries], cliques, queries).mean_average_precision
    assert rotated_map > 3.0 * plain_map


def same_key_corpus(n_cliques=8, per_clique=4, seed=0):
    """Cliques whose members are all in the original key.

    `synthetic_corpus` transposes every performance at random, which already puts a
    key-sensitive method on the floor -- there would be nothing left for an
    intervention to take away. Holding the key fixed gives it something to lose.
    """
    rng = np.random.default_rng(seed)
    rows, chroma = [], {}
    for clique in range(n_cliques):
        progression = _progression(rng)
        for member in range(per_clique):
            perf_id = f"P_{clique}_{member}"
            chroma[perf_id] = _render(
                rng,
                progression,
                frames_per_chord=int(rng.integers(120, 280)),
                n_repeats=int(rng.integers(8, 16)),
                transpose=0,
                noise=0.15,
            )
            rows.append(
                {
                    "perf_id": perf_id,
                    "clique_id": f"W_{clique}",
                    "n_frames": len(chroma[perf_id]),
                }
            )
    return pd.DataFrame(rows), chroma


def test_re_keying_everything_leaves_an_invariant_method_untouched():
    """The intervention the whole analysis rests on, in miniature.

    Roll every performance by its own random amount. That changes each one's key and
    nothing else -- same arrangement, same tempo, same length -- so a method whose
    key invariance is real must return an identical ranking, and one without it must
    lose most of its accuracy.

    This is what lets the analysis say that different-key covers ranking worse is a
    fact about which covers get transposed, not about the method failing on them.
    """
    manifest, chroma = same_key_corpus()
    cliques = manifest["clique_id"].to_list()
    queries = np.arange(len(manifest))

    rng = np.random.default_rng(0)
    shifts = dict(zip(manifest["perf_id"], rng.integers(1, 12, len(manifest)).tolist()))
    rekeyed = lambda pid: np.roll(chroma[pid], shifts[pid], axis=1)  # noqa: E731
    plain = lambda pid: chroma[pid]  # noqa: E731

    def score(method, load):
        d = build(method)(manifest, load, {}, np.random.default_rng(0))
        return evaluate(d[queries], cliques, queries).mean_average_precision

    assert score("ftm2d", rekeyed) == pytest.approx(score("ftm2d", plain), rel=1e-6)
    assert score("chroma_mean", rekeyed) < 0.7 * score("chroma_mean", plain)


def test_the_oti_control_is_symmetric(corpus):
    """A minimum over twelve rotations has to be, or the ranking depends on order.

    Rolling a by s and b by -s give the same inner product, so the twelve scores for
    (i, j) are the twelve for (j, i) in a different order and the minimum matches.
    """
    manifest, chroma = corpus
    d = build("chroma_mean_oti")(manifest, lambda pid: chroma[pid], {}, None)
    assert np.array_equal(d, d.T)
    assert np.array_equal(np.diag(d), np.zeros(len(d)))
