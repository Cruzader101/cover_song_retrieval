"""Splits must never put two performances of the same work on opposite sides."""

import numpy as np
import pytest

from csr.data.datacos import query_mask
from csr.data.splits import Split, collection, split_by_clique, subsample_cliques


def test_splits_are_disjoint_and_complete(corpus):
    """The leakage guard. 8 real cliques at 70/10/20 -> 6/1/1."""
    manifest, _ = corpus
    split = split_by_clique(manifest, seed=0)
    assert set(split.train) & set(split.val) == set()
    assert set(split.train) & set(split.test) == set()
    assert set(split.val) & set(split.test) == set()

    real = {c for c, n in manifest["clique_id"].value_counts().items() if n > 1}
    assert set(split.train) | set(split.val) | set(split.test) == real
    assert (len(split.train), len(split.val), len(split.test)) == (6, 1, 1)


def test_singleton_cliques_are_never_in_a_split(corpus):
    """Distractors carry no supervision, so they belong to no side of the split."""
    manifest, _ = corpus
    split = split_by_clique(manifest, seed=0)
    everything = set(split.train) | set(split.val) | set(split.test)
    assert not any(c.startswith("S_") for c in everything)


def test_split_is_deterministic_under_a_seed(corpus):
    manifest, _ = corpus
    assert split_by_clique(manifest, seed=0) == split_by_clique(manifest, seed=0)
    assert split_by_clique(manifest, seed=0) != split_by_clique(manifest, seed=1)


def test_fractions_must_sum_to_one(corpus):
    manifest, _ = corpus
    with pytest.raises(ValueError, match="sum to 1"):
        split_by_clique(manifest, fractions=(0.5, 0.1, 0.1))


def test_split_round_trips_through_json(corpus, tmp_path):
    manifest, _ = corpus
    split = split_by_clique(manifest, seed=3)
    path = tmp_path / "splits.json"
    split.to_json(path)
    assert Split.from_json(path) == split


def test_collection_includes_distractors_but_they_are_not_queries(corpus):
    """15000 items but 13000 queries is the real dataset's shape; mirror it here."""
    manifest, _ = corpus
    full = collection(manifest)
    assert len(full) == len(manifest)
    assert query_mask(full).sum() == 8 * 4
    assert (~query_mask(full)).sum() == 6


def test_collection_without_distractors(corpus):
    manifest, _ = corpus
    only_real = collection(manifest, with_distractors=False)
    assert query_mask(only_real).all()


def test_subsample_draws_whole_cliques_only(corpus):
    manifest, _ = corpus
    sub = subsample_cliques(manifest, n_cliques=3, seed=0)
    assert sub["clique_id"].nunique() == 3
    assert (sub["clique_id"].value_counts() == 4).all()
    assert query_mask(sub).all()


def test_subsample_rejects_more_cliques_than_exist(corpus):
    manifest, _ = corpus
    with pytest.raises(ValueError, match="only 8 exist"):
        subsample_cliques(manifest, n_cliques=99)
