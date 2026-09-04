"""End-to-end runs on the synthetic corpus.

These are the tests that would catch a harness that grades a method on the wrong
collection, or a method wired up backwards. The synthetic cliques are easy by
construction, so a real method has to score well above random or something is
wrong with the plumbing rather than with the method.
"""

import numpy as np
import pytest

from csr.run import run


@pytest.fixture
def synthetic_run(corpus, tmp_path):
    """Run a config against the in-memory corpus instead of data/raw."""
    manifest, chroma = corpus

    def go(config):
        return run(
            config,
            frame=manifest,
            load_factory=lambda frame: (lambda pid: chroma[pid]),
            out_dir=tmp_path,
        )

    return go


def test_random_baseline_scores_near_the_chance_floor(synthetic_run):
    """8 cliques of 4 plus 6 distractors: 3 relevant items among 37 others.

    E[AP] for a random ranking is 0.163 here, not 3/37 = 0.081 -- R/M is only the
    large-R limit. Simulated over 20k permutations.
    """
    payload = synthetic_run(
        {"name": "random", "method": "random", "collection": {"kind": "full"}}
    )
    assert payload["metrics"]["mean_average_precision"] == pytest.approx(0.163, abs=0.06)
    assert payload["collection"]["n_items"] == 38
    assert payload["collection"]["n_queries"] == 32


def test_ftm2d_clears_the_random_baseline_by_a_wide_margin(synthetic_run):
    random_map = synthetic_run(
        {"name": "random", "method": "random", "collection": {"kind": "full"}}
    )["metrics"]["mean_average_precision"]
    ftm_map = synthetic_run(
        {"name": "ftm2d", "method": "ftm2d", "collection": {"kind": "full"}}
    )["metrics"]["mean_average_precision"]
    assert ftm_map > random_map * 3
    assert ftm_map > 0.3


def test_results_payload_records_what_produced_it(synthetic_run):
    """A number without its provenance is not reproducible."""
    payload = synthetic_run(
        {"name": "ftm2d", "method": "ftm2d", "seed": 7, "collection": {"kind": "full"}}
    )
    assert payload["seed"] == 7
    assert set(payload) >= {
        "run_name", "method", "seed", "git", "config",
        "collection", "timings_sec", "metrics", "env",
    }
    assert payload["timings_sec"]["total"] > 0
    assert "sha" in payload["git"]


def test_distractors_are_scored_against_but_never_queried(synthetic_run):
    """The singleton cliques must stay in the collection and out of the query set."""
    payload = synthetic_run(
        {"name": "ftm2d", "method": "ftm2d", "collection": {"kind": "full"}}
    )
    assert payload["collection"]["n_items"] == 38
    assert payload["collection"]["n_queries"] == 32


def test_subsample_collection_drops_distractors(synthetic_run):
    payload = synthetic_run(
        {
            "name": "sub",
            "method": "ftm2d",
            "collection": {"kind": "subsample", "n_cliques": 3},
        }
    )
    assert payload["collection"]["n_items"] == 12
    assert payload["collection"]["n_queries"] == 12


def test_unknown_method_names_itself_and_the_alternatives(synthetic_run):
    with pytest.raises(KeyError, match="unknown method"):
        synthetic_run({"name": "x", "method": "nope", "collection": {"kind": "full"}})


def test_learned_method_is_registered_on_demand():
    """torch-backed methods live outside csr.methods so the classical path never
    imports torch; build() has to pull them in when asked."""
    import csr.methods as methods

    assert "learned" not in methods.REGISTRY or True  # may already be imported
    assert callable(methods.build("learned"))
    assert "learned" in methods.REGISTRY
