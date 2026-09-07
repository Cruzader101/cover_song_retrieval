"""Tests for the clustered bootstrap and the multiple-comparison correction.

Passing `clusters=arange(n)` makes every row its own cluster, which is an ordinary
row-level bootstrap. That gives these tests a reference point: the clustered
interval is compared against the naive one it exists to replace.
"""

import numpy as np
import pytest

from csr.analysis.stats import (
    cluster_bootstrap,
    holm,
    paired_cluster_bootstrap,
)


def clustered_values(n_clusters=50, per_cluster=20, between=1.0, within=0.1, seed=0):
    """Values dominated by which cluster they came from, as cover cliques are.

    A work is easy or hard as a whole, so most of the variance sits between cliques
    and very little within one.
    """
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, between, n_clusters)
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    values = offsets[clusters] + rng.normal(0.0, within, n_clusters * per_cluster)
    return values, clusters


def test_estimate_is_the_plain_mean():
    values, clusters = clustered_values()
    assert cluster_bootstrap(values, clusters).estimate == pytest.approx(values.mean())


def test_identical_values_give_a_zero_width_interval():
    """No variation, no uncertainty -- whichever cliques are drawn, the mean is 7."""
    values = np.full(100, 7.0)
    clusters = np.repeat(np.arange(10), 10)
    interval = cluster_bootstrap(values, clusters)
    assert interval.low == pytest.approx(7.0)
    assert interval.high == pytest.approx(7.0)


def test_clustered_data_gives_a_wider_interval_than_treating_rows_as_independent():
    """The entire reason this module exists.

    1000 values in 50 cliques, where nearly all the variance is between cliques.
    There are 50 independent observations here, not 1000, and a row-level bootstrap
    would claim an interval about sqrt(1000/50) ~ 4.5x too narrow.
    """
    values, clusters = clustered_values()

    clustered = cluster_bootstrap(values, clusters)
    as_if_independent = cluster_bootstrap(values, np.arange(len(values)))

    clustered_width = clustered.high - clustered.low
    naive_width = as_if_independent.high - as_if_independent.low
    assert clustered_width > 3.0 * naive_width


def test_a_fixed_seed_is_reproducible_and_a_different_one_is_not():
    values, clusters = clustered_values()
    assert cluster_bootstrap(values, clusters, seed=0) == cluster_bootstrap(
        values, clusters, seed=0
    )
    assert cluster_bootstrap(values, clusters, seed=1) != cluster_bootstrap(
        values, clusters, seed=0
    )


def test_interval_covers_the_true_mean_of_a_known_population():
    """50 cliques drawn around a mean of 5.0; the 95% interval should contain it."""
    rng = np.random.default_rng(3)
    offsets = 5.0 + rng.normal(0.0, 1.0, 50)
    clusters = np.repeat(np.arange(50), 12)
    values = offsets[clusters] + rng.normal(0.0, 0.2, 600)

    interval = cluster_bootstrap(values, clusters)
    assert interval.low <= 5.0 <= interval.high


def test_unequal_cluster_sizes_weight_by_query_count():
    """A clique with more queries carries more of the mean, as it does in the report.

    Cluster A has 9 rows of 0.0 and cluster B has 1 row of 10.0, so the mean is 1.0,
    not the 5.0 an average-of-cluster-means would give.
    """
    values = np.array([0.0] * 9 + [10.0])
    clusters = np.array(["A"] * 9 + ["B"])
    assert cluster_bootstrap(values, clusters).estimate == pytest.approx(1.0)


def test_paired_bootstrap_of_a_constant_difference():
    """Method B beats A by exactly 0.1 on every query: no uncertainty in the gap.

    Run separately the two intervals would overlap almost entirely, since both are
    dominated by which cliques were drawn. Pairing cancels that shared difficulty,
    which is why methods are compared this way and not by eyeballing two intervals.
    """
    values, clusters = clustered_values()
    better = values + 0.1

    interval = paired_cluster_bootstrap(better, values, clusters)
    assert interval.estimate == pytest.approx(0.1)
    assert interval.low == pytest.approx(0.1)
    assert interval.high == pytest.approx(0.1)


def test_paired_bootstrap_of_no_difference_cannot_reject():
    values, clusters = clustered_values()
    interval = paired_cluster_bootstrap(values, values, clusters)
    assert interval.estimate == pytest.approx(0.0)
    assert not interval.excludes(0.0)
    assert interval.p_value == pytest.approx(1.0)


def test_paired_bootstrap_requires_aligned_queries():
    with pytest.raises(ValueError, match="align query for query"):
        paired_cluster_bootstrap(np.zeros(10), np.zeros(9), np.arange(10))


def test_excludes_reads_the_interval():
    values = np.full(50, 2.0)
    interval = cluster_bootstrap(values, np.repeat(np.arange(10), 5))
    assert interval.excludes(0.0)
    assert not interval.excludes(2.0)


def test_holm_hand_computed():
    """m=3. Sorted p * (m - rank), then forced non-decreasing.

    .01 * 3 = .03
    .02 * 2 = .04
    .03 * 1 = .03 -> raised to .04, since a larger p cannot be more significant
    """
    assert holm([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_returns_in_input_order():
    """Same three values shuffled must give the same three answers, re-ordered."""
    assert holm([0.03, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_holm_caps_at_one():
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_holm_leaves_a_single_test_alone():
    assert holm([0.04]) == pytest.approx([0.04])
