"""Tests for the within-group regression and its cluster-robust standard errors.

Two kinds of test here. The first plants a coefficient in generated data and checks
it comes back. The second plants a *confound* -- a group-level effect correlated
with the factor -- and checks that the naive estimate is fooled by it and the
within-group one is not, which is the entire reason the fixed effect exists.
"""

import numpy as np
import pytest

from csr.analysis.regress import bootstrap_coefficients, demean, within_ols


def planted(n_groups=200, per_group=12, beta=(2.0, -0.5), noise=0.3, seed=0):
    """Rows where y is a known linear function of X plus a per-group offset."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    n = len(groups)
    X = rng.normal(0.0, 1.0, (n, len(beta)))
    offsets = rng.normal(0.0, 5.0, n_groups)
    y = X @ np.array(beta) + offsets[groups] + rng.normal(0.0, noise, n)
    return y, X, groups


def test_demean_removes_the_group_mean():
    """Each group's rows must sum to zero afterwards, and only the group's own."""
    values = np.array([1.0, 3.0, 10.0, 20.0])
    codes = np.array([0, 0, 1, 1])
    assert demean(values, codes, 2) == pytest.approx([-1.0, 1.0, -5.0, 5.0])


def test_demean_handles_a_matrix_column_by_column():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 100.0]])
    codes = np.array([0, 0, 1])
    out = demean(X, codes, 2)
    assert out == pytest.approx(np.array([[-1.0, -10.0], [1.0, 10.0], [0.0, 0.0]]))


def test_planted_coefficients_are_recovered():
    """beta = (2.0, -0.5) buried under group offsets five times larger."""
    y, X, groups = planted()
    fit = within_ols(y, X, groups, groups, ["a", "b"])
    assert fit.coef == pytest.approx([2.0, -0.5], abs=0.02)
    assert fit.n_obs == 2400
    assert fit.n_groups == 200


def test_the_fixed_effect_removes_a_planted_confound():
    """The test that justifies the whole design.

    Easy groups are given systematically smaller x. Pooled, that makes x look
    strongly protective -- groups with low x score well -- when within any single
    group x does nothing at all. The true within-group coefficient is 0.

    This is exactly the trap in the real data: works that are easy to retrieve may
    also be works whose covers stay in the original key.
    """
    rng = np.random.default_rng(1)
    n_groups, per_group = 300, 12
    groups = np.repeat(np.arange(n_groups), per_group)

    difficulty = rng.normal(0.0, 3.0, n_groups)
    # x is centred on the group's difficulty, so between groups the two move
    # together; inside a group x is pure noise and has no effect on y.
    x = difficulty[groups] + rng.normal(0.0, 1.0, n_groups * per_group)
    y = difficulty[groups] + rng.normal(0.0, 1.0, n_groups * per_group)
    X = x[:, None]

    naive = np.linalg.lstsq(
        np.column_stack([np.ones_like(x), x]), y, rcond=None
    )[0][1]
    within = within_ols(y, X, groups, groups, ["x"])

    assert naive > 0.5  # fooled: it sees the between-group correlation
    assert within.coef[0] == pytest.approx(0.0, abs=0.05)
    assert not within.ci_low[0] <= naive <= within.ci_high[0]


def test_clustered_errors_are_wider_than_pretending_rows_are_independent():
    """Correlated residuals that survive the fixed effect must widen the interval.

    Built with the real shape of this data: a clique of members, one row per
    *ordered* pair, the query fixed effect on the first member. A shock attached to
    the second member -- one particular cover being hard to find from every
    direction -- is not constant within a query, so demeaning cannot remove it, and
    it leaves every row of a clique correlated.

    That is what clustering on the clique is for. Clustering on the row index
    instead is the heteroskedasticity-robust estimate, which assumes it away.
    """
    rng = np.random.default_rng(2)
    n_cliques, members = 40, 5

    clusters, groups, pairs = [], [], []
    for clique in range(n_cliques):
        for i in range(members):
            for j in range(members):
                if i != j:
                    clusters.append(clique)
                    groups.append(clique * members + i)
                    # Same id for (i,j) and (j,i): a transformation between two
                    # performances is the same transformation read either way.
                    pairs.append((clique, min(i, j), max(i, j)))
    clusters = np.array(clusters)
    groups = np.array(groups)
    pair_ids = np.unique(np.array(pairs), axis=0, return_inverse=True)[1]

    n = len(clusters)
    # Both the factor and the shock have a clique-level component, which is what
    # makes clustering bite: an inflated interval needs the regressor to be
    # correlated within the cluster too, not just the errors.
    x = rng.normal(0.0, 1.0, n_cliques)[clusters] + rng.normal(
        0.0, 0.5, pair_ids.max() + 1
    )[pair_ids]
    shock = rng.normal(0.0, 2.0, n_cliques)[clusters] + rng.normal(
        0.0, 1.0, pair_ids.max() + 1
    )[pair_ids]
    X = x[:, None]
    y = x + shock + rng.normal(0.0, 0.2, n)

    clustered = within_ols(y, X, groups, clusters, ["x"])
    by_row = within_ols(y, X, groups, np.arange(n), ["x"])
    assert clustered.se[0] > 1.5 * by_row.se[0]


def test_rank_deficient_design_raises():
    """A duplicated column, which is what a full set of dummies looks like."""
    y, X, groups = planted()
    doubled = np.column_stack([X[:, 0], X[:, 0]])
    with pytest.raises(ValueError, match="rank deficient"):
        within_ols(y, doubled, groups, groups, ["a", "a_again"])


def test_a_factor_constant_within_every_group_raises():
    """Group-level factors carry no within information and cannot be estimated."""
    y, X, groups = planted()
    constant = np.repeat(np.arange(200) * 1.0, 12)[:, None]
    with pytest.raises(ValueError, match="rank deficient"):
        within_ols(y, constant, groups, groups, ["group_level"])


def test_mismatched_names_raise():
    y, X, groups = planted()
    with pytest.raises(ValueError, match="X must be"):
        within_ols(y, X, groups, groups, ["only_one"])


def test_within_r2_is_zero_when_x_explains_nothing():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(100), 10)
    y = rng.normal(0.0, 1.0, 1000)
    X = rng.normal(0.0, 1.0, (1000, 1))
    fit = within_ols(y, X, groups, groups, ["noise"])
    assert fit.r2_within == pytest.approx(0.0, abs=0.01)


def test_bootstrap_intervals_cover_the_planted_coefficients():
    """The few-cluster path must land in the same place as the sandwich."""
    y, X, groups = planted(n_groups=50)
    low, high = bootstrap_coefficients(y, X, groups, groups, ["a", "b"], n_boot=199)
    assert low[0] <= 2.0 <= high[0]
    assert low[1] <= -0.5 <= high[1]
