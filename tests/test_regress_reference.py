"""Cross-check the hand-rolled estimator against statsmodels.

`csr.analysis.regress` absorbs the group means by subtraction and builds its own
cluster-robust sandwich, which is about forty lines of numpy. This asserts those
forty lines agree with an established implementation doing the same thing the
expensive way: one explicit dummy column per group.

statsmodels is a dev dependency only. Nothing under src/ imports it, and the suite
passes without it installed.
"""

import numpy as np
import pytest

from csr.analysis.regress import within_ols

sm = pytest.importorskip("statsmodels.api", reason="statsmodels is a dev extra")

pytestmark = pytest.mark.needs_statsmodels


def nested_design(n_cliques=60, queries_per_clique=4, per_query=12, seed=0):
    """Rows nested the way the real data is: pairs in queries in cliques."""
    rng = np.random.default_rng(seed)
    n_groups = n_cliques * queries_per_clique
    groups = np.repeat(np.arange(n_groups), per_query)
    clusters = groups // queries_per_clique

    n = len(groups)
    X = rng.normal(0.0, 1.0, (n, 2))
    offsets = rng.normal(0.0, 4.0, n_groups)
    shock = rng.normal(0.0, 1.5, n_cliques)
    y = X @ np.array([1.7, -0.8]) + offsets[groups] + shock[clusters]
    y = y + rng.normal(0.0, 0.5, n)
    return y, X, groups, clusters


def reference(y, X, groups, clusters):
    """The same model with a dummy per group instead of a within transformation."""
    dummies = np.eye(len(np.unique(groups)))[np.unique(groups, return_inverse=True)[1]]
    design = np.column_stack([X, dummies])
    return sm.OLS(y, design).fit(
        cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": True}
    )


def test_coefficients_match_a_dummy_variable_regression():
    """Frisch-Waugh-Lovell: absorbing the group means must give identical slopes."""
    y, X, groups, clusters = nested_design()
    mine = within_ols(y, X, groups, clusters, ["a", "b"])
    theirs = reference(y, X, groups, clusters)
    assert mine.coef == pytest.approx(theirs.params[:2], rel=1e-9)


def test_cluster_robust_standard_errors_match():
    """Including the finite-sample correction, which is where conventions differ.

    Both apply G/(G-1) * (N-1)/(N-K) with K counting the absorbed group means, so
    these must agree to numerical precision rather than merely to a constant factor.
    """
    y, X, groups, clusters = nested_design()
    mine = within_ols(y, X, groups, clusters, ["a", "b"])
    theirs = reference(y, X, groups, clusters)
    assert mine.se == pytest.approx(theirs.bse[:2], rel=1e-9)


def test_within_r2_matches_the_dummy_regression_partial_r2():
    """1 - RSS/TSS on demeaned data is what the dummy model explains beyond groups."""
    y, X, groups, clusters = nested_design()
    mine = within_ols(y, X, groups, clusters, ["a", "b"])
    theirs = reference(y, X, groups, clusters)

    group_only = sm.OLS(
        y, np.eye(len(np.unique(groups)))[np.unique(groups, return_inverse=True)[1]]
    ).fit()
    expected = 1.0 - theirs.ssr / group_only.ssr
    assert mine.r2_within == pytest.approx(expected, rel=1e-9)


def test_a_single_factor_also_matches():
    """Guards against an error that only shows up with more than one column."""
    y, X, groups, clusters = nested_design(n_cliques=30)
    mine = within_ols(y, X[:, :1], groups, clusters, ["a"])
    theirs = reference(y, X[:, :1], groups, clusters)
    assert mine.coef == pytest.approx(theirs.params[:1], rel=1e-9)
    assert mine.se == pytest.approx(theirs.bse[:1], rel=1e-9)
