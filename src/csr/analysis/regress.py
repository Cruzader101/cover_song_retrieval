"""Which transformation hurts, holding the query fixed.

A cover pair's rank depends on two things at once: how hard the transformation
between the two performances was, and how hard the query is to begin with. A query
with a muddy chroma ranks all twelve of its covers badly, whatever key they are in,
and a plain comparison of ranks across pairs would attribute that to whichever
factor happens to correlate with it.

So every regression here is *within* a query. Each query's own mean is subtracted
from its twelve rows, and what identifies a coefficient is only the comparison of
one query's covers against each other: of these twelve, did the ones further from
the original key land further down? Anything about the query itself -- its length,
its recording quality, how distinctive its progression is -- is gone before the
first coefficient is estimated, whether or not we thought to measure it.

Standard errors are clustered on the **clique**, one level coarser than the query.
Two rows of one clique are dependent in ways the query effect does not absorb: every
pair appears twice, once in each direction, and both directions live in the same
clique. Clustering there allows arbitrary correlation within a work.

A note on the linear model for a binary outcome. Fixed-effects logit would drop
every query whose covers were all hits or all misses, which at twelve rows per query
and a hit rate near a tenth is most of the sample. A linear probability model keeps
them, and its coefficient reads directly as a change in the chance of landing in the
top ten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Fit:
    """A within-group fit with cluster-robust standard errors."""

    names: tuple[str, ...]
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_obs: int
    n_groups: int
    n_clusters: int
    r2_within: float

    def table(self) -> str:  # pragma: no cover - formatting only
        width = max(len(n) for n in self.names)
        lines = [f"{'factor':<{width}}  {'coef':>9}  {'se':>8}  {'p':>7}  95% CI"]
        for i, name in enumerate(self.names):
            lines.append(
                f"{name:<{width}}  {self.coef[i]:>9.4f}  {self.se[i]:>8.4f}  "
                f"{self.p[i]:>7.4f}  [{self.ci_low[i]:.4f}, {self.ci_high[i]:.4f}]"
            )
        lines.append(
            f"n={self.n_obs}  queries={self.n_groups}  cliques={self.n_clusters}  "
            f"within R2={self.r2_within:.4f}"
        )
        return "\n".join(lines)


def _codes(labels) -> tuple[np.ndarray, int]:
    codes = np.unique(np.asarray(labels), return_inverse=True)[1]
    return codes.astype(np.intp, copy=False), int(codes.max()) + 1


def demean(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Subtract each group's own mean. Accepts a vector or a column matrix.

    This is the fixed effect. Absorbing it by subtraction rather than by adding a
    dummy per query matters at this scale: the full benchmark has 13000 queries, and
    a design matrix with 13000 extra columns would be 13000 times larger than the
    one that produces identical coefficients.
    """
    values = np.asarray(values, dtype=np.float64)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    if values.ndim == 1:
        means = np.bincount(codes, weights=values, minlength=n_groups) / counts
        return values - means[codes]

    out = np.empty_like(values)
    for column in range(values.shape[1]):
        means = (
            np.bincount(codes, weights=values[:, column], minlength=n_groups) / counts
        )
        out[:, column] = values[:, column] - means[codes]
    return out


def within_ols(
    y,
    X,
    groups,
    clusters,
    names: tuple[str, ...] | list[str],
    alpha: float = 0.05,
) -> Fit:
    """Fit `y` on `X` within `groups`, with standard errors clustered on `clusters`.

    Args:
        y: outcome, one row per (query, cover) pair.
        X: `(n_obs, n_factors)` covariates. No intercept column -- the group means
            absorb it, and including one would make the demeaned matrix singular.
        groups: what to hold fixed. The query, here.
        clusters: what to allow correlation within. The clique, here.
        names: one label per column of `X`, for the reported table.

    Raises:
        ValueError: if the demeaned design is rank deficient. That usually means a
            factor is constant within every group, or a set of dummies was passed
            without dropping a reference level.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    names = tuple(names)
    if X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] != len(names):
        raise ValueError(
            f"X must be (len(y), len(names)) = ({len(y)}, {len(names)}), "
            f"got {X.shape}"
        )

    group_codes, n_groups = _codes(groups)
    cluster_codes, n_clusters = _codes(clusters)

    y_within = demean(y, group_codes, n_groups)
    x_within = demean(X, group_codes, n_groups)

    n_obs, n_factors = x_within.shape
    if np.linalg.matrix_rank(x_within) < n_factors:
        raise ValueError(
            f"the within-group design is rank deficient ({n_factors} columns, rank "
            f"{np.linalg.matrix_rank(x_within)}). A factor that never varies inside "
            "a group carries no within information, and a full set of dummies needs "
            "a reference level dropped."
        )

    xtx = x_within.T @ x_within
    xtx_inv = np.linalg.inv(xtx)
    coef = xtx_inv @ (x_within.T @ y_within)
    residuals = y_within - x_within @ coef

    # Liang-Zeger: sum the score contributions inside each cluster before squaring,
    # so correlation between two rows of one clique is carried rather than assumed
    # away. bincount per column beats np.add.at by a wide margin here.
    scores = x_within * residuals[:, None]
    totals = np.stack(
        [
            np.bincount(cluster_codes, weights=scores[:, j], minlength=n_clusters)
            for j in range(n_factors)
        ],
        axis=1,
    )
    meat = totals.T @ totals

    # The absorbed group means cost a degree of freedom each, so they are counted in
    # the parameter total even though they never appear as columns.
    n_params = n_factors + n_groups
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
    covariance = xtx_inv @ meat @ xtx_inv * correction

    se = np.sqrt(np.diag(covariance))
    t = coef / se
    # Inference on the number of clusters, not the number of rows: 50 cliques is 50
    # independent pieces of evidence however many pairs they contain.
    df = n_clusters - 1
    p = 2.0 * stats.t.sf(np.abs(t), df)
    half_width = stats.t.ppf(1.0 - alpha / 2.0, df) * se

    total_ss = float((y_within**2).sum())
    return Fit(
        names=names,
        coef=coef,
        se=se,
        t=t,
        p=p,
        ci_low=coef - half_width,
        ci_high=coef + half_width,
        n_obs=int(n_obs),
        n_groups=int(n_groups),
        n_clusters=int(n_clusters),
        r2_within=float(1.0 - (residuals**2).sum() / total_ss) if total_ss else 0.0,
    )


def bootstrap_coefficients(
    y,
    X,
    groups,
    clusters,
    names: tuple[str, ...] | list[str],
    n_boot: int = 999,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile intervals for the coefficients, resampling whole cliques.

    The sandwich above is asymptotic in the number of clusters, and one collection
    here has only fifty. This resamples cliques instead and refits, which is slower
    but does not lean on that limit.

    A clique drawn twice becomes two separate cliques, and its queries two separate
    sets of queries. Pooling the duplicate under one fixed effect would demean it
    against itself and understate the spread -- the thing being estimated.

    Returns:
        (ci_low, ci_high), each `(n_factors,)`.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    group_codes, _ = _codes(groups)
    cluster_codes, n_clusters = _codes(clusters)

    order = np.argsort(cluster_codes, kind="stable")
    starts = np.searchsorted(cluster_codes[order], np.arange(n_clusters + 1))
    members = [order[starts[c] : starts[c + 1]] for c in range(n_clusters)]

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        rows = np.concatenate([members[c] for c in drawn])
        # Fresh ids so a repeated clique is treated as a distinct one.
        offsets = np.repeat(
            np.arange(n_clusters) * (group_codes.max() + 1),
            [len(members[c]) for c in drawn],
        )
        try:
            fit = within_ols(
                y[rows], X[rows], group_codes[rows] + offsets, rows, names, alpha
            )
        except (ValueError, np.linalg.LinAlgError):
            continue  # a draw that lost a factor's variation carries no information
        samples.append(fit.coef)

    stacked = np.stack(samples)
    return (
        np.quantile(stacked, alpha / 2, axis=0),
        np.quantile(stacked, 1 - alpha / 2, axis=0),
    )
