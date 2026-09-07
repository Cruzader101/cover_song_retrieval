"""Confidence intervals that respect how these queries are grouped.

The unit that repeats in this dataset is the clique, not the query. One work
contributes thirteen performances, all of them queries, all ranking each other, and
they succeed and fail together: a work with a distinctive progression is easy from
every direction, and a work built on two chords is hard from every direction. Two
queries from one clique are therefore nowhere near independent draws.

Treating 13000 queries as 13000 independent observations would shrink every interval
by roughly the square root of thirteen and turn ordinary between-work variation into
significance. So everything here resamples **cliques**, with replacement, and lets
each drawn clique bring all of its queries along.

What this does and does not cover: it is inference about the query sample, given a
fixed collection. The 2000 singleton distractors never move, and neither does which
performances exist. It says how much a reported number would wobble if the works had
been drawn differently, not how it would move on another dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Every metric in this project is a mean over queries -- MAP, MR1 and P@10 all are
#: -- so the resample only ever needs per-cluster sums and counts. That turns a
#: bootstrap over 156000 rows into arithmetic on 1000 of them.
DEFAULT_N_BOOT = 2000


@dataclass(frozen=True)
class Interval:
    """A point estimate and a percentile bootstrap interval around it."""

    estimate: float
    low: float
    high: float
    n_boot: int
    alpha: float
    p_value: float

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"

    def excludes(self, value: float) -> bool:
        """Whether `value` falls outside the interval.

        The honest way to read a null: `excludes(0)` is evidence of an effect, but
        failing to exclude 0 is not evidence of no effect. For that, ask whether the
        interval excludes the smallest effect you would have cared about.
        """
        return not (self.low <= value <= self.high)


def _codes(clusters) -> tuple[np.ndarray, int]:
    codes = np.unique(np.asarray(clusters), return_inverse=True)[1]
    return codes.astype(np.intp, copy=False), int(codes.max()) + 1


def _draws(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """`(n_boot, n_clusters)` cluster ids, drawn with replacement."""
    return np.random.default_rng(seed).integers(
        0, n_clusters, size=(n_boot, n_clusters)
    )


def _resampled_means(
    values: np.ndarray, codes: np.ndarray, n_clusters: int, draws: np.ndarray
) -> np.ndarray:
    """Mean of `values` under each row of `draws`. -> `(n_boot,)`.

    A clique drawn twice contributes twice, so the weights are its summed values and
    its query count, not its mean -- a clique with more queries carries more of the
    average, exactly as it does in the reported number.
    """
    sums = np.bincount(codes, weights=values, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def _interval(samples: np.ndarray, estimate: float, alpha: float, seed_n: int) -> Interval:
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    # Two-sided: the proportion of resamples on the far side of zero, doubled. The
    # useful reading is for a difference, where zero means "these two methods scored
    # the same on the works we happened to draw".
    # Both tails are counted inclusively. Using 1 - P(<= 0) for the upper tail would
    # call a difference of exactly zero significant, because no resample is strictly
    # above it.
    below = float(np.mean(samples <= 0.0))
    above = float(np.mean(samples >= 0.0))
    p = min(2.0 * min(below, above), 1.0)
    return Interval(
        estimate=float(estimate),
        low=float(low),
        high=float(high),
        n_boot=seed_n,
        alpha=alpha,
        p_value=p,
    )


def cluster_bootstrap(
    values,
    clusters,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile interval for the mean of `values`, resampling whole clusters.

    Args:
        values: one number per query -- an average precision, a rank, a hit.
        clusters: that query's clique. Equal labels move together.
    """
    values = np.asarray(values, dtype=np.float64)
    codes, n_clusters = _codes(clusters)
    draws = _draws(n_clusters, n_boot, seed)
    samples = _resampled_means(values, codes, n_clusters, draws)
    return _interval(samples, values.mean(), alpha, n_boot)


def paired_cluster_bootstrap(
    first,
    second,
    clusters,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Interval for `mean(first) - mean(second)` on the same queries.

    Both methods are scored on the *same* resampled cliques, so the shared
    difficulty of the works cancels instead of being counted twice as noise. Two
    separate intervals that happen to overlap say much less than this does: methods
    are compared here on the queries they both answered.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"paired inputs must align query for query, got {first.shape} and "
            f"{second.shape}"
        )
    codes, n_clusters = _codes(clusters)
    draws = _draws(n_clusters, n_boot, seed)
    samples = _resampled_means(first, codes, n_clusters, draws) - _resampled_means(
        second, codes, n_clusters, draws
    )
    return _interval(samples, first.mean() - second.mean(), alpha, n_boot)


def holm(pvalues) -> np.ndarray:
    """Holm step-down family-wise correction. -> adjusted p-values, input order.

    Asking eight questions of one dataset and reporting the best answer is how a
    null result becomes a finding. Holm is uniformly more powerful than Bonferroni
    and assumes nothing about how the tests are related, which matters here because
    they very much are related.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted
