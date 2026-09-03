"""Retrieval metrics for cover song identification.

Conventions used throughout (these match the Da-TACOS / MIREX cover song task):

* A *query* is one performance. The ranking it produces covers every other
  performance in the collection; the query itself is always excluded.
* An item is *relevant* iff it shares a clique with the query.
* Ranks are 1-indexed: the nearest non-self item has rank 1.
* Ties are broken by ascending item index (``numpy.argsort`` with a stable kind),
  which is deterministic but arbitrary. A method that produces many exact ties is
  getting flattered by this - check ``tie_fraction`` if a number looks too good.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

__all__ = ["RetrievalResults", "rankings_from_distances", "evaluate"]


@dataclass(frozen=True)
class RetrievalResults:
    """Aggregate retrieval scores over a query set."""

    mean_average_precision: float
    mean_rank_first_correct: float
    precision_at_10: float
    n_queries: int
    tie_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"MAP   {self.mean_average_precision:.4f}\n"
            f"MR1   {self.mean_rank_first_correct:.2f}\n"
            f"P@10  {self.precision_at_10:.4f}\n"
            f"(n_queries={self.n_queries}, ties={self.tie_fraction:.3%})"
        )


def rankings_from_distances(
    distances: np.ndarray,
    query_index: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Rank collection items per query, nearest first, excluding each query itself.

    Args:
        distances: ``(n_queries, n_items)`` float array. Smaller means more similar.
            Non-finite entries are pushed to the end of the ranking.
        query_index: for each query row, the column in ``distances`` holding that
            same performance, so it can be excluded. ``None`` means the matrix is
            square and the diagonal is the self-comparison.

    Returns:
        ``(n_queries, n_items - 1)`` int array of column indices, nearest first.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError(f"distances must be 2-D, got shape {distances.shape}")
    n_queries, n_items = distances.shape

    if query_index is None:
        if n_queries != n_items:
            raise ValueError(
                "query_index=None requires a square matrix; "
                f"got {n_queries}x{n_items}"
            )
        query_index = np.arange(n_queries)
    query_index = np.asarray(query_index, dtype=np.intp)
    if query_index.shape != (n_queries,):
        raise ValueError(
            f"query_index must have shape ({n_queries},), got {query_index.shape}"
        )

    work = np.where(np.isfinite(distances), distances, np.inf)
    work[np.arange(n_queries), query_index] = np.inf  # self never ranks

    order = np.argsort(work, axis=1, kind="stable")
    # The self item now sits somewhere in the trailing inf block; drop it explicitly
    # rather than assuming it landed last, since other entries may also be inf.
    keep = order != query_index[:, None]
    return order[keep].reshape(n_queries, n_items - 1)


def _adjacent_tie_fraction(distances: np.ndarray, ranking: np.ndarray) -> float:
    """Fraction of adjacent pairs in the rankings that are exact distance ties.

    A high value means the ordering is being decided by index order rather than by
    the method, which inflates every metric here. Non-finite distances are ignored.
    """
    if ranking.shape[1] < 2:
        return 0.0
    sorted_d = np.take_along_axis(distances, ranking, axis=1)
    left, right = sorted_d[:, :-1], sorted_d[:, 1:]
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return 0.0
    return float((valid & (left == right)).sum() / valid.sum())


def evaluate(
    distances: np.ndarray,
    cliques: Sequence,
    query_index: Sequence[int] | np.ndarray | None = None,
    k: int = 10,
) -> RetrievalResults:
    """Score a distance matrix against ground-truth clique labels.

    Args:
        distances: ``(n_queries, n_items)``, smaller = more similar.
        cliques: length ``n_items`` clique label per collection item. Hashable
            labels of any type; equality defines relevance.
        query_index: column of each query within the collection (see
            :func:`rankings_from_distances`).
        k: cutoff for precision@k. Defaults to 10.

    Raises:
        ValueError: if a query has no relevant items in the collection. That is a
            dataset construction bug, not a score of zero, so it fails loudly.
    """
    distances = np.asarray(distances, dtype=np.float64)
    n_queries, n_items = distances.shape

    labels = np.asarray(cliques)
    if labels.shape != (n_items,):
        raise ValueError(
            f"cliques must have length {n_items} (n_items), got {labels.shape}"
        )

    if query_index is None:
        if n_queries != n_items:
            raise ValueError("query_index=None requires a square distance matrix")
        query_index = np.arange(n_queries)
    query_index = np.asarray(query_index, dtype=np.intp)

    ranking = rankings_from_distances(distances, query_index)
    query_labels = labels[query_index]
    relevant = labels[ranking] == query_labels[:, None]  # (n_queries, n_items-1)

    n_relevant = relevant.sum(axis=1)
    empty = np.flatnonzero(n_relevant == 0)
    if empty.size:
        raise ValueError(
            f"{empty.size} quer{'y has' if empty.size == 1 else 'ies have'} no "
            f"relevant item in the collection (first at row {int(empty[0])}, "
            f"clique {query_labels[empty[0]]!r}). Filter singleton cliques out of "
            "the query set instead of scoring them as zero."
        )

    ranks = np.arange(1, ranking.shape[1] + 1, dtype=np.float64)

    # Average precision: mean of precision@rank over the ranks holding a hit.
    hits_so_far = np.cumsum(relevant, axis=1)
    precision_at_rank = hits_so_far / ranks
    ap = (precision_at_rank * relevant).sum(axis=1) / n_relevant

    # MR1: 1-indexed rank of the first hit. argmax finds the first True.
    r1 = relevant.argmax(axis=1) + 1

    kk = min(k, relevant.shape[1])
    p_at_k = relevant[:, :kk].sum(axis=1) / kk

    tie_fraction = _adjacent_tie_fraction(distances, ranking)

    return RetrievalResults(
        mean_average_precision=float(ap.mean()),
        mean_rank_first_correct=float(r1.mean()),
        precision_at_10=float(p_at_k.mean()),
        n_queries=int(n_queries),
        tie_fraction=tie_fraction,
    )
