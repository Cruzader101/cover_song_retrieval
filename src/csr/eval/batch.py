"""Scoring a distance matrix without holding several copies of it.

`csr.eval.metrics.evaluate` casts to float64 and argsorts, so on the full
13000 x 15000 matrix it needs several gigabytes at once. Slicing the query axis
fixes that, and the arithmetic stays exact: MAP, MR1 and P@k are all means over
queries, so a query-count-weighted average of per-chunk means is the same number.

`metrics.py` holds the tested contract; this only calls it and re-aggregates.
"""

from __future__ import annotations

import numpy as np

from csr.eval.metrics import RetrievalResults, pair_ranks, per_query


def evaluate_chunked_pairs(
    distances: np.ndarray,
    cliques,
    query_index=None,
    k: int = 10,
    chunk_size: int = 2048,
) -> tuple[RetrievalResults, np.ndarray]:
    """Scores and the per-pair table, from a single pass over the queries.

    The pair table is where every true cover landed, `(n_pairs, 3)` int32; see
    `metrics.pair_ranks`. It is tiny next to the ranking it comes from -- twelve
    rows per query against a row of 15000 -- so it is accumulated rather than
    recomputed, which would mean a second argsort over the whole collection.

    `tie_fraction` is exact when every distance is finite. With non-finite entries
    the per-chunk valid-pair counts differ slightly, so it becomes a close
    approximation; it is a diagnostic, not a reported metric.
    """
    distances = np.asarray(distances)
    n_queries = distances.shape[0]
    if query_index is None:
        query_index = np.arange(n_queries)
    query_index = np.asarray(query_index, dtype=np.intp)

    total = {"map": 0.0, "mr1": 0.0, "p_at_k": 0.0, "ties": 0.0}
    pairs = []
    for start in range(0, n_queries, chunk_size):
        stop = min(start + chunk_size, n_queries)
        detail = per_query(
            distances[start:stop], cliques, query_index[start:stop], k=k
        )
        # pair_ranks reports collection indices, which query_index already carries,
        # so chunk blocks concatenate without an offset correction.
        pairs.append(pair_ranks(detail))

        part = detail.summary()
        weight = stop - start
        total["map"] += part.mean_average_precision * weight
        total["mr1"] += part.mean_rank_first_correct * weight
        total["p_at_k"] += part.precision_at_10 * weight
        total["ties"] += part.tie_fraction * weight

    results = RetrievalResults(
        mean_average_precision=total["map"] / n_queries,
        mean_rank_first_correct=total["mr1"] / n_queries,
        precision_at_10=total["p_at_k"] / n_queries,
        n_queries=n_queries,
        tie_fraction=total["ties"] / n_queries,
    )
    return results, np.concatenate(pairs)


def evaluate_chunked(
    distances: np.ndarray,
    cliques,
    query_index=None,
    k: int = 10,
    chunk_size: int = 2048,
) -> RetrievalResults:
    """Same result as `evaluate`, computed a block of queries at a time."""
    return evaluate_chunked_pairs(distances, cliques, query_index, k, chunk_size)[0]
