"""Known-answer tests for the per-pair layer under the aggregate metrics.

Every expected number below is hand-computed in the docstring of its test. If a
test fails, check the arithmetic in the docstring before changing the assertion.

The point of this layer is that a pair table and the MAP reported for the same run
can never disagree, so several of these tests rebuild an aggregate from the pairs
alone and compare it to `evaluate`.
"""

import numpy as np
import pytest

from csr.eval.batch import evaluate_chunked_pairs
from csr.eval.metrics import evaluate, pair_ranks, per_query

MIXED = np.array(
    [[0, 1, 2, 3], [1, 0, 2, 3], [2, 2, 0, 5], [3, 3, 5, 0]], dtype=float
)
CLIQUES = [0, 0, 1, 1]


def test_pair_rows_hand_computed():
    """Same matrix as test_mixed_ranking_hand_computed, cliques {0,1}|{2,3}.

    q0 -> ranks [1,2,3], relevant item 1 sits at rank 1  -> (0, 1, 1)
    q1 -> ranks [0,2,3], relevant item 0 sits at rank 1  -> (1, 0, 1)
    q2 -> ranks [0,1,3], relevant item 3 sits at rank 3  -> (2, 3, 3)
    q3 -> ranks [0,1,2], relevant item 2 sits at rank 3  -> (3, 2, 3)
    """
    pairs = pair_ranks(per_query(MIXED, CLIQUES))
    assert pairs.tolist() == [[0, 1, 1], [1, 0, 1], [2, 3, 3], [3, 2, 3]]


def test_every_relevant_item_appears_exactly_once():
    """Six items in two cliques of three: each query owns 2 pairs, 12 in total."""
    d = np.array(
        [
            [0.0, 1.0, 2.0, 7.0, 8.0, 9.0],
            [1.0, 0.0, 3.0, 7.0, 8.0, 9.0],
            [2.0, 3.0, 0.0, 7.0, 8.0, 9.0],
            [7.0, 8.0, 9.0, 0.0, 1.0, 2.0],
            [7.0, 8.0, 9.0, 1.0, 0.0, 3.0],
            [7.0, 8.0, 9.0, 2.0, 3.0, 0.0],
        ]
    )
    cliques = ["A", "A", "A", "B", "B", "B"]
    pairs = pair_ranks(per_query(d, cliques))

    assert len(pairs) == 12
    # Every clique-mate of every query, once each, and never the query itself.
    for query in range(6):
        targets = sorted(pairs[pairs[:, 0] == query][:, 1].tolist())
        mates = [i for i in range(6) if cliques[i] == cliques[query] and i != query]
        assert targets == mates


def test_average_precision_rebuilt_from_pairs_alone():
    """AP is a function of the hit ranks, so the pair table must reproduce it.

    For one query, AP = mean over its hits of (hits so far / rank). Rebuilding it
    from nothing but (query, target, rank) rows and comparing to `evaluate` is what
    guarantees the analysis layer and the results table describe the same run.
    """
    rng = np.random.default_rng(0)
    n, per = 60, 6
    cliques = np.repeat(np.arange(n // per), per)
    d = rng.random((n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)

    pairs = pair_ranks(per_query(d, cliques))
    average_precisions = []
    for query in range(n):
        ranks = np.sort(pairs[pairs[:, 0] == query][:, 2])
        hits_so_far = np.arange(1, len(ranks) + 1)
        average_precisions.append(np.mean(hits_so_far / ranks))

    assert np.mean(average_precisions) == pytest.approx(
        evaluate(d, cliques).mean_average_precision
    )


def test_rank_of_first_pair_is_mr1():
    """MR1 keeps only the best rank per query; the pair table keeps all of them."""
    pairs = pair_ranks(per_query(MIXED, CLIQUES))
    first = [pairs[pairs[:, 0] == q][:, 2].min() for q in range(4)]
    assert np.mean(first) == pytest.approx(evaluate(MIXED, CLIQUES).mean_rank_first_correct)


@pytest.mark.parametrize("chunk", [1, 2, 3, 17, 5000])
def test_chunked_pairs_match_unchunked(chunk):
    """Chunking must not renumber a query. Rows are compared as a sorted set.

    This is the test that catches a missing offset correction: pair_ranks reports
    collection indices, so a chunk starting at query 40 must still say 40.
    """
    rng = np.random.default_rng(0)
    n, per = 60, 6
    cliques = np.repeat(np.arange(n // per), per)
    d = rng.random((n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)

    reference = pair_ranks(per_query(d, cliques))
    _, chunked = evaluate_chunked_pairs(d, cliques, chunk_size=chunk)

    order = np.lexsort((reference[:, 1], reference[:, 0]))
    chunked_order = np.lexsort((chunked[:, 1], chunked[:, 0]))
    assert np.array_equal(reference[order], chunked[chunked_order])


def test_chunked_pairs_carry_collection_indices_for_a_query_subset():
    """Queries 0 and 2 of a 4-item collection: rows must say 0 and 2, not 0 and 1."""
    d = np.array([[0.0, 1.0, 5.0, 6.0], [5.0, 5.0, 0.0, 1.0]])
    _, pairs = evaluate_chunked_pairs(
        d, [0, 0, 1, 1], query_index=[0, 2], chunk_size=1
    )
    assert pairs.tolist() == [[0, 1, 1], [2, 3, 1]]


def test_summary_is_exactly_what_evaluate_returns():
    """`evaluate` is defined as a reduction of this layer; assert it stays that way."""
    detail = per_query(MIXED, CLIQUES)
    assert detail.summary() == evaluate(MIXED, CLIQUES)


def test_clique_labels_of_any_type_agree():
    """Relevance compares integer codes internally; the label type must not matter."""
    as_strings = ["w1", "w1", "w2", "w2"]
    assert evaluate(MIXED, as_strings) == evaluate(MIXED, CLIQUES)
