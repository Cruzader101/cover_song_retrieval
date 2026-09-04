"""evaluate_chunked must agree with evaluate exactly.

The expected values come from the hand-computed matrices in test_metrics.py, so
this checks the chunking rather than re-deriving the metrics.
"""

import numpy as np
import pytest

from csr.eval.batch import evaluate_chunked
from csr.eval.metrics import evaluate

MIXED = np.array(
    [[0, 1, 2, 3], [1, 0, 2, 3], [2, 2, 0, 5], [3, 3, 5, 0]], dtype=float
)
CLIQUES = [0, 0, 1, 1]


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 1000])
def test_chunking_reproduces_the_hand_computed_result(chunk):
    """Same matrix as test_mixed_ranking_hand_computed: MAP 2/3, MR1 2.0, P@3 1/3."""
    r = evaluate_chunked(MIXED, CLIQUES, chunk_size=chunk)
    assert r.mean_average_precision == pytest.approx(2 / 3)
    assert r.mean_rank_first_correct == pytest.approx(2.0)
    assert r.precision_at_10 == pytest.approx(1 / 3)
    assert r.n_queries == 4


@pytest.mark.parametrize("chunk", [1, 2, 3, 17, 5000])
def test_chunking_matches_unchunked_on_a_larger_random_case(chunk):
    rng = np.random.default_rng(0)
    n, per = 60, 6
    cliques = np.repeat(np.arange(n // per), per)
    d = rng.random((n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)

    reference = evaluate(d, cliques)
    chunked = evaluate_chunked(d, cliques, chunk_size=chunk)
    assert chunked.mean_average_precision == pytest.approx(
        reference.mean_average_precision
    )
    assert chunked.mean_rank_first_correct == pytest.approx(
        reference.mean_rank_first_correct
    )
    assert chunked.precision_at_10 == pytest.approx(reference.precision_at_10)
    assert chunked.tie_fraction == pytest.approx(reference.tie_fraction)


def test_chunking_handles_a_non_square_query_subset():
    d = np.array([[0.0, 1.0, 5.0, 6.0], [5.0, 5.0, 0.0, 1.0]])
    r = evaluate_chunked(d, [0, 0, 1, 1], query_index=[0, 2], chunk_size=1)
    assert r.mean_average_precision == pytest.approx(1.0)
    assert r.n_queries == 2
