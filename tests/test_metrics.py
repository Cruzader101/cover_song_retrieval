"""Known-answer tests for the retrieval metrics.

Every expected number below is hand-computed in the docstring of its test. If a
test fails, check the arithmetic in the docstring before changing the assertion.
"""

import numpy as np
import pytest

from csr.eval.metrics import evaluate, rankings_from_distances

INF = np.inf


def test_perfect_ranking():
    """Cliques {0,1}|{2,3}; same-clique always nearest.

    Every query: first hit at rank 1, one relevant item -> AP = 1.0, R1 = 1.
    MAP = 1.0, MR1 = 1.0. Only 3 non-self items, so P@10 collapses to P@3 = 1/3.
    """
    d = np.array(
        [
            [0, 1, 5, 6],
            [1, 0, 5, 6],
            [5, 5, 0, 1],
            [6, 6, 1, 0],
        ],
        float,
    )
    r = evaluate(d, cliques=[0, 0, 1, 1])
    assert r.mean_average_precision == pytest.approx(1.0)
    assert r.mean_rank_first_correct == pytest.approx(1.0)
    assert r.precision_at_10 == pytest.approx(1 / 3)
    assert r.n_queries == 4


def test_mixed_ranking_hand_computed():
    """Queries 0,1 rank their partner first; queries 2,3 rank theirs last.

    q0 -> [1,2,3] rel [T,F,F]  AP = 1.0        R1 = 1
    q1 -> [0,2,3] rel [T,F,F]  AP = 1.0        R1 = 1
    q2 -> [0,1,3] rel [F,F,T]  AP = (1/3)/1    R1 = 3
    q3 -> [0,1,2] rel [F,F,T]  AP = (1/3)/1    R1 = 3
    MAP = (1 + 1 + 1/3 + 1/3)/4 = 2/3;  MR1 = (1+1+3+3)/4 = 2.0;  P@3 = 1/3
    """
    d = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 2, 0, 5],
            [3, 3, 5, 0],
        ],
        float,
    )
    r = evaluate(d, cliques=[0, 0, 1, 1])
    assert r.mean_average_precision == pytest.approx(2 / 3)
    assert r.mean_rank_first_correct == pytest.approx(2.0)
    assert r.precision_at_10 == pytest.approx(1 / 3)
    # q2 ties 0 vs 1 (both 2.0) and q3 ties 0 vs 1 (both 3.0): 2 tied pairs of 8.
    assert r.tie_fraction == pytest.approx(0.25)


def test_average_precision_with_two_relevant():
    """One query, clique A = {0,1,2}, clique B = {3,4,5}.

    Query 0 ranking by distance: 3 (1.0), 1 (2.0), 4 (3.0), 2 (4.0), 5 (5.0)
    relevance:                   F         T        F        T        F
    hits at ranks 2 and 4 -> AP = (1/2 + 2/4) / 2 = 0.5;  R1 = 2
    """
    d = np.array([[0.0, 2.0, 4.0, 1.0, 3.0, 5.0]])
    r = evaluate(d, cliques=["A", "A", "A", "B", "B", "B"], query_index=[0], k=10)
    assert r.mean_average_precision == pytest.approx(0.5)
    assert r.mean_rank_first_correct == pytest.approx(2.0)
    assert r.precision_at_10 == pytest.approx(2 / 5)  # k clipped to 5 non-self items


def test_non_square_matrix_with_query_index():
    """Two queries (items 0 and 2) scored against a 4-item collection."""
    d = np.array(
        [
            [0.0, 1.0, 5.0, 6.0],  # query is item 0
            [5.0, 5.0, 0.0, 1.0],  # query is item 2
        ]
    )
    r = evaluate(d, cliques=[0, 0, 1, 1], query_index=[0, 2])
    assert r.n_queries == 2
    assert r.mean_average_precision == pytest.approx(1.0)
    assert r.mean_rank_first_correct == pytest.approx(1.0)


def test_self_is_always_excluded_even_when_it_is_not_the_minimum():
    """Self-distance of 99 must still be dropped, not ranked last on merit."""
    d = np.array(
        [
            [99.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 2.0, 3.0],
            [2.0, 2.0, 0.0, 5.0],
            [3.0, 3.0, 5.0, 0.0],
        ]
    )
    ranking = rankings_from_distances(d)
    assert ranking.shape == (4, 3)
    for i, row in enumerate(ranking):
        assert i not in row


def test_non_finite_distances_sort_last_and_are_still_ranked():
    """NaN/inf are unrankable, not relevant-by-accident: they go to the back."""
    d = np.array(
        [
            [0.0, np.nan, 1.0, INF],
            [np.nan, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [INF, 1.0, 1.0, 0.0],
        ]
    )
    ranking = rankings_from_distances(d)
    assert list(ranking[0]) == [2, 1, 3]  # finite 1.0 first, then nan and inf by index
    r = evaluate(d, cliques=[0, 0, 1, 1])
    assert 0.0 <= r.mean_average_precision <= 1.0


def test_query_with_no_relevant_item_raises():
    """A singleton clique cannot be scored; that is a dataset bug, not a 0.0."""
    d = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    with pytest.raises(ValueError, match="no relevant item"):
        evaluate(d, cliques=[0, 0, 1])


def test_shape_mismatches_raise():
    d = np.zeros((2, 4))
    with pytest.raises(ValueError, match="square"):
        evaluate(d, cliques=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="cliques must have length"):
        evaluate(d, cliques=[0, 0, 1], query_index=[0, 2])
    with pytest.raises(ValueError, match="2-D"):
        rankings_from_distances(np.zeros(4))


def test_random_distances_beat_nothing():
    """Sanity floor: random ranking of 100 items in 10 cliques of 10.

    R/M is only the asymptotic value of E[AP]; for small R it understates it badly,
    because the first hit arrives early enough to lift precision at every later
    hit. Simulated over 200k permutations with R=9 relevant among M=99 ranked
    items, E[AP] = 0.1295, and across 200 seeds this construction gives
    mean 0.1303, std 0.0072. P@k is a fixed cutoff and is unbiased at R/M = 0.0909.

    Anything a real method produces must clear this comfortably.
    """
    rng = np.random.default_rng(0)
    n, per = 100, 10
    cliques = np.repeat(np.arange(n // per), per)
    d = rng.random((n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    r = evaluate(d, cliques=cliques)
    assert r.mean_average_precision == pytest.approx(0.1303, abs=0.03)
    assert r.precision_at_10 == pytest.approx(0.09, abs=0.05)
