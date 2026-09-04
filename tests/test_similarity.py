"""Distances: symmetry claims, chunking equivalence, and the Qmax dynamic program."""

import numpy as np
import pytest

from csr.similarity.qmax import (
    cross_recurrence_plot,
    delay_embed,
    prepare,
    qmax_distance,
    qmax_score,
)
from csr.similarity.vector import cosine_distance_matrix


def test_cosine_known_answers():
    """Identical -> 0, orthogonal -> 1, opposite -> 2."""
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    d = cosine_distance_matrix(x)
    assert d[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert d[0, 2] == pytest.approx(1.0, abs=1e-6)
    assert d[0, 3] == pytest.approx(2.0, abs=1e-6)


def test_cosine_self_matrix_is_exactly_symmetric():
    """The docstring claims exact symmetry, not symmetry to within rounding."""
    rng = np.random.default_rng(0)
    d = cosine_distance_matrix(rng.random((40, 16)))
    assert np.array_equal(d, d.T)
    assert np.array_equal(np.diag(d), np.zeros(40, dtype=d.dtype))


@pytest.mark.parametrize("chunk", [1, 3, 7, 10_000])
def test_cosine_chunking_does_not_change_the_answer(chunk):
    rng = np.random.default_rng(0)
    x = rng.random((25, 8))
    assert np.array_equal(
        cosine_distance_matrix(x, chunk=chunk), cosine_distance_matrix(x, chunk=25)
    )


def test_delay_embed_shape_and_content():
    """m=3, tau=1 over 5 frames -> 3 states, each holding 3 consecutive frames."""
    x = np.arange(5 * 12, dtype=np.float32).reshape(5, 12)
    out = delay_embed(x, m=3, tau=1)
    assert out.shape == (3, 36)
    assert np.array_equal(out[0], np.concatenate([x[0], x[1], x[2]]))


def test_delay_embed_rejects_too_short_input():
    with pytest.raises(ValueError, match="need more than"):
        delay_embed(np.zeros((4, 12), dtype=np.float32), m=9, tau=1)


def test_crp_is_exactly_transpose_symmetric():
    """The mutual-neighbour rule is order-independent, so crp(x,y) == crp(y,x).T."""
    rng = np.random.default_rng(0)
    x, y = rng.random((30, 8)), rng.random((25, 8))
    assert np.array_equal(cross_recurrence_plot(x, y), cross_recurrence_plot(y, x).T)


def test_crp_matches_a_sequence_against_itself_along_the_diagonal():
    rng = np.random.default_rng(0)
    x = rng.random((20, 8))
    assert cross_recurrence_plot(x, x).diagonal().all()


def test_qmax_full_diagonal():
    """A perfect 6-long diagonal is a 6-long path."""
    assert qmax_score(np.eye(6, dtype=bool)) == pytest.approx(6.0)


def test_qmax_empty_plot_has_no_path():
    assert qmax_score(np.zeros((6, 6), dtype=bool)) == pytest.approx(0.0)


def test_qmax_partial_diagonal_hand_computed():
    """Matches at (1,1),(2,2),(3,3) only.

    Q[1,1] = 1 (no usable predecessor), Q[2,2] = Q[1,1] + 1 = 2,
    Q[3,3] = Q[2,2] + 1 = 3. Nothing else is a match, so max(Q) = 3.
    """
    r = np.zeros((6, 6), dtype=bool)
    r[1, 1] = r[2, 2] = r[3, 3] = True
    assert qmax_score(r) == pytest.approx(3.0)


def test_qmax_follows_a_two_for_one_tempo_change():
    """b moves twice as fast as a. The DP allows a 2-for-1 step, so the four
    matches still chain into a path of length 4 rather than four paths of 1."""
    r = np.zeros((8, 8), dtype=bool)
    for i, j in [(1, 1), (2, 3), (3, 5), (4, 7)]:
        r[i, j] = True
    assert qmax_score(r) == pytest.approx(4.0)


def test_qmax_prefers_a_real_cover_to_an_unrelated_song(corpus):
    """The property the method exists for, on the synthetic corpus."""
    _, chroma = corpus
    a, b, other = (prepare(chroma[k]) for k in ("P_0_0", "P_0_1", "P_3_0"))
    assert qmax_distance(a, b) < qmax_distance(a, other)


def test_qmax_symmetrized_is_symmetric_and_raw_is_not(corpus):
    """Symmetry is claimed only for symmetrize=True, and the asymmetry of the raw
    direction is asserted rather than assumed."""
    _, chroma = corpus
    a, b = prepare(chroma["P_0_0"]), prepare(chroma["P_1_0"])
    assert qmax_distance(a, b, symmetrize=True) == qmax_distance(
        b, a, symmetrize=True
    )
    assert qmax_distance(a, b, symmetrize=False) != qmax_distance(
        b, a, symmetrize=False
    )
