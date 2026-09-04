"""Qmax: cover detection by local alignment of two chroma sequences.

Following Serra et al. (2009). Where the 2D-FTM descriptor throws away time to get
one comparable vector, this keeps time and looks for a long diagonal path of
matching moments -- covers usually follow the same sequence of harmonies, even at
a different tempo or in a different key. That is a much stronger signal, and much
more expensive: it is a dynamic program per pair rather than a dot product.

The pipeline for one pair:

    downsample -> transpose b onto a's key (OTI) -> time-delay embed both
    -> cross recurrence plot -> dynamic program -> longest cumulative path

The DP is the part that usually gets written as a slow double loop. It does not
need to be: Q[i, j] depends only on rows i-1 and i-2, never on row i, so a whole
row can be computed in one vectorised pass over j.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csr.features.chroma import (
    downsample_to_rate,
    global_chroma,
    normalize_frames,
)


@dataclass(frozen=True)
class Sequence:
    """One performance, prepared once and reused across every pair it appears in.

    states: (n_states, 12 * m) time-delay embedded chroma.
    profile: (12,) average pitch-class profile, for computing the transposition.
    """

    states: np.ndarray
    profile: np.ndarray
    m: int


def delay_embed(chroma: np.ndarray, m: int = 9, tau: int = 1) -> np.ndarray:
    """Stack each frame with the m-1 that follow it.

    A single chroma frame is a weak description of a moment; a short run of them
    captures how the harmony is moving, which is what actually distinguishes one
    song from another.

    (n_frames, 12) -> (n_frames - (m-1)*tau, 12*m)
    """
    n = len(chroma) - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"need more than {(m - 1) * tau} frames, got {len(chroma)}")
    return np.concatenate([chroma[i * tau : i * tau + n] for i in range(m)], axis=1)


def prepare(
    chroma: np.ndarray, target_hz: float = 2.0, m: int = 9, tau: int = 1
) -> Sequence:
    """Downsample, normalise and embed one performance."""
    coarse = normalize_frames(downsample_to_rate(chroma, target_hz), kind="max")
    return Sequence(delay_embed(coarse, m, tau), global_chroma(coarse), m)


def transpose_states(states: np.ndarray, semitones: int, m: int) -> np.ndarray:
    """Roll the pitch axis of every frame inside an embedded state vector."""
    if semitones % 12 == 0:
        return states
    blocks = states.reshape(len(states), m, 12)
    return np.roll(blocks, semitones, axis=2).reshape(len(states), m * 12)


def cross_recurrence_plot(
    x: np.ndarray, y: np.ndarray, kappa: float = 0.095
) -> np.ndarray:
    """Which moments of x and y count as the same moment.

    A cell is recurrent only if y_j is among x_i's nearest `kappa` fraction *and*
    x_i is among y_j's. Requiring both directions rejects the "hub" frames that
    are close to everything, and makes the plot exactly symmetric:
    crp(x, y) == crp(y, x).T.

    (n_x, d), (n_y, d) -> (n_x, n_y) bool
    """
    d = np.sqrt(
        np.maximum(
            (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * (x @ y.T), 0.0
        )
    )
    eps_x = np.quantile(d, kappa, axis=1, keepdims=True)
    eps_y = np.quantile(d, kappa, axis=0, keepdims=True)
    return (d <= eps_x) & (d <= eps_y)


def _shift(row: np.ndarray, by: int) -> np.ndarray:
    """row[j - by], with zeros where that would run off the left edge."""
    out = np.zeros_like(row)
    out[by:] = row[:-by]
    return out


def qmax_score(crp: np.ndarray, gap_open: float = 0.5, gap_extend: float = 0.5) -> float:
    """Length of the best cumulative diagonal path through a recurrence plot.

    Rewards runs of matching moments that advance through both songs together,
    and tolerates small tempo differences by allowing the path to step 2-for-1 in
    either direction. Gaps cost `gap_open` to start and `gap_extend` to continue.

    Each row is computed from the two rows above it in one vectorised pass, so
    this costs a few milliseconds for a 500 x 500 plot instead of seconds.
    """
    n_rows, n_cols = crp.shape
    if n_rows < 3 or n_cols < 3:
        return 0.0

    zeros = np.zeros(n_cols)
    q_prev1, q_prev2 = zeros, zeros.copy()
    r_prev1, r_prev2 = np.zeros(n_cols, bool), np.zeros(n_cols, bool)
    best = 0.0

    for i in range(n_rows):
        match = crp[i]

        # The three predecessors, all from earlier rows.
        diag = _shift(q_prev1, 1)  # Q[i-1, j-1]
        skip_row = _shift(q_prev2, 1)  # Q[i-2, j-1]
        skip_col = _shift(q_prev1, 2)  # Q[i-1, j-2]

        extended = np.maximum(np.maximum(diag, skip_row), skip_col) + 1.0

        # A gap costs less to continue than to open, so the penalty depends on
        # whether the predecessor cell was itself a match.
        penalty = lambda r, s: np.where(_shift(r, s), gap_open, gap_extend)
        broken = np.maximum(
            np.maximum(diag - penalty(r_prev1, 1), skip_row - penalty(r_prev2, 1)),
            skip_col - penalty(r_prev1, 2),
        )

        q_row = np.where(match, extended, np.maximum(broken, 0.0))
        best = max(best, float(q_row.max()))

        q_prev1, q_prev2 = q_row, q_prev1
        r_prev1, r_prev2 = match, r_prev1

    return best


def qmax_distance(
    a: Sequence,
    b: Sequence,
    kappa: float = 0.095,
    gap_open: float = 0.5,
    gap_extend: float = 0.5,
    symmetrize: bool = True,
) -> float:
    """Distance between two prepared performances. Smaller means more alike.

    Serra's normalisation: sqrt(len(b)) / score, so a long alignment between long
    songs does not automatically beat a long alignment between short ones. An
    empty alignment gives inf, which `rankings_from_distances` puts last.

    Symmetry: the recurrence plot is symmetric, but the DP is not -- its three
    predecessors treat the two axes differently, and the normalisation uses the
    length of `b`. So this is NOT a symmetric function. With `symmetrize=True`
    (the default) it returns the smaller of both directions, which is symmetric
    and is asserted as such in the tests; the tests also assert that the two
    directions genuinely differ, so the asymmetry is a measured fact rather than
    an assumption.
    """

    def one_way(p: Sequence, q: Sequence) -> float:
        shift = int(np.argmax([p.profile @ np.roll(q.profile, s) for s in range(12)]))
        score = qmax_score(
            cross_recurrence_plot(p.states, transpose_states(q.states, shift, q.m), kappa),
            gap_open,
            gap_extend,
        )
        return np.sqrt(len(q.states)) / score if score > 0 else np.inf

    forward = one_way(a, b)
    return min(forward, one_way(b, a)) if symmetrize else forward
