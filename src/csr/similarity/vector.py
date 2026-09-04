"""Distances between fixed-length descriptors.

This is the cheap half of the project: once every performance is one vector, the
whole 13000 x 15000 comparison is a single matrix product, and the only real
concern is not allocating too much at once.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit length. (n, d) -> (n, d)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def cosine_distance_matrix(
    a: np.ndarray,
    b: np.ndarray | None = None,
    chunk: int = 2048,
    dtype: type = np.float32,
) -> np.ndarray:
    """Pairwise cosine distance, 1 - cos(angle), in [0, 2].

    Args:
        a: (n_a, d) descriptors, one per row.
        b: (n_b, d), or None to compare `a` against itself.
        chunk: rows of `a` per matrix product. 13000 x 15000 float32 is 780 MB,
            and this keeps the temporaries far below that.

    Returns:
        (n_a, n_b) of `dtype`.

    Symmetric when `b` is None: the result is explicitly averaged with its own
    transpose and the diagonal forced to zero, so `d == d.T` holds exactly rather
    than to within BLAS rounding. The tests assert that exact equality.
    """
    a = l2_normalize(np.asarray(a, dtype=np.float64))
    square = b is None
    other = a if square else l2_normalize(np.asarray(b, dtype=np.float64))

    out = np.empty((len(a), len(other)), dtype=dtype)
    for start in range(0, len(a), chunk):
        stop = start + chunk
        out[start:stop] = (1.0 - a[start:stop] @ other.T).astype(dtype)

    if square:
        out = (out + out.T) / 2
        np.fill_diagonal(out, 0.0)
    return out
