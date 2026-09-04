"""Known-answer tests for chroma preprocessing and the 2D-FTM descriptor.

Expected numbers are hand-computed in each docstring.
"""

import numpy as np
import pytest

from csr.features.chroma import (
    downsample,
    global_chroma,
    normalize_frames,
    optimal_transposition_index,
    transpose,
)
from csr.features.ftm2d import ftm2d


def test_downsample_block_means():
    """4 frames of constant value 0,1,2,3 with factor 2 -> means 0.5 and 2.5."""
    x = np.repeat(np.arange(4, dtype=np.float32)[:, None], 12, axis=1)
    out = downsample(x, 2)
    assert out.shape == (2, 12)
    assert out[0, 0] == pytest.approx(0.5)
    assert out[1, 0] == pytest.approx(2.5)


def test_downsample_drops_trailing_partial_block():
    """5 frames with factor 2 -> 2 full blocks; the odd frame is dropped."""
    x = np.ones((5, 12), dtype=np.float32)
    assert downsample(x, 2).shape == (2, 12)


def test_normalize_max_leaves_silent_frames_at_zero():
    """A frame of all zeros has no pitch content, so it must not be scaled up."""
    x = np.zeros((2, 12), dtype=np.float32)
    x[0, 3] = 4.0
    out = normalize_frames(x, "max")
    assert out[0, 3] == pytest.approx(1.0)
    assert out[1].sum() == pytest.approx(0.0)


def test_transpose_is_a_circular_roll():
    """Rolling by 12 returns the original; rolling by 5 then 7 also does."""
    x = np.eye(12, dtype=np.float32)
    assert np.array_equal(transpose(x, 12), x)
    assert np.array_equal(transpose(transpose(x, 5), 7), x)


def test_oti_recovers_a_known_transposition():
    """b is a transposed by 5, so rolling b by (-5) % 12 == 7 restores a."""
    rng = np.random.default_rng(0)
    a = rng.random((200, 12)).astype(np.float32)
    b = transpose(a, 5)
    shift = optimal_transposition_index(a, b)
    assert shift == 7
    assert np.allclose(transpose(b, shift), a)


def test_oti_is_antisymmetric():
    """oti(a, b) == -oti(b, a) mod 12, as the docstring claims."""
    rng = np.random.default_rng(1)
    a = rng.random((150, 12)).astype(np.float32)
    b = transpose(a, 3)
    assert optimal_transposition_index(a, b) == (
        -optimal_transposition_index(b, a)
    ) % 12


def test_global_chroma_is_unit_norm():
    rng = np.random.default_rng(2)
    profile = global_chroma(rng.random((50, 12)).astype(np.float32))
    assert profile.shape == (12,)
    assert np.linalg.norm(profile) == pytest.approx(1.0)


@pytest.mark.parametrize("semitones", range(12))
def test_ftm2d_is_exactly_transposition_invariant(semitones):
    """The whole point of the descriptor: key must not change it.

    |FFT2| is unchanged by a circular shift of either input axis, and transposing
    is exactly a circular shift of the pitch axis.
    """
    rng = np.random.default_rng(0)
    x = rng.random((600, 12)).astype(np.float32)
    assert np.allclose(ftm2d(transpose(x, semitones)), ftm2d(x), atol=1e-6)


def test_ftm2d_is_time_shift_invariant_within_a_patch():
    """Exact for a single patch. Across patches the aggregation drifts slightly,
    which is why the whole-song case is asserted at a looser tolerance below."""
    rng = np.random.default_rng(0)
    x = rng.random((180, 12)).astype(np.float32)
    base = ftm2d(x, patch_frames=180)
    for shift in (1, 37, 90):
        assert np.allclose(ftm2d(np.roll(x, shift, axis=0), patch_frames=180), base,
                           atol=1e-6)


def test_ftm2d_whole_song_shift_drifts_but_stays_close():
    rng = np.random.default_rng(0)
    x = rng.random((600, 12)).astype(np.float32)
    drift = np.abs(ftm2d(np.roll(x, 37, axis=0)) - ftm2d(x)).max()
    assert drift < 0.05


def test_ftm2d_is_unit_norm_and_right_shape():
    rng = np.random.default_rng(0)
    d = ftm2d(rng.random((600, 12)).astype(np.float32), n_time_coeffs=32)
    assert d.shape == (32 * 7,)
    assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-6)


def test_ftm2d_pads_songs_shorter_than_one_patch():
    """Short recordings exist in the collection; dropping them would change the task."""
    rng = np.random.default_rng(0)
    d = ftm2d(rng.random((20, 12)).astype(np.float32), patch_frames=180)
    assert d.shape == (32 * 7,)
    assert np.isfinite(d).all()
