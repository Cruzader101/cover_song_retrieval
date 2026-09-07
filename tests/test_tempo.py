"""Known-answer tests for the chroma-derived pulse estimate.

There is no madmom ground truth to check against, so the ground truth here is
constructed: `synthetic._render` holds every chord for exactly `frames_per_chord`
frames, which puts a known period in the novelty curve.
"""

import numpy as np
import pytest

from csr.analysis.tempo import BPM_RANGE, novelty, octave_fold, pulse_bpm, pulse_period
from csr.data.datacos import FRAME_RATE_HZ
from csr.data.synthetic import _progression, _render


def test_novelty_fires_only_on_change():
    """A held chord is one event, not a wall of them.

    Four frames of one chord then four of another: flux is zero everywhere except
    the single frame where the chord changes.
    """
    chroma = np.zeros((8, 12), dtype=np.float32)
    chroma[:4, 0] = 1.0
    chroma[4:, 7] = 1.0

    curve = novelty(chroma)
    assert len(curve) == 7
    assert curve[3] == pytest.approx(1.0)  # pitch class 7 arrives
    assert np.count_nonzero(curve) == 1  # class 0 leaving is not a second event


def test_pulse_period_recovers_an_exact_impulse_train():
    """Impulses every 50 frames -> period 50.

    At 86.13 frames/s that is 60 * 86.13 / 50 = 103.4 BPM, inside the search range.
    """
    curve = np.zeros(2000)
    curve[::50] = 1.0
    assert pulse_period(curve) == 50


def test_pulse_period_is_not_dragged_to_the_shortest_lag():
    """The lag window's floor must not win by default.

    Autocorrelation at lag k sums n-k terms, so without normalising by the overlap
    count a long lag is penalised for arithmetic reasons alone and the peak drifts
    down to the bottom of the window. Period 120 is near the top of the range; the
    floor is at lag 25.
    """
    curve = np.zeros(4000)
    curve[::120] = 1.0
    assert pulse_period(curve) == 120


@pytest.mark.parametrize("frames_per_chord", [40, 60, 100])
def test_pulse_bpm_recovers_the_synthetic_chord_rate(frames_per_chord):
    """A rendered performance's pulse must match the rate its chords change at.

    Exact, not up to a multiple. At 40 frames per chord the autocorrelation at one,
    two and three chords sits within 1% -- so this is the test that fails if the
    peak picker stops preferring the fundamental and starts reporting the bar.
    """
    rng = np.random.default_rng(0)
    chroma = _render(
        rng,
        _progression(rng),
        frames_per_chord=frames_per_chord,
        n_repeats=40,
        transpose=0,
        noise=0.15,
    )
    expected = 60.0 * FRAME_RATE_HZ / frames_per_chord
    assert pulse_bpm(chroma) == pytest.approx(expected, rel=0.02)


def test_pulse_bpm_stays_inside_the_searched_range():
    rng = np.random.default_rng(1)
    chroma = _render(
        rng, _progression(rng), frames_per_chord=70, n_repeats=30, transpose=3, noise=0.2
    )
    assert BPM_RANGE[0] <= pulse_bpm(chroma) <= BPM_RANGE[1]


def test_octave_fold_ignores_half_and_double_time():
    """120 vs 240 BPM is the same performance re-counted, not a re-timed one."""
    assert octave_fold(240 / 120) == pytest.approx(0.0)
    assert octave_fold(120 / 240) == pytest.approx(0.0)
    assert octave_fold(120 / 480) == pytest.approx(0.0)
    assert octave_fold(1.0) == pytest.approx(0.0)


def test_octave_fold_is_symmetric_and_bounded():
    """A ratio and its inverse describe the same difference, and 0.5 is the maximum.

    sqrt(2) is the furthest any ratio can be from an octave: log2 = 0.5 exactly.
    """
    assert octave_fold(1.5) == pytest.approx(octave_fold(1 / 1.5))
    assert octave_fold(np.sqrt(2)) == pytest.approx(0.5)
    ratios = np.array([1.1, 1.3, 1.7, 3.9, 0.2])
    assert np.all(octave_fold(ratios) <= 0.5)


def test_too_short_a_curve_raises():
    """A performance shorter than the slowest period searched cannot be measured."""
    with pytest.raises(ValueError, match="too short"):
        pulse_period(np.zeros(20))
