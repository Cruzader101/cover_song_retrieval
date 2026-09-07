"""Tests for the per-performance metadata loader.

The key files store their values as HDF5 *group attributes* rather than datasets,
which is the detail that makes them invisible to the ordinary chroma loader. These
tests write that exact shape into tmp_path so the parsing is checked without the
9.6 GB download.
"""

import h5py
import numpy as np
import pytest

from csr.data.metadata import KEY_TO_PITCH_CLASS, parse_year, read_key


def write_key_file(path, key: str, scale: str, strength: float):
    """A file shaped like a real one: a group with attributes and no datasets."""
    with h5py.File(path, "w") as fh:
        group = fh.create_group("key_extractor")
        group.attrs["key"] = np.bytes_(key)
        group.attrs["scale"] = np.bytes_(scale)
        group.attrs["strength"] = np.float64(strength)


def test_read_key_decodes_group_attributes(tmp_path):
    path = tmp_path / "P_1_key.h5"
    write_key_file(path, "F", "minor", 0.8437259793281555)
    assert read_key(path) == (5, "minor", pytest.approx(0.8437259793281555))


def test_read_key_rejects_a_name_outside_the_observed_twelve(tmp_path):
    """D# never appears in this dataset; Eb does. A new spelling must not pass.

    Silently mapping an unknown name would put a wrong pitch class into every pair
    that performance takes part in, which is worse than refusing to load it.
    """
    path = tmp_path / "P_2_key.h5"
    write_key_file(path, "D#", "major", 0.9)
    with pytest.raises(ValueError, match="unknown key"):
        read_key(path)


def test_the_twelve_key_names_cover_every_pitch_class():
    """Twelve names, twelve distinct classes, no gaps: a bijection onto 0-11."""
    assert sorted(KEY_TO_PITCH_CLASS.values()) == list(range(12))


def test_semitone_spacing_of_the_key_names():
    """Spot-check the mapping against the interval it claims to encode.

    A fifth is 7 semitones (C->G), a minor third is 3 (C->Eb), and the flats sit
    where the sharps would: Ab is 8, one below A.
    """
    assert KEY_TO_PITCH_CLASS["G"] - KEY_TO_PITCH_CLASS["C"] == 7
    assert KEY_TO_PITCH_CLASS["Eb"] - KEY_TO_PITCH_CLASS["C"] == 3
    assert KEY_TO_PITCH_CLASS["A"] - KEY_TO_PITCH_CLASS["Ab"] == 1


def test_failed_key_detection_keeps_its_negative_strength(tmp_path):
    """Essentia reports -1.0 when it gave up. That must survive as a usable flag."""
    path = tmp_path / "P_3_key.h5"
    write_key_file(path, "C", "major", -1.0)
    assert read_key(path)[2] == -1.0


@pytest.mark.parametrize(
    "value, expected", [("2016", 2016.0), ("1902", 1902.0)]
)
def test_parse_year_reads_a_plain_year(value, expected):
    assert parse_year(value) == expected


@pytest.mark.parametrize("value", ["", "Unreleased"])
def test_parse_year_treats_sentinels_as_missing(value):
    """112 of 15000 rows are one of these two. NaN, not 0, which would be a date."""
    assert np.isnan(parse_year(value))
