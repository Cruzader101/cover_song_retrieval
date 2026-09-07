"""Per-performance musical metadata: what a cover pair differs by.

The manifest answers *which* performances exist. This answers what each one is --
its key, its mode, when it came out, whether anyone is singing -- so that a pair of
them can be described by the transformation between them.

Two sources, neither of which the rest of the project reads:

* ``data/raw/da-tacos_benchmark_subset_key/`` -- a whole feature directory of its
  own, one .h5 per performance. The values are HDF5 *group attributes* on a
  ``/key_extractor`` group, not datasets, which is why `datacos.load_chroma` cannot
  reach them.
* the metadata JSON, from which `datacos.build_manifest` currently keeps only
  ``work_title`` and ``perf_artist``.

Read the caveats on `KEY_TO_PITCH_CLASS` and `read_key` before using a key value for
anything. These are estimates from a key detector, not ground truth, and they carry
a specific bias that the analysis has to account for rather than assume away.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from csr.data import datacos

CACHE = Path("data/interim/metadata.csv")
PROFILES = Path("data/interim/chroma_profiles.npy")

#: The twelve key names Essentia actually produced across all 15000 files, checked
#: rather than assumed. It mixes sharps and flats -- Eb and Ab appear, D# and G# do
#: not -- so a mapping built from a guessed naming scheme would silently drop rows.
KEY_TO_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "Ab": 8,
    "A": 9,
    "Bb": 10,
    "B": 11,
}


def read_key(path: Path) -> tuple[int, str, float]:
    """One performance's key file -> (pitch class 0-11, 'major'|'minor', strength).

    ``strength`` is the detector's own confidence and reaches -1.0 when it failed
    outright, so it is kept as a column rather than folded in: a key is only worth
    conditioning on if the estimate behind it was any good.
    """
    with h5py.File(path, "r") as fh:
        attrs = fh["key_extractor"].attrs
        name = _text(attrs["key"])
        scale = _text(attrs["scale"])
        strength = float(attrs["strength"])

    if name not in KEY_TO_PITCH_CLASS:
        raise ValueError(f"{path}: unknown key {name!r}")
    return KEY_TO_PITCH_CLASS[name], scale, strength


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def parse_year(value: str) -> float:
    """Release year as a float, or NaN. 105 rows are empty and 7 are 'Unreleased'."""
    return float(value) if str(value).isdigit() else np.nan


def build_metadata(
    manifest: pd.DataFrame, raw: Path = datacos.RAW, chroma: dict | None = None
) -> pd.DataFrame:
    """One row per performance, describing what it is.

    Columns: perf_id, key_pc, scale, key_strength, release_year, instrumental,
    and pulse_bpm when `chroma` is supplied.

    Performer identity is deliberately not here. `perf_artist_mbid` would be the
    stable id to use, but it is present on only 66.9% of records, and a factor whose
    meaning changes depending on whether a row has an id is worse than one built on
    the `perf_artist` string the manifest already carries for every row.

    Args:
        manifest: rows to describe, as built by `datacos.build_manifest`.
        raw: dataset root, holding both the key feature dir and the metadata JSON.
        chroma: perf_id -> (n_frames, 12), only needed for the tempo column. Left
            out, `pulse_bpm` is absent rather than NaN, so a caller cannot mistake
            "not computed" for "computed and failed".
    """
    meta = datacos.load_metadata(raw)
    key_dir = datacos.feature_dir("key", raw)

    rows = []
    for perf_id, clique_id in zip(manifest["perf_id"], manifest["clique_id"]):
        key_pc, scale, strength = read_key(
            key_dir / f"{clique_id}_key" / f"{perf_id}_key.h5"
        )
        info = meta[clique_id][perf_id]
        rows.append(
            {
                "perf_id": perf_id,
                "key_pc": key_pc,
                "scale": scale,
                "key_strength": strength,
                "release_year": parse_year(info["release_year"]),
                "instrumental": info["instrumental"] == "Yes",
            }
        )

    frame = pd.DataFrame(rows)
    if chroma is not None:
        from csr.analysis.tempo import pulse_bpm

        frame["pulse_bpm"] = [pulse_bpm(chroma[p]) for p in frame["perf_id"]]
    return frame


def cached(
    manifest: pd.DataFrame, chroma: dict, raw: Path = datacos.RAW
) -> tuple[pd.DataFrame, np.ndarray]:
    """Metadata and chroma profiles, built and cached on first use.

    Both are derived from 15000 files and neither changes, so they are written down
    the way `run.manifest` writes the manifest down. Returns them in manifest row
    order, which is the order every distance matrix is indexed by.
    """
    if CACHE.is_file() and PROFILES.is_file():
        frame = pd.read_csv(CACHE)
        if list(frame["perf_id"]) == list(manifest["perf_id"]):
            return frame, np.load(PROFILES)

    frame = build_metadata(manifest, raw, chroma)
    profiles = chroma_profiles(manifest, chroma)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False)
    np.save(PROFILES, profiles)
    return frame, profiles


def chroma_profiles(manifest: pd.DataFrame, chroma: dict) -> np.ndarray:
    """Mean pitch-class profile per performance, in manifest order. -> (n, 12).

    The input to an empirical transposition estimate. Deriving key shift from the
    same chroma the methods see is the check on the key detector, whose C bin is
    visibly a dumping ground: it claims 20% of the collection.
    """
    from csr.features.chroma import global_chroma

    return np.stack([global_chroma(chroma[p]) for p in manifest["perf_id"]])
