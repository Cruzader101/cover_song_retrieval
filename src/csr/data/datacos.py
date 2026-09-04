"""Reading the Da-TACOS benchmark subset off disk.

Layout, confirmed against the real archives:

    data/raw/da-tacos_metadata/da-tacos_benchmark_subset_metadata.json
    data/raw/da-tacos_benchmark_subset_hpcp/W_163930_hpcp/P_546633_hpcp.h5

The subset is 3000 cliques: 1000 real ones of 13 performances each, plus 2000
singleton cliques that exist only as distractors. So the collection holds 15000
performances but only 13000 of them can be a query -- a singleton has no relevant
item, and `evaluate` raises on it by design rather than scoring it zero.

The .h5 files were written by deepdish: arrays are ordinary datasets, while scalars
and strings live in HDF5 attributes.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RAW = Path("data/raw")
METADATA = "da-tacos_metadata/da-tacos_benchmark_subset_metadata.json"

# Dataset key inside each .h5, per feature.
H5_KEY = {"hpcp": "hpcp", "crema": "crema", "cens": "chroma_cens"}

# Chroma is computed from 44100 Hz audio at hop 512, so ~86.1 frames per second.
# Verified against MusicBrainz track lengths in the metadata: over 300 files the
# implied rate is 86.3 frames/s, against 44100/512 = 86.13. Nothing else in the
# project hard-codes a hop -- downsampling takes a target rate in Hz and derives
# the factor from here -- so this constant is the only place it can be wrong.
FRAME_RATE_HZ = 44100 / 512


def feature_dir(feature: str = "hpcp", raw: Path = RAW) -> Path:
    """Directory holding the per-clique subdirectories for one feature."""
    return Path(raw) / f"da-tacos_benchmark_subset_{feature}"


def load_metadata(raw: Path = RAW) -> dict[str, dict[str, dict]]:
    """{clique_id: {perf_id: {work_title, perf_artist, ...}}}"""
    import json

    path = Path(raw) / METADATA
    if not path.is_file():
        raise FileNotFoundError(f"metadata JSON not found at {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_ids(path: Path, feature: str = "hpcp") -> tuple[str, str]:
    """Recover (clique_id, perf_id) from a path like W_163930_hpcp/P_546633_hpcp.h5."""
    suffix = f"_{feature}"
    return path.parent.name.removesuffix(suffix), path.stem.removesuffix(suffix)


def perf_files(feature: str = "hpcp", raw: Path = RAW) -> list[Path]:
    """Every performance .h5 for one feature, sorted."""
    root = feature_dir(feature, raw)
    if not root.is_dir():
        raise FileNotFoundError(f"no feature directory at {root}; download it first")
    files = sorted(root.glob("*/*.h5"))
    if not files:
        raise FileNotFoundError(f"{root} contains no .h5 files")
    return files


def load_chroma(path: Path, feature: str = "hpcp") -> np.ndarray:
    """Load one performance's chroma as (n_frames, 12) float32.

    Raises rather than returning anything partial: a missing key or a malformed
    array is a data bug we want to hear about immediately.
    """
    key = H5_KEY[feature]
    with h5py.File(path, "r") as fh:
        if key not in fh:
            raise KeyError(f"{path} has no dataset {key!r} (found {list(fh)})")
        chroma = np.asarray(fh[key], dtype=np.float32)

    if chroma.ndim != 2:
        raise ValueError(f"{path}: expected 2-D chroma, got shape {chroma.shape}")
    if chroma.shape[1] != 12:
        # Some features are stored transposed; accept (12, n) and fix it here.
        if chroma.shape[0] == 12:
            chroma = chroma.T
        else:
            raise ValueError(f"{path}: neither axis is 12, shape {chroma.shape}")
    if chroma.shape[0] == 0:
        raise ValueError(f"{path}: zero frames")
    if not np.isfinite(chroma).all():
        raise ValueError(f"{path}: contains NaN or inf")
    return np.ascontiguousarray(chroma)


def n_frames(path: Path, feature: str = "hpcp") -> int:
    """Frame count without reading the whole array."""
    with h5py.File(path, "r") as fh:
        shape = fh[H5_KEY[feature]].shape
    return int(shape[0] if shape[1] == 12 else shape[1])


def build_manifest(
    feature: str = "hpcp", raw: Path = RAW, count_frames: bool = True
) -> pd.DataFrame:
    """One row per performance in the collection.

    Columns: perf_id, clique_id, path, work_title, perf_artist, n_frames.

    Cross-checks the files on disk against the metadata JSON and raises if they
    disagree, since a silent mismatch would quietly change what we are measuring.
    """
    meta = load_metadata(raw)
    files = perf_files(feature, raw)

    rows = []
    for path in files:
        clique_id, perf_id = parse_ids(path, feature)
        info = meta.get(clique_id, {}).get(perf_id)
        if info is None:
            raise ValueError(f"{path} is not in the metadata ({clique_id}/{perf_id})")
        rows.append(
            {
                "perf_id": perf_id,
                "clique_id": clique_id,
                "path": str(path),
                "work_title": info.get("work_title", ""),
                "perf_artist": info.get("perf_artist", ""),
            }
        )

    expected = sum(len(perfs) for perfs in meta.values())
    if len(rows) != expected:
        raise ValueError(
            f"found {len(rows)} feature files but metadata lists {expected} "
            "performances; the download is incomplete"
        )

    manifest = pd.DataFrame(rows)
    if count_frames:
        manifest["n_frames"] = [
            n_frames(Path(p), feature) for p in manifest["path"]
        ]
    return manifest


def query_mask(manifest: pd.DataFrame) -> np.ndarray:
    """Boolean mask of rows that may be used as queries.

    A performance in a singleton clique has no relevant item, so it stays in the
    collection as a distractor but never becomes a query.
    """
    counts = manifest["clique_id"].map(manifest["clique_id"].value_counts())
    return (counts > 1).to_numpy()
