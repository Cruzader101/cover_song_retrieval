"""A on-disk cache of downsampled chroma.

Reading 15000 HDF5 files takes minutes, and every method wants the same
downsampled arrays, so we pay that cost once. Songs have different lengths, so
they are stored end to end in one flat array with an index of offsets -- which
also means loading is a memory-map rather than 15000 allocations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from csr.data.datacos import FRAME_RATE_HZ, load_chroma
from csr.features.chroma import downsample_to_rate, normalize_frames


def build(
    manifest: pd.DataFrame,
    out: Path,
    target_hz: float | None = None,
    feature: str = "hpcp",
    normalize: str = "none",
) -> Path:
    """Read every performance once and write <out>.npy plus <out>.json.

    Defaults to storing chroma at its native rate, unnormalised. Methods
    downsample and normalise to whatever they need, and doing it here as well
    would silently downsample twice -- `downsample_to_rate` assumes its input is
    at the native frame rate, so a cache that had already been reduced would be
    reduced again by the same factor.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if "n_frames" not in manifest.columns:
        raise ValueError("manifest needs n_frames so the cache can be preallocated")

    label = "native" if target_hz is None else f"{target_hz}Hz"
    factor = 1 if target_hz is None else max(
        int(round(FRAME_RATE_HZ / target_hz)), 1
    )
    lengths = (manifest["n_frames"].to_numpy() // factor).astype(int)
    total = int(lengths.sum())

    # Write straight into a memory-mapped file. Collecting every array first and
    # concatenating at the end would hold two full copies at once, which is ~7 GB
    # for the benchmark subset.
    flat = np.lib.format.open_memmap(
        out.with_suffix(".npy"), mode="w+", dtype=np.float32, shape=(total, 12)
    )

    index, cursor = {}, 0
    rows = list(zip(manifest["perf_id"], manifest["path"]))
    for (perf_id, path), expected in tqdm(
        list(zip(rows, lengths)), desc=f"cache {label}"
    ):
        chroma = load_chroma(Path(path), feature)
        if target_hz is not None:
            chroma = downsample_to_rate(chroma, target_hz)
        if normalize != "none":
            chroma = normalize_frames(chroma, normalize)
        chroma = chroma[:expected]
        flat[cursor : cursor + len(chroma)] = chroma
        index[perf_id] = [cursor, cursor + len(chroma)]
        cursor += len(chroma)

    flat.flush()
    del flat
    out.with_suffix(".json").write_text(
        json.dumps({"target_hz": target_hz, "feature": feature,
                    "normalize": normalize, "index": index}),
        encoding="utf-8",
    )
    return out


def load(out: Path) -> dict[str, np.ndarray]:
    """perf_id -> (n_frames, 12) views into one memory-mapped array."""
    out = Path(out)
    flat = np.load(out.with_suffix(".npy"), mmap_mode="r")
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    return {pid: flat[a:b] for pid, (a, b) in meta["index"].items()}


def exists(out: Path) -> bool:
    out = Path(out)
    return out.with_suffix(".npy").is_file() and out.with_suffix(".json").is_file()
