"""The downsampled-chroma cache must return exactly what it was given."""

import numpy as np
import pandas as pd
import pytest

from csr.data import cache


@pytest.fixture
def built(tmp_path, corpus, monkeypatch):
    """Build a cache over the synthetic corpus without touching data/raw."""
    manifest, chroma = corpus
    monkeypatch.setattr(cache, "load_chroma", lambda path, feature: chroma[path.name])
    frame = pd.DataFrame({
        "perf_id": manifest["perf_id"],
        "path": manifest["perf_id"],
        "n_frames": manifest["n_frames"],
    })
    out = tmp_path / "chroma"
    cache.build(frame, out)
    return out, chroma


def test_cache_round_trips_the_arrays(built):
    out, chroma = built
    loaded = cache.load(out)
    assert set(loaded) == set(chroma)
    for perf_id, original in chroma.items():
        assert np.array_equal(np.asarray(loaded[perf_id]), original)


def test_cache_reports_when_it_exists(tmp_path, built):
    out, _ = built
    assert cache.exists(out)
    assert not cache.exists(tmp_path / "nothing")


def test_cache_needs_frame_counts_to_preallocate(tmp_path, corpus, monkeypatch):
    """Without n_frames the output size is unknown, so it fails loudly rather
    than falling back to the memory-hungry concatenate."""
    manifest, chroma = corpus
    monkeypatch.setattr(cache, "load_chroma", lambda path, feature: chroma[path.name])
    frame = pd.DataFrame({"perf_id": manifest["perf_id"], "path": manifest["perf_id"]})
    with pytest.raises(ValueError, match="needs n_frames"):
        cache.build(frame, tmp_path / "x")


def test_cache_records_how_it_was_built(built):
    import json
    out, _ = built
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["target_hz"] is None
    assert meta["normalize"] == "none"
