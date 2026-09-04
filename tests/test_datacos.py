"""The loader, against .h5 files written in the real Da-TACOS layout.

Builds a miniature copy of the archive on disk rather than mocking h5py, so the
path parsing, the metadata cross-check and the failure paths are all exercised for
real. The fail-loud contract matters here: a silently skipped file would change
what is being measured without changing any number that looks wrong.
"""

import json

import h5py
import numpy as np
import pytest

from csr.data.datacos import (
    build_manifest,
    load_chroma,
    load_metadata,
    n_frames,
    parse_ids,
    perf_files,
    query_mask,
)


@pytest.fixture
def raw(tmp_path):
    """Two cliques of 2 plus one singleton, laid out like the real archive."""
    cliques = {"W_1": ["P_1", "P_2"], "W_2": ["P_3", "P_4"], "W_9": ["P_9"]}
    meta = {
        c: {
            p: {"work_title": f"title {c}", "perf_artist": f"artist {p}",
                "work_id": c, "perf_id": p}
            for p in perfs
        }
        for c, perfs in cliques.items()
    }
    (tmp_path / "da-tacos_metadata").mkdir()
    (tmp_path / "da-tacos_metadata" / "da-tacos_benchmark_subset_metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )

    root = tmp_path / "da-tacos_benchmark_subset_hpcp"
    rng = np.random.default_rng(0)
    for i, (clique, perfs) in enumerate(cliques.items()):
        folder = root / f"{clique}_hpcp"
        folder.mkdir(parents=True)
        for j, perf in enumerate(perfs):
            with h5py.File(folder / f"{perf}_hpcp.h5", "w") as fh:
                fh.create_dataset("hpcp", data=rng.random((50 + 10 * i + j, 12)))
    return tmp_path


def test_parse_ids_strips_the_feature_suffix(raw):
    path = raw / "da-tacos_benchmark_subset_hpcp" / "W_1_hpcp" / "P_1_hpcp.h5"
    assert parse_ids(path, "hpcp") == ("W_1", "P_1")


def test_perf_files_finds_every_performance(raw):
    assert len(perf_files("hpcp", raw)) == 5


def test_perf_files_raises_when_the_feature_is_missing(raw):
    with pytest.raises(FileNotFoundError, match="no feature directory"):
        perf_files("crema", raw)


def test_load_metadata_shape(raw):
    meta = load_metadata(raw)
    assert set(meta) == {"W_1", "W_2", "W_9"}
    assert meta["W_1"]["P_1"]["work_title"] == "title W_1"


def test_load_chroma_returns_frames_by_twelve(raw):
    path = raw / "da-tacos_benchmark_subset_hpcp" / "W_1_hpcp" / "P_1_hpcp.h5"
    chroma = load_chroma(path)
    assert chroma.shape == (50, 12)
    assert chroma.dtype == np.float32
    assert n_frames(path) == 50


def test_load_chroma_accepts_a_transposed_array(tmp_path):
    """Some feature dumps store (12, n). Fix it at the boundary, not downstream."""
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("hpcp", data=np.random.rand(12, 30))
    assert load_chroma(path).shape == (30, 12)


def test_load_chroma_raises_on_a_missing_key(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("something_else", data=np.zeros((10, 12)))
    with pytest.raises(KeyError, match="has no dataset"):
        load_chroma(path)


def test_load_chroma_raises_on_non_finite_values(tmp_path):
    path = tmp_path / "nan.h5"
    data = np.zeros((10, 12))
    data[3, 3] = np.nan
    with h5py.File(path, "w") as fh:
        fh.create_dataset("hpcp", data=data)
    with pytest.raises(ValueError, match="NaN or inf"):
        load_chroma(path)


def test_load_chroma_raises_on_zero_frames(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("hpcp", data=np.zeros((0, 12)))
    with pytest.raises(ValueError, match="zero frames"):
        load_chroma(path)


def test_load_chroma_raises_when_neither_axis_is_twelve(tmp_path):
    path = tmp_path / "wrong.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("hpcp", data=np.zeros((10, 7)))
    with pytest.raises(ValueError, match="neither axis is 12"):
        load_chroma(path)


def test_build_manifest_matches_the_metadata(raw):
    manifest = build_manifest("hpcp", raw)
    assert len(manifest) == 5
    assert set(manifest.columns) >= {
        "perf_id", "clique_id", "path", "work_title", "perf_artist", "n_frames"
    }
    assert manifest["n_frames"].min() == 50


def test_build_manifest_raises_when_a_file_is_missing_from_the_metadata(raw):
    """A file on disk with no metadata entry means we would be scoring something
    whose clique we do not know."""
    folder = raw / "da-tacos_benchmark_subset_hpcp" / "W_1_hpcp"
    with h5py.File(folder / "P_999_hpcp.h5", "w") as fh:
        fh.create_dataset("hpcp", data=np.zeros((10, 12)))
    with pytest.raises(ValueError, match="not in the metadata"):
        build_manifest("hpcp", raw)


def test_build_manifest_raises_on_an_incomplete_download(raw):
    (raw / "da-tacos_benchmark_subset_hpcp" / "W_2_hpcp" / "P_3_hpcp.h5").unlink()
    with pytest.raises(ValueError, match="download is incomplete"):
        build_manifest("hpcp", raw)


def test_query_mask_excludes_singleton_cliques(raw):
    manifest = build_manifest("hpcp", raw)
    mask = query_mask(manifest)
    assert mask.sum() == 4
    assert not mask[manifest["clique_id"].to_numpy() == "W_9"].any()
