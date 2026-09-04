"""Running one method end to end and writing down what happened.

A result is only worth having if you can tell later what produced it, so every
run writes the resolved config, the seed, the git commit, the collection it was
scored on and how long each stage took, alongside the metrics.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from csr.data import datacos, splits
from csr.eval.batch import evaluate_chunked
from csr.methods import build

MANIFEST_DIR = Path("data/interim")
RESULTS_DIR = Path("results")


def git_state() -> dict:
    """Current commit and whether the tree was dirty when this ran."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def manifest(feature: str = "hpcp", raw: Path = datacos.RAW) -> pd.DataFrame:
    """Load the manifest, building and caching it the first time."""
    path = MANIFEST_DIR / f"manifest_{feature}.csv"
    if path.is_file():
        return pd.read_csv(path)
    frame = datacos.build_manifest(feature=feature, raw=raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def select_collection(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Pick the rows a run is scored over.

    kind='full'      every real clique plus all 2000 distractors
    kind='subsample' n whole cliques, no distractors, for the slow methods
    kind='split'     one side of the clique split, plus distractors
    """
    kind = spec.get("kind", "full")
    if kind == "full":
        return splits.collection(frame)
    if kind == "subsample":
        return splits.subsample_cliques(frame, spec["n_cliques"], spec.get("seed", 0))
    if kind == "split":
        split = splits.split_by_clique(frame, seed=spec.get("seed", 0))
        cliques = getattr(split, spec["split"])
        return splits.collection(frame, cliques, spec.get("distractors", True))
    raise ValueError(f"unknown collection kind {kind!r}")


def chroma_loader(feature: str = "hpcp"):
    """perf_id lookup is by path, so close over the manifest rows we were given."""

    def make(frame: pd.DataFrame):
        paths = dict(zip(frame["perf_id"], frame["path"]))
        return lambda pid: datacos.load_chroma(Path(paths[pid]), feature)

    return make


def run(
    config: dict,
    frame: pd.DataFrame | None = None,
    load_factory=None,
    out_dir: Path = RESULTS_DIR,
) -> dict:
    """Execute one config and write results/<name>.json. Returns the payload."""
    feature = config.get("feature", "hpcp")
    seed = config.get("seed", 0)
    rng = np.random.default_rng(seed)
    timings: dict[str, float] = {}

    @contextmanager
    def stage(label: str):
        start = time.perf_counter()
        yield
        timings[label] = round(time.perf_counter() - start, 2)

    with stage("load"):
        if frame is None:
            frame = manifest(feature)
        collection = select_collection(frame, config.get("collection", {}))
        load = (load_factory or chroma_loader(feature))(collection)

    queries = np.flatnonzero(datacos.query_mask(collection))
    if len(queries) == 0:
        raise ValueError("collection has no query-able rows (all cliques singleton)")

    with stage("distances"):
        distances = build(config["method"])(
            collection, load, config.get("params", {}), rng
        )

    with stage("evaluate"):
        results = evaluate_chunked(
            distances[queries], collection["clique_id"].to_list(), queries
        )

    timings["total"] = round(sum(timings.values()), 2)
    payload = {
        "run_name": config["name"],
        "method": config["method"],
        "seed": seed,
        "git": git_state(),
        "config": config,
        "collection": {
            "kind": config.get("collection", {}).get("kind", "full"),
            "n_items": int(len(collection)),
            "n_cliques": int(collection["clique_id"].nunique()),
            "n_queries": int(len(queries)),
            "spec": config.get("collection", {}),
        },
        "timings_sec": timings,
        "metrics": results.to_dict(),
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.system(),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{config['name']}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
