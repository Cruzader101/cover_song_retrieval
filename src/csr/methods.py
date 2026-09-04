"""The methods being compared, all behind one interface.

Each method takes a collection and returns a square distance matrix over it. The
runner slices out the query rows and hands the result to the same `evaluate` every
other method is scored by, so nothing here can accidentally grade itself on an
easier task than its neighbours.

Signature:
    method(collection, load, params, rng) -> (n_items, n_items) float32

where `load` maps a perf_id to its (n_frames, 12) chroma. Distances are symmetric
with a zero diagonal; the query's own column is excluded downstream by `evaluate`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from csr.features.chroma import downsample_to_rate, normalize_frames
from csr.features.ftm2d import ftm2d_matrix
from csr.similarity.qmax import prepare, qmax_distance
from csr.similarity.vector import cosine_distance_matrix

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def wrap(fn):
        REGISTRY[name] = fn
        return fn

    return wrap


#: Methods that live outside this module, imported only when asked for. This is
#: what keeps torch off the critical path: the classical methods never touch it.
LAZY = {"learned": "csr.models"}


def build(name: str) -> Callable:
    if name not in REGISTRY and name in LAZY:
        import importlib

        importlib.import_module(LAZY[name])
    if name not in REGISTRY:
        raise KeyError(
            f"unknown method {name!r}; have {sorted(REGISTRY) + sorted(LAZY)}"
        )
    return REGISTRY[name]


@register("random")
def random_method(collection, load, params, rng) -> np.ndarray:
    """Shuffled distances. The floor every real method has to clear.

    Expected MAP is roughly (relevant items) / (collection size), which on the
    full benchmark is about 12/14999 = 0.0008.
    """
    n = len(collection)
    # float32 directly: at n=15000 a float64 intermediate is 1.8 GB.
    d = rng.random((n, n), dtype=np.float32)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return d


@register("duration")
def duration_method(collection, load, params, rng) -> np.ndarray:
    """Rank by similarity of length alone. No audio content at all.

    Covers of a work do tend to have similar durations, so this should edge above
    random -- and its `tie_fraction` will be high, because frame counts collide.
    That is the diagnostic doing its job.
    """
    frames = collection["n_frames"].to_numpy(dtype=np.float32)
    d = np.abs(frames[:, None] - frames[None, :])
    np.fill_diagonal(d, 0.0)
    return d


@register("ftm2d")
def ftm2d_method(collection, load, params, rng) -> np.ndarray:
    """2D Fourier magnitude descriptors compared by cosine distance."""
    rate = params.get("target_hz", 5.0)
    descriptors = ftm2d_matrix(
        [
            normalize_frames(
                downsample_to_rate(load(pid), rate), params.get("normalize", "max")
            )
            for pid in tqdm(collection["perf_id"], desc="ftm2d", leave=False)
        ],
        patch_frames=params.get("patch_frames", 180),
        patch_hop=params.get("patch_hop", 60),
        n_time_coeffs=params.get("n_time_coeffs", 32),
        compress=params.get("compress", "sqrt"),
    )
    return cosine_distance_matrix(descriptors)


# --- Qmax runs in worker processes, so its state is module level ------------
# On Windows every worker is a fresh interpreter, so the prepared sequences are
# handed over once at start-up rather than pickled per task.

_SEQUENCES: list | None = None
_QMAX_KWARGS: dict = {}


def _init_qmax_worker(sequences, kwargs) -> None:
    global _SEQUENCES, _QMAX_KWARGS
    _SEQUENCES, _QMAX_KWARGS = sequences, kwargs


def _qmax_row(i: int) -> tuple[int, np.ndarray]:
    """Distances from item i to every item after it."""
    assert _SEQUENCES is not None
    row = np.zeros(len(_SEQUENCES), dtype=np.float32)
    for j in range(i + 1, len(_SEQUENCES)):
        row[j] = qmax_distance(_SEQUENCES[i], _SEQUENCES[j], **_QMAX_KWARGS)
    return i, row


@register("qmax")
def qmax_method(collection, load, params, rng) -> np.ndarray:
    """Sequence alignment over chroma. Accurate, and far too slow for the full set.

    Cost is quadratic in the collection *and* in song length, so this is only ever
    run on a clique subsample. Which cliques were drawn is recorded in the results
    file, and the cheap methods are re-run on the identical subsample so the
    comparison is fair.
    """
    sequences = [
        prepare(
            load(pid),
            target_hz=params.get("target_hz", 2.0),
            m=params.get("m", 9),
            tau=params.get("tau", 1),
        )
        for pid in tqdm(collection["perf_id"], desc="qmax prep", leave=False)
    ]
    kwargs = {
        "kappa": params.get("kappa", 0.095),
        "gap_open": params.get("gap_open", 0.5),
        "gap_extend": params.get("gap_extend", 0.5),
        "symmetrize": params.get("symmetrize", True),
    }

    n = len(sequences)
    d = np.zeros((n, n), dtype=np.float32)
    workers = params.get("n_jobs", 12)

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_qmax_worker, initargs=(sequences, kwargs)
    ) as pool:
        for i, row in tqdm(
            pool.map(_qmax_row, range(n), chunksize=4),
            total=n,
            desc="qmax pairs",
            leave=False,
        ):
            d[i, i + 1 :] = row[i + 1 :]

    d = d + d.T
    np.fill_diagonal(d, 0.0)
    return d
