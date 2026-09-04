"""Train/val/test splits, always by clique.

Splitting by performance would put covers of the same work on both sides of the
split, which is the leakage failure mode for this domain: a model could memorise a
work in training and be scored on it again at test time.

The 2000 singleton cliques carry no clique information at all, so they are never
trained on. They are added to evaluation collections as distractors, which is the
role they were put in the dataset for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """Clique ids per split. Disjoint by construction."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "seed": self.seed,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Split":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            tuple(d["train"]), tuple(d["val"]), tuple(d["test"]), int(d["seed"])
        )


def split_by_clique(
    manifest: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> Split:
    """Partition the multi-performance cliques into train/val/test.

    Singleton cliques are excluded: they hold no supervision and are only ever
    distractors. With the standard Da-TACOS benchmark this yields 700/100/200
    cliques out of the 1000 real ones.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    sizes = manifest["clique_id"].value_counts()
    cliques = np.array(sorted(sizes[sizes > 1].index))

    rng = np.random.default_rng(seed)
    rng.shuffle(cliques)

    n_train = int(round(fractions[0] * len(cliques)))
    n_val = int(round(fractions[1] * len(cliques)))
    return Split(
        train=tuple(cliques[:n_train]),
        val=tuple(cliques[n_train : n_train + n_val]),
        test=tuple(cliques[n_train + n_val :]),
        seed=seed,
    )


def collection(
    manifest: pd.DataFrame,
    cliques: tuple[str, ...] | None = None,
    with_distractors: bool = True,
) -> pd.DataFrame:
    """Rows to score against: the given cliques, plus singleton distractors.

    Passing ``cliques=None`` keeps every multi-performance clique, i.e. the full
    benchmark. The returned frame is the *collection*; use `query_mask` to pick
    which of its rows may be queries.
    """
    sizes = manifest["clique_id"].value_counts()
    is_singleton = manifest["clique_id"].map(sizes).eq(1)

    if cliques is None:
        keep = ~is_singleton
    else:
        keep = manifest["clique_id"].isin(set(cliques))

    if with_distractors:
        keep = keep | is_singleton
    return manifest[keep].reset_index(drop=True)


def subsample_cliques(
    manifest: pd.DataFrame, n_cliques: int, seed: int = 0
) -> pd.DataFrame:
    """A seeded subsample of whole cliques, for methods too slow to run on all of them.

    Only multi-performance cliques are drawn, so every row of the result is a
    usable query. Which cliques were drawn is recorded in the run's results file.
    """
    sizes = manifest["clique_id"].value_counts()
    cliques = np.array(sorted(sizes[sizes > 1].index))
    if n_cliques > len(cliques):
        raise ValueError(f"asked for {n_cliques} cliques, only {len(cliques)} exist")

    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(cliques, size=n_cliques, replace=False))
    return manifest[manifest["clique_id"].isin(chosen)].reset_index(drop=True)
