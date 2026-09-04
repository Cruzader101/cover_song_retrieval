"""One-off: dump chroma + cross-recurrence + similarity data for a cover pair and a
non-cover pair, as JSON to embed in an interactive HTML visualization.

    python scripts/make_chroma_viz_data.py > /path/to/chroma_viz_data.json
"""

import json
import sys

import numpy as np

from csr.data.datacos import feature_dir, load_chroma, load_metadata
from csr.similarity.qmax import cross_recurrence_plot, prepare


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n_a, 12), (n_b, 12) -> (n_a, n_b), each row/col L2-normalised first."""
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return an @ bn.T


def pick_performances(meta: dict) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    """Return (cover_a, cover_b, distractor) as (clique_id, perf_id) triples.

    cover_a/cover_b share a clique with >= 2 performances; distractor comes from a
    different, singleton clique so it is genuinely unrelated to either.
    """
    cover_clique = next(cid for cid, perfs in meta.items() if len(perfs) >= 2)
    a_id, b_id = list(meta[cover_clique])[:2]

    distractor_clique = next(
        cid for cid, perfs in meta.items() if len(perfs) == 1 and cid != cover_clique
    )
    d_id = next(iter(meta[distractor_clique]))

    return (cover_clique, a_id), (cover_clique, b_id), (distractor_clique, d_id)


def load_prepared(clique_id: str, perf_id: str, feature: str = "hpcp"):
    path = feature_dir(feature) / f"{clique_id}_{feature}" / f"{perf_id}_{feature}.h5"
    chroma = load_chroma(path, feature)
    return prepare(chroma, target_hz=2.0)


def pair_payload(meta, a_key, b_key, feature="hpcp") -> dict:
    a_clique, a_id = a_key
    b_clique, b_id = b_key
    a = load_prepared(a_clique, a_id, feature)
    b = load_prepared(b_clique, b_id, feature)

    # `states` are (n, 12*m) delay-embedded; drop back to (n, 12) via the first block
    # for display purposes (the embedding is for Qmax's DP, not for the chromagram).
    # float64 so rounding for JSON output doesn't print float32 imprecision.
    chroma_a = a.states[:, :12].astype(np.float64)
    chroma_b = b.states[:, :12].astype(np.float64)

    crp = cross_recurrence_plot(a.states, b.states)
    # float64 before rounding: rounding a float32 array and upcasting on .tolist()
    # would otherwise print each value's full float32 imprecision (e.g.
    # 0.5699999928474426 instead of 0.57), bloating the JSON for no reason.
    sim = cosine_similarity_matrix(chroma_a, chroma_b).astype(np.float64)
    crp_rows, crp_cols = np.nonzero(crp)

    return {
        "a": {
            "work_title": meta[a_clique][a_id].get("work_title", ""),
            "perf_artist": meta[a_clique][a_id].get("perf_artist", ""),
            "chroma": np.round(chroma_a, 3).tolist(),
        },
        "b": {
            "work_title": meta[b_clique][b_id].get("work_title", ""),
            "perf_artist": meta[b_clique][b_id].get("perf_artist", ""),
            "chroma": np.round(chroma_b, 3).tolist(),
        },
        # Flat, 2 decimals: a nested array of arrays doubles the bracket overhead
        # for what is otherwise a quarter-million-cell matrix.
        "similarity_shape": list(sim.shape),
        "similarity": np.round(sim, 2).flatten().tolist(),
        # crp is ~2-6% true, so coordinates are far smaller than a dense array.
        "crp_points": list(zip(crp_rows.tolist(), crp_cols.tolist())),
        "frame_rate_hz": 2.0,
    }


def main() -> int:
    meta = load_metadata()
    cover_a, cover_b, distractor = pick_performances(meta)

    payload = {
        "cover_pair": pair_payload(meta, cover_a, cover_b),
        "non_cover_pair": pair_payload(meta, cover_a, distractor),
    }
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
