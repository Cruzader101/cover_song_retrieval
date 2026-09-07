"""Ask which musical transformations cost a method its ranking.

    python scripts/analyze.py test
    python scripts/analyze.py full --methods ftm2d_nodc_full chroma_mean_full

One collection at a time, because a rank only means something against the number of
items it was competing with. Every method scored on that collection gets the same
treatment: where each of its true covers landed, what changed between the two
performances, and a within-query regression of the first on the second.

Nothing here re-derives a metric. The pair table is scored back into a MAP and
checked against the number in results/<run>.json, so an analysis can never quietly
describe a different matrix than the one the results table reports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from csr.analysis import factors, regress, stats  # noqa: E402
from csr.data import cache, metadata  # noqa: E402
from csr.data.datacos import query_mask  # noqa: E402
from csr.eval.batch import evaluate_chunked, evaluate_chunked_pairs  # noqa: E402
from csr.methods import build  # noqa: E402
from csr.run import DISTANCE_DIR, manifest, select_collection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import load_runs  # noqa: E402

RESULTS = Path("results")
ANALYSIS = RESULTS / "analysis"
PAIRS = Path("data/interim/pairs")
CHROMA_CACHE = "data/interim/chroma_native"

#: How much of the sensitive control's key effect we would have to be able to see
#: before calling an invariant method's null a real null. Pre-registered here rather
#: than chosen once the coefficients are on screen.
EQUIVALENCE_FRACTION = 0.25

#: Fewest cliques a factor has to vary across before its coefficient is reported.
#: Below this the clustered standard error is being asked for more than a handful
#: of independent observations can give.
MIN_FACTOR_CLIQUES = 5


def distances_for(name: str, config: dict, collection, chroma):
    """The run's distance matrix: whatever the eval saved, else computed here.

    Same rule `make_figures.distances_for` follows. A run configured with
    save_distances has already written the matrix it was scored on, and reusing it
    keeps the 47-minute method from being re-run to answer a question about it.
    """
    path = DISTANCE_DIR / f"{name}.npy"
    if path.is_file():
        return np.load(path)
    rng = np.random.default_rng(config.get("seed", 0))
    return build(config["method"])(
        collection, lambda pid: chroma[pid], config.get("params", {}), rng
    )


def pairs_for(name: str, config: dict, collection, chroma) -> np.ndarray:
    """Where every true cover landed, cached. -> (n_pairs, 3) int32.

    Checked against the reported MAP before it is cached. A pair table that scores
    differently than results/<name>.json describes some other run, and continuing
    from it would put a plausible number under a wrong label.
    """
    path = PAIRS / f"{name}.npz"
    if path.is_file():
        stored = np.load(path)
        if list(stored["perf_id"]) == list(collection["perf_id"]):
            return stored["pairs"]

    queries = queries_of(collection)
    distances = distances_for(name, config, collection, chroma)
    # Slice out the query rows and drop the square matrix before scoring. On the
    # full collection that is 900 MB released before the part that allocates most.
    rows = distances[queries]
    del distances
    results, pairs = evaluate_chunked_pairs(
        rows, collection["clique_id"].to_list(), queries, chunk_size=1024
    )

    reported = json.loads((RESULTS / f"{name}.json").read_text())["metrics"]
    if abs(results.mean_average_precision - reported["mean_average_precision"]) > 1e-6:
        raise ValueError(
            f"{name}: this matrix scores MAP {results.mean_average_precision:.6f} but "
            f"results/{name}.json reports {reported['mean_average_precision']:.6f}. "
            "The analysis and the results table would be describing different runs."
        )

    PAIRS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, pairs=pairs, perf_id=collection["perf_id"].to_numpy().astype(str)
    )
    return pairs


def queries_of(collection) -> np.ndarray:
    """Positional rows that may be queries -- the same rule `run.run` scores by."""
    return np.flatnonzero(query_mask(collection))


def transposition_test(name, config, collection, chroma, seed) -> float:
    """MAP when every performance is re-keyed at random, by its own amount.

    The observational finding -- that covers which happen to be in another key rank
    worse -- cannot say whether the transposition caused that. A singer who moves a
    song into their own range is usually reinterpreting it in every other way too,
    and no regression can control for a variable nobody measured.

    This intervenes instead. Same recordings, same arrangements, same everything;
    one thing changed. A method whose key invariance is real must return an
    identical ranking, so the difference between these two numbers separates "the
    method fails on transposed covers" from "transposed covers are different covers".
    """
    rng = np.random.default_rng(seed)
    shifts = dict(
        zip(collection["perf_id"], rng.integers(0, 12, len(collection)).tolist())
    )
    distances = build(config["method"])(
        collection,
        lambda pid: np.roll(chroma[pid], shifts[pid], axis=1),
        config.get("params", {}),
        np.random.default_rng(config.get("seed", 0)),
    )
    queries = queries_of(collection)
    return evaluate_chunked(
        distances[queries], collection["clique_id"].to_list(), queries
    ).mean_average_precision


def analyse_one(name, config, collection, meta, profiles, chroma, n_boot, seed):
    """Factors, a within-query regression, and clustered intervals for one method."""
    pairs = pairs_for(name, config, collection, chroma)
    frame = factors.pair_factors(pairs, collection, meta, profiles)

    usable = frame["year_gap"].notna().to_numpy()
    dropped = int((~usable).sum())
    model_frame = frame[usable]

    X, names = factors.design_matrix(model_frame)
    queries = model_frame["query_id"].to_numpy()

    # Two ways a factor can be unestimable here, both reported rather than
    # silently absorbed. It can have no variation inside any query, which is what
    # a query fixed effect needs. Or it can vary in too few cliques to support a
    # standard error clustered on them: `same_artist` is true for 2 of 30,648
    # pairs on the test split, all inside one clique, and the random control duly
    # reported that as a significant effect on a ranking made of noise.
    cliques = model_frame["clique_id"].to_numpy()
    codes = np.unique(cliques, return_inverse=True)[1]
    keep = regress.within_variation(X, queries) > 1e-10
    for column in np.flatnonzero(keep):
        touched = np.unique(codes[X[:, column] != 0]).size
        if touched < MIN_FACTOR_CLIQUES:
            keep[column] = False

    unestimable = [n for n, k in zip(names, keep) if not k]
    X, names = X[:, keep], tuple(n for n, k in zip(names, keep) if k)

    fit = regress.within_ols(
        model_frame["log_rank"].to_numpy(),
        X,
        queries,
        model_frame["clique_id"].to_numpy(),
        names,
    )

    # Descriptive, not modelled: how often a cover lands in the top ten at each key
    # distance, with an interval that resamples cliques.
    by_key = {}
    for shift in range(7):
        rows = frame["key_shift"] == shift
        if rows.sum() < 30:
            continue
        interval = stats.cluster_bootstrap(
            frame.loc[rows, "hit_at_10"].to_numpy(),
            frame.loc[rows, "clique_id"].to_numpy(),
            n_boot=n_boot,
            seed=seed,
        )
        by_key[shift] = {
            "n_pairs": int(rows.sum()),
            "hit_at_10": interval.estimate,
            "low": interval.low,
            "high": interval.high,
        }

    agreement = float(np.mean(frame["key_shift"] == frame["oti_shift"]))
    return {
        "method": config["method"],
        "n_pairs": int(len(frame)),
        "n_pairs_dropped_no_year": dropped,
        "factors_not_estimable": unestimable,
        "key_label_agrees_with_chroma": agreement,
        "coefficients": {
            n: {
                "coef": float(fit.coef[i]),
                "se": float(fit.se[i]),
                "p": float(fit.p[i]),
                "ci_low": float(fit.ci_low[i]),
                "ci_high": float(fit.ci_high[i]),
            }
            for i, n in enumerate(fit.names)
        },
        "holm_adjusted_p": dict(zip(fit.names, stats.holm(fit.p).tolist())),
        "n_obs": fit.n_obs,
        "n_queries": fit.n_groups,
        "n_cliques": fit.n_clusters,
        "r2_within": fit.r2_within,
        "hit_at_10_by_key_shift": by_key,
    }


def key_effect(entry: dict) -> float:
    """Mean of the six key-shift dummies: how much being in another key costs."""
    return float(
        np.mean([entry["coefficients"][f"key_shift_{s}"]["coef"] for s in range(1, 7)])
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("collection", help="a collection group from configs/runs.json")
    ap.add_argument("--methods", nargs="*", help="run names; default is the whole group")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--transpose-budget",
        type=float,
        default=600.0,
        help="skip the re-keying test for methods that took longer than this to "
             "score, since it costs one more full distance computation",
    )
    args = ap.parse_args()

    grouped = json.loads(Path("configs/runs.json").read_text(encoding="utf-8"))
    if args.collection not in grouped:
        groups = [g for g in grouped if not g.startswith("_")]
        print(f"unknown collection {args.collection!r}; have: {', '.join(groups)}")
        return 1

    runs = load_runs()
    names = args.methods or [c["name"] for c in grouped[args.collection]]
    missing = [n for n in names if not (RESULTS / f"{n}.json").is_file()]
    if missing:
        print(f"no results yet for: {', '.join(missing)}. Run scripts/run_eval.py first.")
        return 1

    frame = manifest()
    chroma = cache.load(CHROMA_CACHE)
    meta_all, profiles_all = metadata.cached(frame, chroma)

    earlier = ANALYSIS / f"{args.collection}.json"
    previous = (
        json.loads(earlier.read_text(encoding="utf-8"))["methods"]
        if earlier.is_file()
        else {}
    )

    # Start from what is already there so `--methods` can be run one at a time.
    # The full collection needs about 2 GB per method and will not survive doing
    # six of them in one process; this makes that a scheduling choice rather than
    # a reason to lose the other five.
    entries = dict(previous)
    for name in names:
        started = time.perf_counter()
        collection = select_collection(frame, runs[name]["collection"])
        meta, profiles = factors.align(collection, meta_all, profiles_all)
        entries[name] = analyse_one(
            name, runs[name], collection, meta, profiles, chroma, args.n_boot, args.seed
        )

        reported = json.loads((RESULTS / f"{name}.json").read_text())
        entries[name]["map"] = reported["metrics"]["mean_average_precision"]
        cost = reported["timings_sec"]["distances"]
        if cost <= args.transpose_budget:
            entries[name]["map_rekeyed"] = transposition_test(
                name, runs[name], collection, chroma, args.seed
            )
        elif (
            "map_rekeyed" in previous.get(name, {})
            and previous[name]["map"] == entries[name]["map"]
        ):
            # Skipped as too expensive, but measured on an earlier run of the same
            # matrix. Qmax takes two and a half hours to re-key; losing that to a
            # re-run over some other method's numbers would be daft.
            entries[name]["map_rekeyed"] = previous[name]["map_rekeyed"]
        entries[name]["seconds"] = round(time.perf_counter() - started, 1)
        print(f"  {name}: {entries[name]['n_pairs']:,} pairs "
              f"({entries[name]['seconds']}s)")

    # The control sets the scale a null is judged against. Without it, "the interval
    # covers zero" could just mean the analysis cannot see anything at all.
    control = next((n for n in entries if runs[n]["method"] == "chroma_mean"), None)
    bound = abs(key_effect(entries[control])) * EQUIVALENCE_FRACTION if control else None

    payload = {
        "collection": args.collection,
        "seed": args.seed,
        "n_boot": args.n_boot,
        "equivalence_bound": bound,
        "equivalence_bound_from": control,
        "methods": entries,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS / f"{args.collection}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n                          observed    re-keyed    MAP")
    print("                          key effect  MAP drop    (reported)")
    for name, entry in sorted(entries.items(), key=lambda kv: key_effect(kv[1])):
        drop = "     --   "
        if "map_rekeyed" in entry:
            drop = f"{100 * (1 - entry['map_rekeyed'] / entry['map']):+9.1f}%"
        print(f"  {name:24s} {key_effect(entry):+.4f}    {drop}  {entry['map']:.4f}")

    if bound is not None:
        print(
            f"\nkey effect is on log10(rank), averaged over the six shift dummies; "
            f"the equivalence bound is {bound:.4f}, {EQUIVALENCE_FRACTION:.0%} of "
            f"{control}'s.\nA method whose key invariance is real drops 0.0% when "
            "every performance is re-keyed at random."
        )
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
