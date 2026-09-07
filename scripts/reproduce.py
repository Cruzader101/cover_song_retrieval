"""Reproduce every number in the README with one command.

    python scripts/reproduce.py --synthetic   # no dataset needed, ~1 min
    python scripts/reproduce.py               # the fast methods on real data
    python scripts/reproduce.py --slow        # adds Qmax on the clique subsample
    python scripts/reproduce.py --learned     # adds the trained embedding

Runs are grouped by collection, and MAP is only comparable within a group.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from report import load, render  # noqa: E402
from run_eval import load_runs  # noqa: E402

from csr.data import cache  # noqa: E402
from csr.eval.metrics import RetrievalResults  # noqa: E402
from csr.run import RESULTS_DIR, manifest, run  # noqa: E402

CACHE = Path("data/interim/chroma_native")
FAST = ["random_full", "duration_full", "ftm2d_full", "ftm2d_nodc_full",
        "random_sub50", "duration_sub50", "ftm2d_sub50", "ftm2d_nodc_sub50",
        "random_test", "duration_test", "ftm2d_test", "ftm2d_nodc_test"]
SLOW = ["qmax_sub50"]
LEARNED = ["learned_test"]


def synthetic() -> int:
    """Whole pipeline on generated data: proves the wiring without the download."""
    from csr.data.synthetic import synthetic_corpus

    frame, chroma = synthetic_corpus(n_cliques=12, per_clique=4, n_singletons=10)
    out = Path("results/synthetic")
    for method in ["random", "duration", "ftm2d"]:
        payload = run(
            {"name": f"{method}_synthetic", "method": method,
             "collection": {"kind": "full"}, "seed": 0},
            frame=frame,
            load_factory=lambda f: (lambda pid: chroma[pid]),
            out_dir=out,
        )
        print(f"\n{method}")
        print(RetrievalResults(**payload["metrics"]))
    print("\n" + render(load(out)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--slow", action="store_true", help="include Qmax")
    ap.add_argument("--learned", action="store_true", help="include the CNN")
    ap.add_argument("--train", action="store_true", help="train the CNN first")
    args = ap.parse_args()

    if args.synthetic:
        return synthetic()

    frame = manifest("hpcp")
    print(f"collection: {len(frame):,} performances, "
          f"{frame['clique_id'].nunique():,} cliques")

    if not cache.exists(CACHE):
        print("building the chroma cache (one-off, a few minutes, ~3.7 GB)")
        cache.build(frame, CACHE)
    chroma = cache.load(CACHE)
    loader = lambda collection: (lambda pid: chroma[pid])

    if args.train:
        subprocess.run([sys.executable, "scripts/train.py"], check=True)

    configs = load_runs()
    for name in FAST:
        if name not in configs:
            print(f"  skipping {name}: not in configs/runs.json")
            continue
        started = time.time()
        payload = run(configs[name], frame=frame, load_factory=loader)
        metrics = payload["metrics"]
        print(f"  {name:<18} MAP {metrics['mean_average_precision']:.4f}  "
              f"MR1 {metrics['mean_rank_first_correct']:8.1f}  "
              f"P@10 {metrics['precision_at_10']:.4f}  "
              f"({time.time() - started:.0f}s)")

    # Qmax and the CNN each get a fresh interpreter. Running them here would put
    # them on top of a process that has already paged in the whole chroma cache
    # and built a 15000 x 15000 distance matrix -- and Qmax then forks twelve
    # workers off that. That combination got this script OOM-killed partway
    # through a two-hour run; a subprocess bounds peak memory to one run at a
    # time. They load from HDF5 rather than the cache, which for these two is
    # also the faster path, since neither reads enough of it to earn the paging.
    for name in (SLOW if args.slow else []) + (LEARNED if args.learned else []):
        if name not in configs:
            print(f"  skipping {name}: not in configs/runs.json")
            continue
        print(f"  {name} (separate process)")
        subprocess.run([sys.executable, "scripts/run_eval.py", name], check=True)

    text = render(load(RESULTS_DIR))
    Path(RESULTS_DIR / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
