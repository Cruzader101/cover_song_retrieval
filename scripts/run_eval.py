"""Run one named configuration from configs/runs.json.

    python scripts/run_eval.py ftm2d_full
    python scripts/run_eval.py qmax_sub50 --out results
"""

import argparse
import json
import sys
from pathlib import Path

from csr.eval.metrics import RetrievalResults
from csr.run import run


def load_runs(path=Path("configs/runs.json")) -> dict:
    """Flatten the grouped config file into {name: config}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        cfg["name"]: cfg
        for group, configs in raw.items()
        if not group.startswith("_")
        for cfg in configs
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="a run name from configs/runs.json")
    ap.add_argument("--configs", type=Path, default=Path("configs/runs.json"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    runs = load_runs(args.configs)
    if args.name not in runs:
        print(f"unknown run {args.name!r}; have: {', '.join(sorted(runs))}")
        return 1

    payload = run(runs[args.name], out_dir=args.out)
    print(f"\n{args.name}  ({payload['method']})")
    print(f"  collection: {payload['collection']['n_items']:,} items, "
          f"{payload['collection']['n_queries']:,} queries")
    print(RetrievalResults(**payload["metrics"]))
    print(f"  took {payload['timings_sec']['total']}s -> {args.out / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
