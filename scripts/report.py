"""Collect results/*.json into comparison tables, one per collection.

Refuses to merge collections: a method scored on 650 items and one scored on
15000 are not comparable, and putting them in one table would invite exactly that
mistake. Flags any method that fails to beat the random baseline it shares a
collection with: that is a bug until proven otherwise.
"""

import json
from collections import defaultdict
from pathlib import Path


def load(results_dir=Path("results")):
    payloads = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def group_key(payload) -> str:
    c = payload["collection"]
    return f"{c['kind']} ({c['n_items']:,} items, {c['n_queries']:,} queries)"


def render(payloads) -> str:
    groups = defaultdict(list)
    for p in payloads:
        groups[group_key(p)].append(p)

    lines = ["# Results", ""]
    for name, runs in sorted(groups.items()):
        baseline = next(
            (r["metrics"]["mean_average_precision"] for r in runs
             if r["method"] == "random"), None
        )
        lines += [f"## {name}", "",
                  "| method | MAP | MR1 | P@10 | ties | time (s) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in sorted(runs, key=lambda r: -r["metrics"]["mean_average_precision"]):
            m = r["metrics"]
            flag = ""
            if baseline is not None and r["method"] != "random" and \
                    m["mean_average_precision"] <= baseline:
                flag = "  **BUG?**"
            lines.append(
                f"| {r['run_name']}{flag} | {m['mean_average_precision']:.4f} | "
                f"{m['mean_rank_first_correct']:.1f} | {m['precision_at_10']:.4f} | "
                f"{m['tie_fraction']:.2%} | {r['timings_sec']['total']:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    payloads = load()
    if not payloads:
        print("no results yet; run scripts/run_eval.py first")
    else:
        text = render(payloads)
        Path("results/summary.md").write_text(text, encoding="utf-8")
        print(text)
