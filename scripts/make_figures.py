"""Render every figure the README embeds, light and dark.

    python scripts/make_figures.py            # all of them
    python scripts/make_figures.py features   # only the ones needing no results

Figures come in two kinds. The *feature* figures explain what a method does and
are built straight from the dataset. The *results* figures read results/*.json,
so reproduce.py has to have run first. Nothing here recomputes a metric: the
number in a figure is the number in the results file, or the figure would be a
second, unversioned source of truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import load_runs  # noqa: E402

from csr import viz  # noqa: E402
from csr.data import cache  # noqa: E402
from csr.data.datacos import query_mask  # noqa: E402
from csr.eval.metrics import rankings_from_distances  # noqa: E402
from csr.features.chroma import (  # noqa: E402
    downsample_to_rate,
    normalize_frames,
    optimal_transposition_index,
    transpose,
)
from csr.features.ftm2d import ftm2d, ftm2d_matrix  # noqa: E402
from csr.methods import build  # noqa: E402
from csr.run import manifest, select_collection  # noqa: E402
from csr.similarity.qmax import cross_recurrence_plot, prepare  # noqa: E402
from csr.similarity.vector import cosine_distance_matrix  # noqa: E402

CACHE = Path("data/interim/chroma_native")
RESULTS = Path("results")
DIST_CACHE = Path("data/interim/distances")

#: Display name, and whether the method uses audio content at all. Keyed by run
#: name with the collection suffix removed, because two runs can share a method
#: and differ only in a parameter -- and then "ftm2d" is not a label, it is two.
METHODS = {
    "random": ("random", False),
    "duration": ("duration", False),
    "ftm2d": ("2D-FTM", True),
    "ftm2d_nodc": ("2D-FTM, no DC", True),
    "qmax": ("Qmax", True),
    "learned": ("learned CNN", True),
    "chroma_mean": ("chroma mean", True),
    "chroma_mean_oti": ("chroma mean, best key", True),
}
SUFFIXES = ("_full", "_sub50", "_test")


def method_key(payload) -> str:
    name = payload["run_name"]
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


COLLECTION_ORDER = ["full", "split", "subsample"]
COLLECTION_TITLES = {
    "full": "full collection",
    "split": "held-out test split",
    "subsample": "50-clique subsample",
}


def load_results() -> list[dict]:
    payloads = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(RESULTS.glob("*.json"))
    ]
    if not payloads:
        raise SystemExit("no results yet -- run scripts/reproduce.py first")
    return payloads


@dataclass(frozen=True)
class Example:
    query: str
    cover: str
    clique: str
    rank: int
    median_rank: int
    n_neighbours: int
    prepared: dict


def typical_example(collection, chroma):
    """A representative cover pair, plus everything needed to place it in context.

    Deliberately not the best pair available. The pair returned is the transposed
    one whose rank under 2D-FTM is closest to the median rank of *all* true covers
    in this collection, so the figure shows a typical case rather than a
    flattering one. Returns the query, that cover, the ranking context and the
    prepared chroma for the whole collection.
    """
    pids = list(collection["perf_id"])
    labels = collection["clique_id"].to_numpy()
    prepared = {
        p: normalize_frames(downsample_to_rate(chroma[p], 5.0), "max") for p in pids
    }

    distances = cosine_distance_matrix(ftm2d_matrix([prepared[p] for p in pids]))
    ranks, candidates = [], []
    for i in range(len(pids)):
        row = distances[i].copy()
        row[i] = np.inf
        place = {j: k + 1 for k, j in enumerate(np.argsort(row, kind="stable"))}
        for j in np.flatnonzero(labels == labels[i]):
            if j != i:
                ranks.append(place[j])
                candidates.append((place[j], i, j))

    median = int(np.median(ranks))
    for rank, i, j in sorted(candidates, key=lambda t: abs(t[0] - median)):
        if optimal_transposition_index(prepared[pids[i]], prepared[pids[j]]) % 12:
            return Example(
                query=pids[i], cover=pids[j], clique=labels[i], rank=rank,
                median_rank=median, n_neighbours=len(pids) - 1, prepared=prepared,
            )
    raise RuntimeError("no transposed cover pair found in this collection")


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(1 - u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def _strip(ax) -> None:
    """Recessive axes: the marks carry the figure, not the furniture."""
    ax.tick_params(length=0, labelsize=8)
    for side in ax.spines.values():
        side.set_visible(False)


# --------------------------------------------------------------------------
# figure 1 -- what the methods actually score
# --------------------------------------------------------------------------

def fig_map_by_collection(payloads, theme) -> None:
    """MAP per method, one panel per collection, never merged onto one axis.

    Separate panels with separate x-axes is the whole point: a method scored on
    650 items and one scored on 15000 are not on the same scale, and a shared
    axis would invite exactly that comparison.
    """
    groups: dict[str, list[dict]] = {}
    for p in payloads:
        groups.setdefault(p["collection"]["kind"], []).append(p)
    kinds = [k for k in COLLECTION_ORDER if k in groups]

    fig, axes = plt.subplots(
        1, len(kinds), figsize=(4.1 * len(kinds), 2.9), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, kind in zip(axes, kinds):
        runs = sorted(
            groups[kind], key=lambda r: r["metrics"]["mean_average_precision"]
        )
        values = [r["metrics"]["mean_average_precision"] for r in runs]
        colors = [
            theme.series[0] if METHODS[method_key(r)][1] else theme.baseline
            for r in runs
        ]

        y = np.arange(len(runs))
        ax.barh(y, values, color=colors, height=0.62)
        ax.set_yticks(y, [METHODS[method_key(r)][0] for r in runs])

        floor = next(
            (r["metrics"]["mean_average_precision"] for r in runs
             if r["method"] == "random"),
            None,
        )
        if floor is not None:
            ax.axvline(floor, color=theme.muted, lw=1, ls=(0, (3, 3)), zorder=0)

        # Direct labels: no reader should have to measure a bar against a tick.
        span = max(values)
        for yi, v in zip(y, values):
            ax.text(v + span * 0.03, yi, f"{v:.3f}", va="center", fontsize=8,
                    color=theme.secondary)

        meta = runs[0]["collection"]
        ax.set_title(
            f"{COLLECTION_TITLES.get(kind, kind)}\n"
            f"{meta['n_items']:,} items, {meta['n_queries']:,} queries",
            color=theme.text, pad=8,
        )
        ax.set_xlim(0, span * 1.3)
        ax.set_xlabel("MAP")
        ax.grid(axis="x", lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        _strip(ax)

    fig.suptitle(
        "Mean average precision by method. Dashed line is the random floor; "
        "grey bars use no audio content.",
        fontsize=9, color=theme.secondary, y=1.13,
    )
    viz.save(fig, "map-by-collection", theme)


# --------------------------------------------------------------------------
# figure 2 -- why the 2D Fourier magnitude is key-invariant
# --------------------------------------------------------------------------

def fig_ftm2d(example, theme) -> None:
    """The invariance the descriptor really has, next to the one it needs.

    Left to middle is the exact claim: roll the pitch axis and |FFT2| does not
    move, so the distance is zero to numerical precision. Left to right is the
    real task, where the cover is not a clean transposition of the query but a
    different performance, and the distance is merely small. The gap between
    those two columns is the whole difficulty of the problem.

    The descriptor's DC bin holds ~97% of its energy, so it is left out of the
    image and the three panels share one log colour scale; without that they
    could not be compared at all.
    """
    query = example.prepared[example.query]
    rolled = transpose(query, 5)
    cover = example.prepared[example.cover]

    descriptors = [ftm2d(x) for x in (query, rolled, cover)]
    # Exactly 0 in theory; floats return signed noise, so the figure shows |d|.
    d_rolled = cosine(descriptors[0], descriptors[1])
    d_cover = cosine(descriptors[0], descriptors[2])

    grids = [np.log10(d.reshape(-1, 7)[:, 1:] + 1e-6) for d in descriptors]
    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)

    seconds, rate = 40, 5.0
    n = int(seconds * rate)
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.2), constrained_layout=True)
    panels = [
        (query, "the query", None, theme.secondary),
        (rolled, "the same query, rolled up 5 semitones",
         f"distance {abs(d_rolled):.0e}\nzero to numerical precision", theme.series[0]),
        (cover, "a real cover of it, in another key",
         f"distance {d_cover:.4f}\nrank {example.rank} of {example.n_neighbours}",
         theme.series[1]),
    ]

    for col, (chroma, title, note, colour) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(chroma[:n].T, aspect="auto", origin="lower", cmap=theme.ramp,
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, color=theme.text, fontsize=9)
        ax.set_yticks([0, 3, 6, 9], ["C", "D#", "F#", "A"])
        ax.set_xticks([0, n // 2, n], ["0s", f"{seconds // 2}s", f"{seconds}s"])
        if col == 0:
            ax.set_ylabel("chroma\npitch class")
        _strip(ax)

        ax = axes[1, col]
        im = ax.imshow(grids[col], aspect="auto", origin="lower", cmap=theme.ramp,
                       interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 2, 5], ["1", "3", "6"])
        ax.set_yticks([0, 15, 31])
        ax.set_xlabel("pitch frequency\n(DC bin omitted)")
        if col == 0:
            ax.set_ylabel("|FFT2|, log\ntime frequency")
        if note:
            # Inside the axes, so two long captions cannot collide.
            ax.text(0.5, -0.42, note, transform=ax.transAxes, ha="center",
                    va="top", fontsize=8.5, color=colour, linespacing=1.4)
        _strip(ax)

    fig.colorbar(im, ax=axes[1, :], location="right", fraction=0.02, pad=0.01,
                 label="log10 magnitude")

    fig.suptitle(
        "|FFT2| of a chroma patch is unchanged by a roll of either axis, so key "
        "and start offset drop out exactly (middle).\nA real cover is not a clean "
        "transposition, so it only lands close -- typically 110th of 649 here.",
        fontsize=9, color=theme.secondary, y=1.11,
    )
    viz.save(fig, "ftm2d-invariance", theme)


# --------------------------------------------------------------------------
# figure 3 -- what Qmax is looking for
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QmaxExample:
    """A median-ranked cover pair under Qmax, and a median-distance stranger."""
    query: str
    cover: str
    cover_distance: float
    cover_rank: int
    median_rank: int
    n_neighbours: int
    stranger: str
    stranger_distance: float


def qmax_example(collection, distances) -> QmaxExample:
    """Pick the pair Qmax handles *typically*, not the one it handles best.

    Same rule as the 2D-FTM figure: the true cover whose rank is closest to the
    median rank of every true cover in the collection. The stranger is taken at
    the median distance from that same query, so neither panel is a lucky draw.
    """
    pids = list(collection["perf_id"])
    labels = collection["clique_id"].to_numpy()

    ranks, candidates = [], []
    for i in range(len(pids)):
        row = distances[i].copy()
        row[i] = np.inf
        place = {j: k + 1 for k, j in enumerate(np.argsort(row, kind="stable"))}
        for j in np.flatnonzero(labels == labels[i]):
            if j != i:
                ranks.append(place[j])
                candidates.append((place[j], i, j))

    median = int(np.median(ranks))
    rank, i, j = min(candidates, key=lambda t: abs(t[0] - median))

    row = distances[i]
    strangers = sorted(
        (k for k in range(len(pids)) if labels[k] != labels[i] and np.isfinite(row[k])),
        key=lambda k: row[k],
    )
    stranger = strangers[len(strangers) // 2]

    return QmaxExample(
        query=pids[i], cover=pids[j], cover_distance=float(row[j]), cover_rank=rank,
        median_rank=median, n_neighbours=len(pids) - 1, stranger=pids[stranger],
        stranger_distance=float(row[stranger]),
    )


def fig_crp(example, chroma, theme) -> None:
    """What Qmax is actually looking at, and what it gets out of it.

    Both plots carry diagonal fragments: the recurrence threshold fixes the
    density, so a stranger has about as many dots as a cover does. What differs
    is how far one diagonal *runs* before it breaks, which is the quantity the
    dynamic program maximises. That is not reliably eyeballable on a typical
    pair, so the score it produces is quoted rather than left to the reader.
    """
    a = prepare(chroma[example.query], target_hz=2.0)
    panels = [
        (prepare(chroma[example.cover], target_hz=2.0),
         f"its cover: Qmax {example.cover_distance:.2f}, "
         f"rank {example.cover_rank} of {example.n_neighbours}", theme.series[0]),
        (prepare(chroma[example.stranger], target_hz=2.0),
         f"a typical stranger: Qmax {example.stranger_distance:.2f} (the median)",
         theme.secondary),
    ]

    # One shared extent, so the panels are the same size and the same scale.
    plots = [cross_recurrence_plot(a.states, other.states) for other, _, _ in panels]
    height = max(p.shape[0] for p in plots)
    width = max(p.shape[1] for p in plots)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), constrained_layout=True)
    for ax, crp, (_, title, colour) in zip(axes, plots, panels):
        rr, cc = np.nonzero(crp)
        ax.scatter(cc, rr, s=0.4, c=colour, linewidths=0, alpha=0.8)
        ax.set_title(f"{title}\n{crp.mean():.1%} of cells recur", color=colour,
                     fontsize=9)
        ax.set_xlabel("time in the other performance")
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(lw=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        _strip(ax)
    axes[0].set_ylabel("time in the query (frames @ 2 Hz)")

    fig.suptitle(
        "Cross-recurrence: a dot means two moments share a harmony. The threshold "
        "fixes the density, so both\npanels are equally dotty; what Qmax measures "
        "is how far a diagonal runs, and it separates these two.",
        fontsize=9, color=theme.secondary, y=1.06,
    )
    viz.save(fig, "cross-recurrence", theme)


# --------------------------------------------------------------------------
# figure 4 -- how deep you have to read before the first true cover
# --------------------------------------------------------------------------

def distances_for(name: str, frame, chroma):
    """The run's distance matrix: whatever the eval saved, else computed here.

    A run configured with save_distances has already written the matrix it was
    scored on, and reusing it means the figure is drawn from the same numbers as
    the table rather than a lookalike recomputation.
    """
    config = load_runs()[name]
    collection = select_collection(frame, config["collection"])
    path = DIST_CACHE / f"{name}.npy"
    if path.is_file():
        return np.load(path), collection

    DIST_CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.get("seed", 0))
    d = build(config["method"])(
        collection, lambda pid: chroma[pid], config.get("params", {}), rng
    )
    np.save(path, d)
    return d, collection


def fig_rank_curve(frame, chroma, theme) -> None:
    """Fraction of queries whose first true cover sits within the top k.

    MR1 is one summary of this curve, and a mean hides that most of the damage is
    done by a long tail. The curve shows where a method actually lives.
    """
    # Three hues plus grey. The DC variant shares 2D-FTM's hue and changes only
    # its dash: it is the same method with one coefficient removed, and giving it
    # a hue of its own would claim otherwise.
    styles = {
        "random_sub50": (theme.baseline, "random", (0, (3, 3))),
        "duration_sub50": (theme.series[2], "duration", (0, (6, 2))),
        "ftm2d_sub50": (theme.series[0], "2D-FTM", "solid"),
        "ftm2d_nodc_sub50": (theme.series[0], "2D-FTM, no DC", (0, (4, 1, 1, 1))),
        "qmax_sub50": (theme.series[1], "Qmax", "solid"),
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    last_rank = 0
    for name, (color, label, dash) in styles.items():
        if not (RESULTS / f"{name}.json").is_file():
            print(f"  rank curve: skipping {name}, no result for it yet")
            continue
        d, collection = distances_for(name, frame, chroma)
        queries = np.flatnonzero(query_mask(collection))
        labels = collection["clique_id"].to_numpy()
        ranking = rankings_from_distances(d[queries], queries)
        relevant = labels[ranking] == labels[queries][:, None]
        first = relevant.argmax(axis=1) + 1

        ks = np.arange(1, relevant.shape[1] + 1)
        curve = (first[:, None] <= ks[None, :]).mean(axis=0)

        ax.plot(ks, curve, color=color, lw=2.0 if dash == "solid" else 1.6,
                ls=dash, label=label)
        last_rank = max(last_rank, int(ks[-1]))

    # No direct labels here: every curve reaches 1.0 at the last rank by
    # construction, so labels at the right edge would all pile up on the same
    # point and say nothing. The legend carries identity, and the two 2D-FTM
    # variants are told apart by their dash rather than their colour.
    ax.set_xscale("log")
    ax.set_xlim(1, last_rank)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("rank k (log scale)")
    ax.set_ylabel("queries whose first true cover is in the top k")
    ax.set_title("How far down the list you have to read", color=theme.text, pad=6)
    ax.grid(lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", labelcolor=theme.secondary,
              fontsize=8.5)
    _strip(ax)

    fig.suptitle(
        "50-clique subsample: 650 items, 13 covers of each work. "
        "Higher and further left is better.",
        fontsize=9, color=theme.secondary, y=1.05,
    )
    viz.save(fig, "rank-curve", theme)


# --------------------------------------------------------------------------
# figure 5 -- what a key change costs, asked two ways
# --------------------------------------------------------------------------

ANALYSIS = RESULTS / "analysis"

#: Drawn in this order, and only these. The two controls bracket the real methods:
#: same descriptor, one key invariant and one not.
ANALYSIS_METHODS = ("chroma_mean", "ftm2d_nodc", "qmax", "learned", "random")


def load_analysis(collection: str) -> dict | None:
    path = ANALYSIS / f"{collection}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def analysis_rows(analysis: dict):
    """(run name, entry, display label) in ANALYSIS_METHODS order.

    Matched on the run name, not on `entry["method"]`. Two runs can share a method
    and differ only in a parameter -- ftm2d_nodc is the ftm2d method with one
    coefficient dropped -- so the method field cannot tell them apart.
    """
    rows = []
    for wanted in ANALYSIS_METHODS:
        for name, entry in analysis["methods"].items():
            if method_key({"run_name": name}) == wanted:
                rows.append((name, entry, METHODS[wanted][0]))
                break
    return rows


def fig_key_invariance(analysis, theme) -> None:
    """The observational answer beside the interventional one.

    Left: what a key change looks like it costs, from the covers that happen to be
    transposed. Right: what it actually costs, from re-keying every performance at
    random. A method whose invariance is real sits at zero on the right however far
    from zero it sits on the left, and the gap between the panels is the finding.
    """
    rows = analysis_rows(analysis)
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)

    shifts = np.arange(1, 7)
    for index, (name, entry, label) in enumerate(rows):
        coefs = entry["coefficients"]
        if "key_shift_1" not in coefs:
            continue
        uses_audio = METHODS[method_key({"run_name": name})][1]
        color = theme.series[index % 3] if uses_audio else theme.baseline
        values = [coefs[f"key_shift_{s}"]["coef"] for s in shifts]
        low = [coefs[f"key_shift_{s}"]["ci_low"] for s in shifts]
        high = [coefs[f"key_shift_{s}"]["ci_high"] for s in shifts]

        left.plot(shifts, values, color=color, lw=1.6, marker="o", ms=3.5, label=label)
        left.fill_between(shifts, low, high, color=color, alpha=0.13, lw=0)

    left.axhline(0, color=theme.muted, lw=1, ls=(0, (3, 3)), zorder=0)
    # The sensitive control sits an order of magnitude above the real methods, and
    # on a linear axis it flattens them into the zero line. symlog keeps its size
    # visible while leaving the small effects readable, and unlike a log axis it
    # still renders the random floor, which is near zero and sometimes negative.
    left.set_yscale("symlog", linthresh=0.05, linscale=0.4)
    left.set_yticks([0.0, 0.05, 0.1, 0.2, 0.5, 1.0])
    left.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    left.set_xlabel("semitones between the two performances")
    left.set_ylabel("cost in log$_{10}$(rank)")
    left.set_title("What transposed covers look like they cost",
                   color=theme.text, pad=8)
    left.legend(frameon=False, fontsize=8, loc="lower right")
    left.minorticks_off()  # symlog scatters unlabelled decade ticks around zero
    left.grid(axis="y", lw=0.6, alpha=0.7)
    left.set_axisbelow(True)
    _strip(left)

    drawn = [(e, label) for _, e, label in rows if "map_rekeyed" in e]
    y = np.arange(len(drawn))
    drops = [100 * (1 - e["map_rekeyed"] / e["map"]) for e, _ in drawn]
    colors = [
        theme.series[1] if d > 1.0 else theme.baseline for d in drops
    ]
    right.barh(y, drops, color=colors, height=0.6)
    right.set_yticks(y, [label for _, label in drawn])
    for yi, drop in zip(y, drops):
        right.text(drop + 1.4, yi, f"{drop:.1f}%", va="center", fontsize=8,
                   color=theme.secondary)
    right.set_xlim(-1, max(max(drops), 10) * 1.25)
    right.set_xlabel("MAP lost when every performance is re-keyed")
    right.set_title("What it actually costs", color=theme.text, pad=8)
    right.grid(axis="x", lw=0.6, alpha=0.7)
    right.set_axisbelow(True)
    _strip(right)

    fig.suptitle(
        "Left: covers that happen to be in another key rank worse for every method "
        "that reads pitch.\nRight: transposing every performance changes nothing "
        "for the same methods. The key change is a marker, not a cause.",
        fontsize=9, color=theme.secondary, y=1.16,
    )
    viz.save(fig, "key-invariance", theme)


# --------------------------------------------------------------------------
# figure 6 -- everything else that predicts a bad rank
# --------------------------------------------------------------------------

FACTOR_LABELS = {
    "pulse_ratio_sd": "tempo change",
    "duration_ratio_sd": "length change",
    "year_gap_sd": "years apart",
    "mode_change": "major <-> minor",
    "instrumental_mismatch": "one has no singer",
}


def fig_effect_sizes(analysis, theme) -> None:
    """Every non-key factor, per method, with clustered intervals.

    Continuous factors are per standard deviation and binary ones are the whole
    switch, so the two are labelled apart rather than ranked against each other.
    Positive is worse: the cover lands further down.
    """
    rows = [r for r in analysis_rows(analysis) if r[1]["method"] != "random"]
    factors = [f for f in FACTOR_LABELS if all(f in e["coefficients"] for _, e, _ in rows)]

    fig, ax = plt.subplots(figsize=(7.2, 0.23 * len(factors) * len(rows) + 1.3),
                           constrained_layout=True)

    spacing = len(rows) + 1
    ticks, labels = [], []
    for f_index, factor in enumerate(factors):
        base = f_index * spacing
        ticks.append(base + (len(rows) - 1) / 2)
        labels.append(FACTOR_LABELS[factor])
        for m_index, (_, entry, label) in enumerate(rows):
            c = entry["coefficients"][factor]
            y = base + m_index
            color = theme.series[m_index % 3]
            ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=color, lw=1.4,
                    solid_capstyle="butt")
            ax.plot(c["coef"], y, "o", color=color, ms=4.5,
                    label=label if f_index == 0 else None)

    ax.axvline(0, color=theme.muted, lw=1, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlabel("cost in log$_{10}$(rank).  positive means the cover ranks worse")
    ax.legend(frameon=False, fontsize=8, ncol=len(rows), loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _strip(ax)
    viz.save(fig, "effect-sizes", theme)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="?", default="all",
                    choices=["all", "features", "results", "analysis"])
    ap.add_argument("--analysis-collection", default="subsample50")
    args = ap.parse_args()

    frame = manifest("hpcp")
    chroma = cache.load(CACHE)

    if args.which in ("all", "features"):
        subsample = select_collection(
            frame, {"kind": "subsample", "n_cliques": 50, "seed": 0}
        )
        chosen = typical_example(subsample, chroma)
        print(f"2D-FTM example: {chosen.query} vs {chosen.cover} "
              f"(clique {chosen.clique}), rank {chosen.rank} of "
              f"{chosen.n_neighbours}, collection median {chosen.median_rank}")
        for theme in viz.THEMES:
            viz.apply(theme)
            fig_ftm2d(chosen, theme)

    if args.which in ("all", "results"):
        payloads = load_results()

        # Qmax over the subsample, computed once and cached: the rank curve plots
        # it and the recurrence figure picks its example out of it. Skipped
        # entirely when the run has not happened, rather than quietly spending an
        # hour recomputing a method nothing has asked for yet.
        chosen = None
        if (RESULTS / "qmax_sub50.json").is_file():
            qmax_matrix, subsample = distances_for("qmax_sub50", frame, chroma)
            chosen = qmax_example(subsample, qmax_matrix)
            print(f"Qmax example: {chosen.query} vs {chosen.cover}, rank "
                  f"{chosen.cover_rank} of {chosen.n_neighbours}, collection "
                  f"median {chosen.median_rank}; stranger {chosen.stranger}")
        else:
            print("  no qmax_sub50 result; skipping the recurrence figure")

        for theme in viz.THEMES:
            viz.apply(theme)
            fig_map_by_collection(payloads, theme)
            fig_rank_curve(frame, chroma, theme)
            if chosen is not None:
                fig_crp(chosen, chroma, theme)

    if args.which in ("all", "analysis"):
        analysis = load_analysis(args.analysis_collection)
        if analysis is None:
            print(f"  no results/analysis/{args.analysis_collection}.json; "
                  "run scripts/analyze.py first")
        else:
            for theme in viz.THEMES:
                viz.apply(theme)
                fig_key_invariance(analysis, theme)
                fig_effect_sizes(analysis, theme)

    for path in sorted(viz.FIGURES.glob("*.png")):
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
