"""Train the chroma embedding on the training-split cliques.

    python scripts/train.py --epochs 40
"""

import argparse
import sys
from pathlib import Path

from csr.data import cache, splits
from csr.features.chroma import downsample_to_rate, normalize_frames
from csr.models import train
from csr.run import manifest

CACHE = Path("data/interim/chroma_native")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--crop", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-hz", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", type=Path, default=Path("results/cnn.pt"))
    args = ap.parse_args()

    frame = manifest("hpcp")
    split = splits.split_by_clique(frame, seed=args.seed)
    print(f"cliques: {len(split.train)} train / {len(split.val)} val / "
          f"{len(split.test)} test")

    if not cache.exists(CACHE):
        print("building chroma cache (one-off)")
        cache.build(frame, CACHE)
    native = cache.load(CACHE)

    train_rows = frame[frame["clique_id"].isin(set(split.train))]
    val_rows = frame[frame["clique_id"].isin(set(split.val))]
    print(f"performances: {len(train_rows)} train / {len(val_rows)} val")

    # The cache is at the native rate; the model trains on a reduced one. Only the
    # performances this run touches are materialised.
    needed = list(train_rows["perf_id"]) + list(val_rows["perf_id"])
    chroma = {
        p: normalize_frames(downsample_to_rate(native[p], args.target_hz), "max")
        for p in needed
    }

    train(
        train_rows["perf_id"].to_list(), train_rows["clique_id"].to_list(), chroma,
        val_rows["perf_id"].to_list(), val_rows["clique_id"].to_list(),
        dim=args.dim, width=args.width, crop=args.crop, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        target_hz=args.target_hz,
        device=args.device, checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
