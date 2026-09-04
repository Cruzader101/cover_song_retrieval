"""A learned chroma embedding, and the method that scores it.

The classical methods hand-build their invariances: 2D-FTM gets key invariance from
the FFT magnitude, Qmax gets it by transposing to the OTI first. This model gets it
from its architecture instead. Every convolution wraps around the pitch axis, so
shifting the input by a semitone shifts the feature maps by a semitone, and a max
over the pitch axis at the end turns that equivariance into invariance. The test
asserts this holds at random initialisation, before any training -- if it did not,
training would have to spend capacity learning something we can simply build in.

A consequence worth stating: because the invariance is exact by construction,
random transposition as a data augmentation would do literally nothing. It is left
out on purpose, not by oversight.

Training is plain classification over the training cliques with a cosine-softmax
head, not triplet loss. 700 cliques with 13 examples each is a well-posed
classification problem, and it has one hyperparameter that matters (the learning
rate) where triplet has three (margin, sampling, mining). The embedding actually
used is the layer before the classifier, compared with the same
`cosine_distance_matrix` every other descriptor method uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from csr.features.chroma import downsample_to_rate, normalize_frames
from csr.methods import register
from csr.similarity.vector import cosine_distance_matrix


class PitchConv(nn.Module):
    """Convolution that wraps around the pitch axis and zero-pads time.

    Pitch class 11 is a semitone below pitch class 0, so that axis is a circle.
    Padding it with zeros would put a false edge in the middle of the harmony.
    """

    def __init__(self, c_in: int, c_out: int, kernel: int = 3):
        super().__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="circular")
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="constant")
        return self.conv(x)


class ChromaCNN(nn.Module):
    """(B, 1, 12, T) -> (B, dim) unit-norm, invariant to key and to length."""

    def __init__(self, dim: int = 256, width: int = 32):
        super().__init__()
        channels = [1, width, width * 2, width * 4, width * 8, width * 8]
        blocks: list[nn.Module] = []
        for c_in, c_out in zip(channels, channels[1:]):
            blocks += [
                PitchConv(c_in, c_out),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                # Time only. Pooling pitch would destroy the invariance.
                nn.MaxPool2d((1, 2)),
            ]
        self.features = nn.Sequential(*blocks)
        self.project = nn.Linear(channels[-1] * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.amax(dim=2)  # over pitch: equivariance becomes invariance
        pooled = torch.cat([h.amax(dim=2), h.mean(dim=2)], dim=1)  # over time
        return F.normalize(self.project(pooled), dim=1)


class CosineHead(nn.Module):
    """Classifier over cliques on the unit sphere.

    Both the embedding and the class weights are L2-normalised, so each logit is a
    cosine. That trains the model on the exact quantity used at retrieval time.
    """

    def __init__(self, dim: int, n_classes: int, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale = scale

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.scale * embedding @ F.normalize(self.weight, dim=1).T


class Crops(Dataset):
    """Random fixed-length excerpts of training performances.

    Augments by cropping and by resampling in time, a crude tempo change. It does
    not augment by transposition: the network is already exactly invariant to it.
    """

    def __init__(self, perf_ids, labels, chroma, crop: int = 400, train: bool = True):
        self.perf_ids = list(perf_ids)
        self.labels = list(labels)
        self.chroma = chroma
        self.crop = crop
        self.train = train

    def __len__(self) -> int:
        return len(self.perf_ids)

    def __getitem__(self, i: int):
        x = np.asarray(self.chroma[self.perf_ids[i]], dtype=np.float32)

        if self.train:
            want = int(self.crop * np.random.uniform(0.8, 1.25))
            start = np.random.randint(0, max(len(x) - want, 1))
            piece = x[start : start + want]
            idx = np.linspace(0, len(piece) - 1, self.crop).round().astype(int)
            x = piece[np.clip(idx, 0, len(piece) - 1)]
        else:
            x = x[: self.crop]

        if len(x) < self.crop:
            x = np.pad(x, ((0, self.crop - len(x)), (0, 0)))
        return torch.from_numpy(x.T.copy()).unsqueeze(0), self.labels[i]


@torch.no_grad()
def embed(model: ChromaCNN, perf_ids, chroma, device="cuda", batch=16) -> np.ndarray:
    """Embed full-length performances, no cropping. -> (n, dim) float32."""
    model.eval()
    out = []
    for start in range(0, len(perf_ids), batch):
        group = [
            np.asarray(chroma[p], dtype=np.float32)
            for p in perf_ids[start : start + batch]
        ]
        width = max(len(c) for c in group)
        padded = np.zeros((len(group), 1, 12, width), dtype=np.float32)
        for i, c in enumerate(group):
            padded[i, 0, :, : len(c)] = c.T
        out.append(model(torch.from_numpy(padded).to(device)).cpu().numpy())
    return np.concatenate(out)


def train(
    train_ids,
    train_cliques,
    chroma,
    val_ids=None,
    val_cliques=None,
    dim: int = 256,
    width: int = 32,
    crop: int = 400,
    epochs: int = 40,
    batch_size: int = 48,
    lr: float = 1e-3,
    seed: int = 0,
    target_hz: float = 5.0,
    device: str = "cuda",
    checkpoint: Path = Path("results/cnn.pt"),
    log=print,
) -> Path:
    """Train on the training cliques, select on validation MAP, save the best.

    Selection is on retrieval MAP rather than classification accuracy because the
    two come apart: a model can separate the training cliques well and still
    produce an embedding space that ranks unseen works badly, and it is the second
    thing we are actually measuring.
    """
    from csr.eval.batch import evaluate_chunked

    torch.manual_seed(seed)
    np.random.seed(seed)

    classes = sorted(set(train_cliques))
    label_of = {c: i for i, c in enumerate(classes)}
    labels = [label_of[c] for c in train_cliques]

    loader = DataLoader(
        Crops(train_ids, labels, chroma, crop, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # Windows spawns workers, which would re-pickle the whole cache per worker.
        num_workers=0,
    )

    model = ChromaCNN(dim, width).to(device)
    head = CosineHead(dim, len(classes)).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=lr
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = head(model(x))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        schedule.step()

        line = f"epoch {epoch:3d}  loss {total / seen:.4f}  acc {correct / seen:.3f}"
        if val_ids is not None and (epoch % 5 == 0 or epoch == epochs):
            vectors = embed(model, val_ids, chroma, device)
            val_map = evaluate_chunked(
                cosine_distance_matrix(vectors), list(val_cliques)
            ).mean_average_precision
            line += f"  val MAP {val_map:.4f}"
            if val_map > best:
                best = val_map
                torch.save(
                    {
                        "model": model.state_dict(),
                        "dim": dim,
                        "width": width,
                        "target_hz": target_hz,
                        "epoch": epoch,
                        "val_map": val_map,
                    },
                    checkpoint,
                )
                line += "  *"
        log(line)

    log(f"best val MAP {best:.4f} -> {checkpoint}")
    return Path(checkpoint)


@register("learned")
def learned_method(collection, load, params, rng) -> np.ndarray:
    """Embed the collection with a trained checkpoint and compare by cosine."""
    path = Path(params.get("checkpoint", "results/cnn.pt"))
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}; run scripts/train.py first")

    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device)
    model = ChromaCNN(state["dim"], state["width"]).to(device)
    model.load_state_dict(state["model"])

    # Inference has to see chroma at the rate the model was trained on.
    rate = state.get("target_hz", 5.0)
    ids = list(collection["perf_id"])
    chroma = {
        p: normalize_frames(downsample_to_rate(load(p), rate), "max") for p in ids
    }
    return cosine_distance_matrix(embed(model, ids, chroma, device))
