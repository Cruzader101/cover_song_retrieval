# cover_song_retrieval

Cover song identification built up from classical MIR to learned embeddings,
measured the same way at every step.

Given one performance, rank every other performance in the collection by how
likely it is to be the same underlying work. Covers change key, tempo,
instrumentation, structure and length, so almost nothing survives from the raw
audio -- what survives is the sequence of harmonies, which is what every method
here is built around.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest
```

Nothing above needs the dataset. To check the whole pipeline without downloading
15 GB, run it on generated data:

```bash
.venv/Scripts/python.exe scripts/reproduce.py --synthetic
```

## Data

[Da-TACOS](https://github.com/MTG/da-tacos) benchmark subset, pre-extracted
features only -- no audio. The shape of it matters for how anything is scored:

- **3000 cliques, 15000 performances.** 1000 cliques hold exactly 13 covers each;
  the other 2000 are singletons that exist only as distractors.
- So the collection has 15000 items but only **13000 valid queries**. A singleton
  has no relevant item to find, and `evaluate` raises on it rather than scoring it
  zero, because that would be a dataset bug quietly averaged into the result.
- HPCP chroma is `(n_frames, 12)` at ~21.5 frames/s. Files are HDF5 written by
  `deepdish`, so arrays are datasets but scalars live in attributes.

Fetch it (about 9.6 GB for HPCP plus 3 MB of metadata) into `data/raw/`, which is
gitignored:

```bash
curl -L -C - -o data/raw/_archives/da-tacos_metadata.zip \
  "https://zenodo.org/records/3520368/files/da-tacos_metadata.zip?download=1"
curl -L -C - -o data/raw/_archives/da-tacos_benchmark_subset_hpcp.zip \
  "https://zenodo.org/records/3520368/files/da-tacos_benchmark_subset_hpcp.zip?download=1"
```

Unzip both into `data/raw/`, then:

```bash
.venv/Scripts/python.exe scripts/reproduce.py          # fast methods
.venv/Scripts/python.exe scripts/reproduce.py --slow   # adds Qmax
```

## Methods

| | idea | cost |
|---|---|---|
| `random` | shuffled ranking, the floor everything must clear | free |
| `duration` | rank by similarity of length alone, no audio content | free |
| `ftm2d` | 2D Fourier magnitude of chroma patches: `\|FFT2\|` is unchanged by a circular shift of either axis, so key and start offset fall out for free. One vector per song, so the whole collection is a matrix product | minutes |
| `qmax` | cross-recurrence plot plus a local-alignment DP over chroma. Keeps time, finds the longest shared harmonic path. Far more accurate and quadratic in both collection size and song length | hours |
| `learned` | a small CNN whose convolutions wrap around the pitch axis, so key invariance is built into the architecture rather than trained. Classification over training cliques with a cosine-softmax head; the embedding is the layer before the classifier | GPU minutes |

## Reading the numbers

MAP is only comparable **within one collection**. A method scored on 650 items has
far fewer ways to be wrong than one scored on 15000, so it scores higher for free.
`scripts/report.py` groups by collection and refuses to merge them. Three groups
are used:

- **full** -- all 15000 items. Only methods with no fitted parameters, so leakage
  is impossible by construction.
- **subsample50** -- 50 whole cliques, no distractors. Qmax cannot run on the full
  collection, so the cheap methods are re-run here to give it a fair comparison.
- **test** -- the held-out 200 cliques, for anything trained on the other 800.

Splits are always by clique, never by performance: putting two covers of one work
on opposite sides of a split is the leakage failure mode for this task.

## Results

Not checked in, since a number belongs to the run that produced it rather than to
the repository. `scripts/reproduce.py` writes one `results/<run-name>.json` per run
-- metrics alongside the config, seed and commit behind them -- and collects them
into `results/summary.md`.

## Status

- [x] Evaluation harness: MAP / MR1 / P@10 with known-answer tests (`src/csr/eval/metrics.py`)
- [x] Da-TACOS benchmark subset loader (pre-extracted features)
- [x] Dumb baselines (random, duration)
- [x] 2D Fourier magnitude on HPCP
- [x] Qmax / cross-recurrence alignment
- [x] Learned embeddings

## Layout

```
src/csr/
  data/      datacos.py  loading and the manifest
             splits.py   clique-level splits and subsamples
             cache.py    downsampled chroma, stored once
             synthetic.py generated corpus for tests
  features/  chroma.py   normalise, downsample, transpose, OTI
             ftm2d.py    the 2D Fourier magnitude descriptor
  similarity/vector.py   chunked cosine distance
             qmax.py     recurrence plots and the alignment DP
  eval/      metrics.py  MAP / MR1 / P@10   (the fixed contract)
             batch.py    the same thing, a block of queries at a time
  methods.py the registry: every method returns a distance matrix
  models.py  the learned embedding
  run.py     config -> method -> results/<name>.json
```
