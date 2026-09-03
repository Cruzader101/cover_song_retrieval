# cover_song_retrieval

Cover song identification built up from classical MIR to learned embeddings,
measured the same way at every step.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -m pytest
```

## Status

- [x] Evaluation harness: MAP / MR1 / P@10 with known-answer tests (`src/csr/eval/metrics.py`)
- [ ] Da-TACOS benchmark subset loader (pre-extracted features)
- [ ] Dumb baselines (random, tempo/duration-only)
- [ ] 2D Fourier magnitude on HPCP
- [ ] Qmax / cross-recurrence alignment
- [ ] Learned embeddings
