# Finding the same songs from a cover of a song.
It's actually kind of difficult for a model to realize one song is a cover of the same underlying song. So this project tries to rank performances of different songs so that an actual cover of a song ranks at the top over other performances.

This is somewhat hard because a cover of a song can differ in key, tempo, and instruments. 

# How do you rank the cover of a song?
So you want to find what doesn't change in between covers which is the harmonic progression.

However comparing he harmonic progressions of songs brings in the same problems of tempo and key:
- keys change the actual chords being played
- tempo changes in what timeframe the chords are played.

A lot of this project is just finding ways to compare chord sequences while ignoring changes to those chord sequences.

# What methods are used to achieve this?
I used four different methods:
- just look at duration
- squish the entire song into one matrix, and then find the 2d fourier transform of it, and then find the magnitude. 
- use the q-max algorithm to align the two song's chord sequences
- learning invariances through a convulution network

# Evaluation Metrics
I used four evaluation metrics for each method:
- average precision measure the precision-at-rank of every position where a correct song was identified. so correct songs in earlier positions have a higher rank than correct songs in later positions. then the MAP (mean average precision) is the average of all the average precisions for each song.
- MR1 (mean rank of first correct hit) measures on average how far down the ranked list a model ranked the first correct hit.
- P@10 (precision at 10) measures out of the first ten rankings, how many are actually correct.

# Data Used
I did not collect all this data, I used the DA-TACOS dataset. Specifically the "Benchmark subset" which includes 15,000-performances. 

This data set includes 1,000 "cliques" which are groups of 13 cover perfoamcnes of the same song. The remaining 2,000 items are cliques with only one performance used as distractors.

For each performance, there is a file containing the chroma for every fraction of a second of a song.

There is a master list that is basically a dictionary for the label of each performance and the file containing the chroma.

# Results

These come from one run at commit `b80b20b` on a Windows machine with 32 GB of RAM
and an RTX 5070 Laptop. Every run writes its own file in `results/`, with the config,
seed and commit that produced it, and `scripts/report.py` collects them.

MAP is only comparable **within one collection**. A method scored on 650 items has
far fewer ways to be wrong than one scored on 15,000, so it scores higher for free.
That is why these are three tables and not one.

The "2D-FTM, no DC" rows are the same 2D Fourier method with one coefficient
removed. That is explained under the tables.

**All 15,000 performances** — 13,000 of them can be queries, since the 2,000
singletons have no correct answer to find.

| method | MAP | MR1 | P@10 | ties | time |
|---|---:|---:|---:|---:|---:|
| 2D-FTM, no DC | 0.0547 | 300.9 | 0.0698 | 0.30% | 7 min |
| 2D-FTM | 0.0368 | 342.6 | 0.0510 | 0.28% | 2 min |
| duration | 0.0024 | 1043.7 | 0.0017 | 48.90% | 1 min |
| random | 0.0014 | 1183.9 | 0.0008 | 0.05% | 1 min |

**Held-out test split** — 4,600 items, 2,600 queries. The only collection the
trained network is allowed to be scored on.

| method | MAP | MR1 | P@10 | ties | time |
|---|---:|---:|---:|---:|---:|
| learned CNN | 0.2926 | 27.1 | 0.3263 | 0.12% | 80 s |
| 2D-FTM, no DC | 0.0901 | 83.0 | 0.1095 | 0.17% | 2 min |
| 2D-FTM | 0.0593 | 97.3 | 0.0773 | 0.16% | 19 s |
| duration | 0.0070 | 317.4 | 0.0046 | 22.96% | 3 s |
| random | 0.0046 | 352.5 | 0.0032 | 0.02% | 4 s |

**50 cliques on their own** — 650 items, no distractors. Q-max is too slow to run
on the full 15,000, so the cheap methods are re-run here to meet it fairly.

| method | MAP | MR1 | P@10 | ties | time |
|---|---:|---:|---:|---:|---:|
| Qmax | 0.5585 | 4.1 | 0.5980 | 34.89% | 47 min |
| 2D-FTM, no DC | 0.2263 | 10.5 | 0.2448 | 0.62% | 17 s |
| 2D-FTM | 0.1781 | 13.5 | 0.2012 | 0.62% | 3 s |
| duration | 0.0489 | 40.5 | 0.0431 | 3.95% | 0 s |
| random | 0.0294 | 46.6 | 0.0237 | 0.00% | 0 s |

The times are wall clock and some runs shared the machine with other work, so treat
them as an upper bound rather than a benchmark.

## Every method against its floor

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/map-by-collection-dark.png">
  <img alt="MAP by method, one panel per collection" src="docs/figures/map-by-collection-light.png">
</picture>

One panel per collection, each with its own scale, so the three are not accidentally
compared. Grey bars are the methods that never look at the audio. The dashed line is
the random floor.

Both of the dumb baselines beat random, and `duration` beating random is expected
rather than impressive: covers of a song do tend to run to similar lengths. The give
away is the ties column. On the full collection **48.9%** of neighbouring pairs in
duration's ranking are exact ties, because frame counts collide constantly, so about
half of that ranking is really just index order.

The MAP numbers look low everywhere, and that is the task rather than a bug. A query
on the full collection is hunting 12 correct answers among 14,999. MR1 is the easier
column to feel: the best untrained method puts the first correct cover at rank 301 on
average, and the network puts it at 27.

## How far down the list you have to read

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/rank-curve-dark.png">
  <img alt="Fraction of queries whose first correct cover falls within rank k" src="docs/figures/rank-curve-light.png">
</picture>

For each rank k, the share of queries that have found a correct cover by then. Q-max
gets one into first place for **86%** of queries, 2D-FTM for 39%, random for 2%.

This is the picture MR1 flattens. 2D-FTM's MR1 of 13.5 on the subsample is not a
typical query at rank 13; it is most queries doing well and a long tail doing badly.

## Why the 2D Fourier method works, and where it stops

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ftm2d-invariance-dark.png">
  <img alt="Chroma and 2D Fourier magnitude for a query, the same query transposed, and a real cover" src="docs/figures/ftm2d-invariance-light.png">
</picture>

The middle column is the reason the method exists, and it holds exactly: roll the
pitch axis, which is what changing key does, and the magnitude spectrum does not
move at all. The distance is 1e-07, zero as far as floating point is concerned.

The right column is the actual job. A real cover is not a clean transposition of the
query, it is a different performance, so it only lands *close* — rank 110 out of 649.
The gap between those two columns is the whole difficulty of the problem. Both
examples are median cases picked by rank, not flattering ones.

**Deleting one number was the biggest win.** The descriptor's DC coefficient is just
the mean of the chroma patch, and it carries about **97%** of the vector's energy.
With it in, every distance on the subsample lands between 0 and 0.05, so the bins
that actually say which song this is are a rounding error on a bin that says almost
nothing. Zeroing it lifts MAP by **27% / 52% / 49%** relative across the three
collections. It is off by default (`drop_dc`), so the plain 2D-FTM row stays the
method as published and the improvement is a measured comparison rather than a
silent edit.

## What Q-max is actually looking at

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/cross-recurrence-dark.png">
  <img alt="Cross-recurrence plots for a cover pair and a typical stranger" src="docs/figures/cross-recurrence-light.png">
</picture>

A dot means two moments of the two songs share a harmony. The threshold fixes how
many dots there are, so an unrelated song is about as dotty as a real cover — 2.3%
against 2.5% — and staring at it does not tell you much. What the alignment measures
is how far a single diagonal runs before it breaks, which here is 0.09 against 0.37.

Keeping time is worth a lot. Q-max reaches **0.5585** MAP where the best descriptor
manages 0.2263 on the same 650 songs. 2D-FTM squashes a whole song into one vector
and can only ask whether two songs have a similar harmonic texture; Q-max keeps the
sequence and asks whether they walk the same path through it. It costs 47 minutes
against 3 seconds, because it is a dynamic program for each of 210,925 pairs.

Its ties column says 35%, which is worth explaining rather than ignoring. The score
is a discrete maximum, so pairs land on the same value: 210,925 pairs take only
37,646 distinct ones. Ties are broken by index order, and this collection is 92%
clique grouped, so index order lines up with the right answer and could be inflating
the score. It is not — re-scoring the same distances under five random shuffles of
the collection moves MAP by 0.0001. The ties sit in the tail, far below the ranks
these metrics are decided at.

## The network

It wins by a wide margin on the only collection it is allowed to be scored on:
**0.2926** against 0.0901 for the best classical method on the same 4,600 items. It
trains on 700 cliques, validates on 100 and tests on 200, split by clique so no song
ever appears on both sides, and the saved checkpoint is the epoch with the best
validation MAP rather than the last one.


# Dependencies Needed
- Numpy, scipy for all the array math, fast fourier transform, signal processing
- h5py reads Da-TACOS performance files in .h5 format
- pandas for joining data
- pytest for testing
  


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
