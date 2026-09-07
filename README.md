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
