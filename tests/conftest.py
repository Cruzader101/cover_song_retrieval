"""Shared fixtures.

The synthetic corpus is session-scoped because building it costs a second or two
and nothing mutates it.
"""

import numpy as np
import pytest

from csr.data.synthetic import synthetic_corpus


@pytest.fixture(scope="session")
def corpus():
    """(manifest, {perf_id: (n_frames, 12)}) with 8 cliques of 4 plus 6 distractors."""
    return synthetic_corpus(n_cliques=8, per_clique=4, n_singletons=6, seed=0)


@pytest.fixture(scope="session")
def chroma(corpus):
    """One representative performance, (n_frames, 12)."""
    return corpus[1]["P_0_0"]


@pytest.fixture
def rng():
    return np.random.default_rng(0)
