"""The learned embedding, checked before any training time is spent."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.needs_torch

from csr.models import ChromaCNN, CosineHead, Crops, embed


def test_output_shape_and_unit_norm():
    model = ChromaCNN(dim=64, width=8).eval()
    with torch.no_grad():
        out = model(torch.randn(3, 1, 12, 400))
    assert out.shape == (3, 64)
    assert torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5)


@pytest.mark.parametrize("semitones", range(12))
def test_transposition_invariance_holds_at_random_init(semitones):
    """The architecture claims exact key invariance, so it must hold before
    training rather than being something training has to discover."""
    model = ChromaCNN(dim=64, width=8).eval()
    x = torch.randn(2, 1, 12, 256)
    with torch.no_grad():
        assert torch.allclose(
            model(torch.roll(x, semitones, dims=2)), model(x), atol=1e-5
        )


@pytest.mark.parametrize("length", [64, 150, 901])
def test_variable_length_input(length):
    """Songs differ in length and are never truncated at inference."""
    model = ChromaCNN(dim=64, width=8).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 12, length)).shape == (1, 64)


def test_cosine_head_logits_are_scaled_cosines():
    head = CosineHead(dim=8, n_classes=4, scale=10.0)
    e = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    logits = head(e)
    assert logits.shape == (3, 4)
    assert logits.abs().max() <= 10.0 + 1e-4


def test_crops_return_fixed_length_regardless_of_song_length():
    chroma = {"a": np.random.rand(50, 12).astype(np.float32),
              "b": np.random.rand(2000, 12).astype(np.float32)}
    ds = Crops(["a", "b"], [0, 1], chroma, crop=128, train=True)
    for i in range(2):
        x, y = ds[i]
        assert x.shape == (1, 12, 128)


def test_embed_pads_a_mixed_length_batch(corpus):
    _, chroma = corpus
    model = ChromaCNN(dim=32, width=8).eval()
    ids = ["P_0_0", "P_0_1", "P_1_0"]
    vectors = embed(model, ids, {k: chroma[k][:600] for k in ids}, device="cpu", batch=2)
    assert vectors.shape == (3, 32)
    assert np.isfinite(vectors).all()


def test_an_embedding_does_not_depend_on_who_shares_its_batch(corpus):
    """A performance must embed the same way alone and beside a longer one.

    It did not. Batching pads the short members, and zeros stop being neutral the
    moment batch norm shifts them -- the ReLU keeps what survives and the result
    leaks back through the convolutions into the last real column. The same
    performance came out at a cosine of 0.68 with itself, and the factor analysis
    read that as the network being unusually sensitive to a change in length.

    On CPU this is exact. The tolerance is for float32 kernel choice, not for the
    effect this test exists to catch, which was four orders of magnitude larger.
    """
    _, chroma = corpus
    model = ChromaCNN(dim=32, width=8).eval()
    ids = ["P_0_0", "P_0_1", "P_1_0", "P_1_1"]
    lengths = [400, 1200, 780, 2000]
    clipped = {p: chroma[p][:n] for p, n in zip(ids, lengths)}

    one_at_a_time = embed(model, ids, clipped, device="cpu", batch=1)
    together = embed(model, ids, clipped, device="cpu", batch=4)
    assert np.abs(one_at_a_time - together).max() < 1e-5

    # And embedding a subset must not move the vectors that survive into it.
    subset = embed(model, ids[:2], clipped, device="cpu", batch=4)
    assert np.abs(subset - one_at_a_time[:2]).max() < 1e-5


def test_embed_returns_vectors_in_the_order_it_was_asked_for(corpus):
    """Bucketing by width reorders the work; it must not reorder the answer."""
    _, chroma = corpus
    model = ChromaCNN(dim=32, width=8).eval()
    ids = ["P_0_0", "P_0_1", "P_1_0", "P_1_1"]
    clipped = {p: chroma[p][:n] for p, n in zip(ids, [400, 1200, 780, 2000])}

    everything = embed(model, ids, clipped, device="cpu", batch=2)
    for position, pid in enumerate(ids):
        alone = embed(model, [pid], clipped, device="cpu", batch=1)[0]
        assert np.abs(everything[position] - alone).max() < 1e-5


def test_model_can_overfit_a_tiny_batch():
    """Catches a broken training loop in seconds: 8 samples, 2 classes."""
    torch.manual_seed(0)
    model, head = ChromaCNN(dim=32, width=8), CosineHead(32, 2)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=3e-3)
    x = torch.randn(8, 1, 12, 128)
    y = torch.tensor([0, 1] * 4)
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(head(model(x)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
