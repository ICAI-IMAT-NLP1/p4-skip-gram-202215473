import torch
import pytest
from src.skipgram import SkipGramNeg, NegativeSamplingLoss


@pytest.mark.order(7)
def test_skipgramneg_init():
    n_vocab = 100
    n_embed = 50
    model = SkipGramNeg(n_vocab, n_embed)

    if model is None:
        pytest.skip()

    assert isinstance(model, SkipGramNeg), "Model instance is not a SkipGramNeg."
    assert model.n_vocab == n_vocab, "Vocabulary size does not match."
    assert model.n_embed == n_embed, "Embedding size does not match."
    assert model.in_embed.embedding_dim == n_embed, "Input embedding size is incorrect."
    assert (
        model.out_embed.embedding_dim == n_embed
    ), "Output embedding size is incorrect."
    assert (
        model.in_embed.num_embeddings == n_vocab
    ), "Input embedding vocabulary size is incorrect."
    assert (
        model.out_embed.num_embeddings == n_vocab
    ), "Output embedding vocabulary size is incorrect."


@pytest.mark.order(8)
def test_forward_input():
    n_vocab = 100
    n_embed = 50
    model = SkipGramNeg(n_vocab, n_embed)
    input_words = torch.randint(0, n_vocab, (10,))  # A batch of 10 random word indices

    input_vectors = model.forward_input(input_words)
    assert input_vectors.shape == (10, n_embed), "Input vectors shape is incorrect."


@pytest.mark.order(9)
def test_forward_output():
    n_vocab = 100
    n_embed = 50
    model = SkipGramNeg(n_vocab, n_embed)
    output_words = torch.randint(0, n_vocab, (10,))  # A batch of 10 random word indices

    output_vectors = model.forward_output(output_words)

    if output_vectors is None:
        pytest.skip()

    assert output_vectors.shape == (10, n_embed), "Output vectors shape is incorrect."


@pytest.mark.order(10)
def test_forward_noise():
    n_vocab = 100
    n_embed = 50
    batch_size = 10
    n_samples = 5
    model = SkipGramNeg(n_vocab, n_embed)

    noise_vectors = model.forward_noise(batch_size, n_samples)

    if noise_vectors is None:
        pytest.skip()

    expected_shape = (batch_size, n_samples, n_embed)
    assert noise_vectors.shape == expected_shape, "Noise vectors shape is incorrect."


@pytest.mark.order(11)
def test_negative_sampling_loss_forward():
    # Initialize the loss module
    loss_module = NegativeSamplingLoss()

    # Create a batch of input vectors, output vectors, and noise vectors
    batch_size, embed_size, n_samples = 2, 5, 3
    torch.manual_seed(42)  # For reproducibility
    input_vectors = torch.rand(batch_size, embed_size)  # Random input vectors
    output_vectors = torch.rand(
        batch_size, embed_size
    )  # Random output vectors (positive samples)
    noise_vectors = torch.rand(
        batch_size, n_samples, embed_size
    )  # Random noise vectors (negative samples)

    # Compute the loss
    loss = loss_module(input_vectors, output_vectors, noise_vectors)

    if loss is None:
        pytest.skip()

    # Check that the loss is a single scalar value
    assert loss.dim() == 0, "Loss is not a scalar."

    # Check that the loss is a positive value
    # Note: This does not check correctness of the loss computation, only that it produces a plausible value.
    assert loss.item() > 0, "Loss should be positive."
