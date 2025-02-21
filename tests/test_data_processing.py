import pytest
import torch
from torch.utils.data import DataLoader
from torch.nn import Embedding

from src.data_processing import (
    load_and_preprocess_data,
    create_lookup_tables,
    subsample_words,
    get_target,
    get_batches,
    cosine_similarity,
)


@pytest.mark.order(1)
def test_load_and_preprocess_data():
    # Given a sample file with known content
    test_file = "data/text8"

    # When the function is called
    examples = load_and_preprocess_data(test_file)

    if examples is None:
        pytest.skip()

    # Then it should return a list of SentimentExample objects
    assert isinstance(examples, list)
    assert all(isinstance(ex, str) for ex in examples)

    expected_results = ["anarchism", "originated", "as", "a", "term"]

    # Validate the first few tokens match expected results
    assert examples[: len(expected_results)] == expected_results


@pytest.mark.order(2)
def test_create_lookup_tables():
    # Given a list of sample words
    sample_words = ["hello", "world", "hello", "test"]

    # When create_lookup_tables is called
    vocab_to_int, int_to_vocab = create_lookup_tables(sample_words)

    if vocab_to_int is None or int_to_vocab is None:
        pytest.skip()

    # Then it should create accurate dictionaries for word to int and int to word mappings
    assert len(vocab_to_int) == len(int_to_vocab) == len(set(sample_words))
    assert all(word in vocab_to_int for word in sample_words)
    assert all(int_to_vocab[vocab_to_int[word]] == word for word in sample_words)
    assert all(
        isinstance(key, str) and isinstance(value, int)
        for key, value in vocab_to_int.items()
    )
    assert all(
        isinstance(key, int) and isinstance(value, str)
        for key, value in int_to_vocab.items()
    )


@pytest.fixture
def vocab_to_int():
    return {"apple": 0, "banana": 1, "cherry": 2, "date": 3}


@pytest.mark.order(3)
def test_subsample_words(vocab_to_int):
    words = ["apple", "banana", "cherry", "date", "apple", "banana", "banana"]
    train_words, freqs = subsample_words(words, vocab_to_int)

    if train_words is None or freqs is None:
        pytest.skip()

    assert isinstance(train_words, list)
    assert isinstance(freqs, dict)
    assert len(train_words) <= len(
        words
    )  # Train words should be less or equal to input words

    words = []
    train_words, freqs = subsample_words(words, vocab_to_int)

    assert isinstance(train_words, list)
    assert isinstance(freqs, dict)
    assert len(train_words) == 0  # Empty input should result in empty train words list

    words = [
        "apple",
        "apple",
        "apple",
        "banana",
        "banana",
        "cherry",
        "date",
        "date",
        "date",
        "date",
    ]
    train_words, freqs = subsample_words(words, vocab_to_int, threshold=0.5)

    assert isinstance(train_words, list)
    assert isinstance(freqs, dict)
    assert len(train_words) <= len(
        words
    )  # Train words should be less or equal to input words


@pytest.mark.order(4)
def test_get_target():
    # Given a list of sample words and a target index
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    idx = 4  # For the word "jumps"
    window_size = 2  # Test with a smaller window

    # When the get_target function is called
    context_words = get_target(words, idx, window_size)

    if context_words is None:
        pytest.skip()

    # Then the context words should be within the window around the index
    # For window_size=2, expected words around "jumps" are ["brown", "fox", "over", "the"]
    expected_words = ["brown", "fox", "over", "the"]
    assert set(context_words).issubset(
        set(expected_words)
    ), "The context words are not correctly selected within the window."


@pytest.mark.order(5)
def test_get_batches():
    # Example input
    words = list(range(100))  # A simple list of integers as stand-ins for words
    batch_size = 10
    window_size = 2

    batches = list(get_batches(words, batch_size, window_size))

    if batches is None:
        pytest.skip()

    # Check that the number of batches is correct
    expected_num_batches = len(words) // batch_size
    assert len(batches) == expected_num_batches, "Incorrect number of batches."

    # Check the first batch structure and content
    inputs, targets = batches[0]
    assert len(inputs) > 0, "Inputs of the first batch are empty."
    assert len(targets) > 0, "Targets of the first batch are empty."
    assert len(inputs) == len(
        targets
    ), "Inputs and targets of the first batch do not match in length."

    # Additional checks can be made for the contents of inputs and targets
    # to ensure context windows are correctly implemented.


@pytest.mark.order(6)
def test_cosine_similarity():
    # Create a simple embedding matrix for testing: 1000 embeddings of size 10
    num_embeddings = 2000
    embedding_dim = 10
    embedding = Embedding(num_embeddings, embedding_dim)

    # Call the cosine_similarity function
    valid_size = 16  # Number of validation examples
    valid_window = 100  # Window of validation examples
    valid_examples, similarities = cosine_similarity(
        embedding, valid_size=valid_size, valid_window=valid_window, device="cpu"
    )

    if valid_examples is None or similarities is None:
        pytest.skip()

    # Check the shape of the output tensors
    assert (
        valid_examples.shape[0] == valid_size
    ), "The number of valid examples should match the requested valid_size."
    assert (
        similarities.shape[0] == valid_size
    ), "The similarities tensor should have one row per valid example."
    assert (
        similarities.shape[1] == num_embeddings
    ), "The similarities tensor should have one column per embedding."
