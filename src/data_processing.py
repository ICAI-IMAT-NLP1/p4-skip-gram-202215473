from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch
import random

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file

    # Preprocess and tokenize the text
    tokens: List[str] = tokenize(text)
    return tokens


def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.

    sorted_vocab: List[str] = sorted(word_counts, key=word_counts.get, reverse=True)

    # Create int_to_vocab and vocab_to_int dictionaries.
    vocab_to_int: Dict[str, int] = {word: idx for idx, word in enumerate(sorted_vocab)}
    int_to_vocab: Dict[int, str] = {idx: word for word, idx in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def subsample_words(
    words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5
) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the
    presence of frequent words according to Mikolov's subsampling technique. This method
    calculates the probability of keeping each word in the dataset based on its frequency,
    with more frequent words having a higher chance of being discarded. The process helps
    in balancing the word distribution, potentially leading to faster training and better
    representations by focusing more on less frequent words.

    Args:
        words (list): List of words to be subsampled.
        vocab_to_int (dict): Dictionary mapping words to unique integers.
        threshold (float): Threshold parameter controlling the extent of subsampling.

    Returns:
        List[int]: A list of integers representing the subsampled words, where some high-frequency words may be removed.
        Dict[str, float]: Dictionary associating each word with its frequency.
    """
    # Convert words to integers
    counter_words: Counter = Counter(words)
    total = len(words)
    freqs: Dict[str, float] = {w: count / total for w, count in counter_words.items()}
    # freqs: Dict[str, float] = {w: 1 - (threshold/(count/total))**(1/2) for w,count in counter_words.items()}

    P_words = {wi: 1 - (threshold / freqs[wi]) ** (1 / 2) for wi in freqs.keys()}
    # P_words = 1 - torch.sqrt(threshold/freqs.values())
    train_words_str: List[str] = [
        wi if P_words[wi] <= threshold else None for wi in P_words.keys()
    ]
    # train_words_str: List[str] = [wi if freqs[wi] <= threshold else None for wi in freqs.keys()]
    train_words: List[int] = [vocab_to_int[w] for w in train_words_str if w is not None]
    print(len(train_words_str), len(train_words))
    return train_words, freqs


def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """
    R: int = random.randint(1, window_size)
    start: int = max(0, idx - R)
    end: int = min(len(words), idx + R + 1)
    target_words: List[str] = [words[i] for i in range(start, end)]
    target_words.remove(words[idx])
    return target_words


def get_batches(
    words: List[int], batch_size: int, window_size: int = 5
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """

    vocab_to_int, _ = create_lookup_tables(words)
    for idx in range(0, len(words), batch_size):
        targets_str: List[str] = get_target(words, idx, window_size)
        inputs: List[int] = [words[idx] for _ in targets_str]
        targets: List[int] = [vocab_to_int[w] for w in targets_str]
        yield inputs, targets

    # # Recorremos cada índice de la lista de palabras
    # for idx in range(0, len(words)):
    #     # Usamos get_target para obtener el contexto de la palabra en posición idx
    #     context_words = get_target(words, idx, window_size)
    #     for target in context_words:
    #         pairs.append((words[idx], target))
    #         # Cuando se alcanza el tamaño del batch, se rinde el batch y se reinicia la lista de pares.
    #         if len(pairs) == batch_size:
    #             inputs, targets = zip(*pairs)  # Se separan las dos listas
    #             yield list(inputs), list(targets)
    #             pairs = []  # Reiniciar el acumulador


def cosine_similarity(
    embedding: torch.nn.Embedding,
    valid_size: int = 16,
    valid_window: int = 100,
    device: str = "cpu",
):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    embedding_device: torch.Tensor = embedding.weight.to(device)
    embedding_device_normal: torch.Tensor = embedding_device / torch.norm(
        embedding_device, dim=1, keepdim=True
    )
    valid_examples: torch.Tensor = torch.randint(
        0, valid_window, (valid_size,), device=device
    )
    valid_embeddings: torch.Tensor = embedding_device[valid_examples]
    valid_embeddings_normal: torch.Tensor = valid_embeddings / torch.norm(
        valid_embeddings.to(torch.float), dim=1, keepdim=True
    )
    similarities: torch.Tensor = torch.matmul(
        valid_embeddings_normal, embedding_device_normal.T
    )

    return valid_examples, similarities
