from typing import List
from collections import Counter
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import os


def tokenize(text: str) -> List[str]:
    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace(".", " <PERIOD> ")
    text = text.replace(",", " <COMMA> ")
    text = text.replace('"', " <QUOTATION_MARK> ")
    text = text.replace(";", " <SEMICOLON> ")
    text = text.replace("!", " <EXCLAMATION_MARK> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace("(", " <LEFT_PAREN> ")
    text = text.replace(")", " <RIGHT_PAREN> ")
    text = text.replace("--", " <HYPHENS> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(":", " <COLON> ")
    words = text.split()

    # Remove all words with  5 or fewer occurences
    word_counts: Counter = Counter(words)
    trimmed_words: List[str] = [word for word in words if word_counts[word] > 5]

    return trimmed_words


def plot_embeddings(model, int_to_vocab, viz_words=400, figsize=(16, 16)):
    """
    Plots a subset of word embeddings in a 2D space using t-SNE.

    Args:
        model: The trained SkipGram model containing the embeddings.
        int_to_vocab: Dictionary mapping word indices back to words.
        viz_words (int): Number of words to visualize.
        figsize (tuple): Size of the figure for the plot.
    """
    # Extract embeddings
    embeddings = model.in_embed.weight.to("cpu").data.numpy()

    # Reduce the dimensionality of embeddings with t-SNE
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color="steelblue")
        plt.annotate(
            int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7
        )

    plt.show()


def save_model(model, model_path="skipgram_model.pth"):
    """Save the trained SkipGram model to a file, creating the directory if it does not exist.

    Args:
        model: The trained SkipGram model.
        model_path: The path to save the model file, including directory and filename.

    Returns:
        The path where the model was saved.
    """
    # Extract the directory path from the model_path
    directory = os.path.dirname(model_path)

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), model_path)
    return model_path
