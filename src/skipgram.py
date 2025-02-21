import torch
from torch import nn
import torch.optim as optim


class SkipGramNeg(nn.Module):
    """A SkipGram model with Negative Sampling.

    This module implements a SkipGram model using negative sampling. It includes
    embedding layers for input and output words and initializes these embeddings
    with a uniform distribution to aid in convergence.

    Attributes:
        n_vocab: An integer count of the vocabulary size.
        n_embed: An integer specifying the dimensionality of the embeddings.
        noise_dist: A tensor representing the distribution of noise words.
        in_embed: The embedding layer for input words.
        out_embed: The embedding layer for output words.
    """

    def __init__(self, n_vocab: int, n_embed: int, noise_dist: torch.Tensor = None):
        """Initializes the SkipGramNeg model with given vocabulary size, embedding size, and noise distribution.

        Args:
            n_vocab: The size of the vocabulary.
            n_embed: The size of each embedding vector.
            noise_dist: The distribution of noise words for negative sampling.
        """
        super().__init__()
        self.n_vocab: int = n_vocab
        self.n_embed: int = n_embed
        self.noise_dist: torch.Tensor = noise_dist

        # Define embedding layers for input and output words
        self.in_embed: nn.Embedding = nn.Embedding(n_vocab, n_embed)
        self.out_embed: nn.Embedding = nn.Embedding(
            n_vocab, n_embed
        )  # Esto creo que estará mal

        # Initialize embedding tables with uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words: torch.Tensor) -> torch.Tensor:
        """Fetches input vectors for a batch of input words.

        Args:
            input_words: A tensor of integers representing input words.

        Returns:
            A tensor containing the input vectors for the given words.
        """
        input_vectors: torch.Tensor = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words: torch.Tensor) -> torch.Tensor:
        """Fetches output vectors for a batch of output words.

        Args:
            output_words: A tensor of integers representing output words.

        Returns:
            A tensor containing the output vectors for the given words.
        """
        output_vectors: torch.Tensor = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self, batch_size: int, n_samples: int) -> torch.Tensor:
        """Generates noise vectors for negative sampling.

        Args:
            batch_size: The number of words in each batch.
            n_samples: The number of negative samples to generate per word.

        Returns:
            A tensor of noise vectors with shape (batch_size, n_samples, n_embed).
        """
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist: torch.Tensor = torch.ones(self.n_vocab)
        else:
            noise_dist: torch.Tensor = self.noise_dist

        # Sample words from our noise distribution
        noise_words: torch.Tensor = torch.multinomial(
            noise_dist, batch_size * n_samples
        )

        device: str = "cuda" if self.out_embed.weight.is_cuda else "cpu"
        noise_words: torch.Tensor = noise_words.to(device)

        # Reshape output vectors to size (batch_size, n_samples, n_embed)
        noise_words = noise_words.view(
            batch_size, n_samples
        )  # Para que cada batch tenga n_samples
        # noise_vectors: torch.Tensor = noise_words.reshape((batch_size, n_samples, self.n_embed))  # ESTO ESTÁ MAL, NO TIENE SENTIDO
        noise_vectors: torch.Tensor = self.out_embed(
            noise_words
        )  # Esto tiene el size que queremos

        return noise_vectors


class NegativeSamplingLoss(nn.Module):
    """Implements the Negative Sampling loss as a PyTorch module.

    This loss is used for training word embedding models like Word2Vec using
    negative sampling. It computes the loss as the sum of the log-sigmoid of
    the dot product of input and output vectors (for positive samples) and the
    log-sigmoid of the dot product of input vectors and noise vectors (for
    negative samples), across a batch.
    """

    def __init__(self):
        """Initializes the NegativeSamplingLoss module."""
        super().__init__()

    def forward(
        self,
        input_vectors: torch.Tensor,
        output_vectors: torch.Tensor,
        noise_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the Negative Sampling loss.

        Args:
            input_vectors: A tensor containing input word vectors,
                            shape (batch_size, embed_size).
            output_vectors: A tensor containing output word vectors (positive samples),
                            shape (batch_size, embed_size).
            noise_vectors: A tensor containing vectors for negative samples,
                            shape (batch_size, n_samples, embed_size).

        Returns:
            A tensor containing the average loss for the batch.
        """

        # Compute log-sigmoid loss for correct classifications
        out_loss = torch.log(torch.sigmoid((input_vectors * output_vectors).sum(dim=1)))

        # Compute log-sigmoid loss for incorrect classifications
        noise_loss = torch.log(
            torch.sigmoid(
                torch.bmm(noise_vectors, output_vectors.unsqueeze(2)).squeeze(2)
            )
        )
        noise_loss = noise_loss.mean(dim=1)
        # Return the negative sum of the correct and noisy log-sigmoid losses, averaged over the batch
        return -(out_loss + noise_loss).mean()
