import torch
import torch.optim as optim
from typing import List, Dict

try:
    from src.skipgram import SkipGramNeg, NegativeSamplingLoss  
    from src.data_processing import get_batches, cosine_similarity
except ImportError:
    from skipgram import SkipGramNeg, NegativeSamplingLoss
    from data_processing import get_batches, cosine_similarity

def train_skipgram(model: SkipGramNeg,
                   words: List[int], 
                   int_to_vocab: Dict[int, str], 
                   batch_size=512, 
                   epochs=5, 
                   learning_rate=0.003, 
                   window_size=5, 
                   print_every=1500,
                   device='cpu'):
    """Trains the SkipGram model using negative sampling.

    Args:
        model: The SkipGram model to be trained.
        words: A list of words (integers) to train on.
        int_to_vocab: A dictionary mapping integers back to vocabulary words.
        batch_size: The size of each batch of input and target words.
        epochs: The number of epochs to train for.
        learning_rate: The learning rate for the optimizer.
        window_size: The size of the context window for generating training pairs.
        print_every: The frequency of printing the training loss and validation examples.
        device: The device (CPU or GPU) where the tensors will be allocated.
    """
    # Define loss and optimizer
    criterion = NegativeSamplingLoss()  # 1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    # Training loop
    for epoch in range(epochs):
        batches = get_batches(words, batch_size, window_size)  # 2
        for input_words, target_words in batches:
            steps += 1
            # Convert inputs and context words into tensors  # 3 (a)
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
            inputs, targets = inputs.to(device), targets.to(device)

            # input, output, and noise vectors  # 3 (b)
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            batch_size_here = inputs.size(0)
            noise_vectors = model.forward_noise(batch_size_here, len(inputs))
            
            # negative sampling loss  # 3(c)
            loss = criterion(input_vectors, output_vectors, noise_vectors)

            # Backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % print_every == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Step: {steps}, Loss: {loss.item()}")
                # Cosine similarity
                valid_examples, valid_similarities = cosine_similarity(model.in_embed.weight, device=device)
                _, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")
