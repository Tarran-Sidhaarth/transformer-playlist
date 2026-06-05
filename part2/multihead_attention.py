import torch
import torch.nn as nn

CONTEXT_LENGTH = 4
EMBEDDING_DIMENSION = 6

generator = torch.Generator()
generator.manual_seed(42)


class AttentionHead:
    def __init__(self, embedding_dimension: int, head_dimension: int) -> None:
        """
        Initialises a single attention head with Q, K, V projection matrices.

        Args:
            embedding_dimension:  Dimensionality of the input token embeddings.
            head_dimension:       Dimensionality of this head's Q, K, V projections
                                  (typically embedding_dimension // number_heads).
        """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Runs scaled dot-product attention for this head.

        Args:
            X:  Input tensor of shape (context_length, embedding_dimension).

        Returns:
            A tensor of shape (context_length, head_dimension) after applying
            softmax-weighted aggregation of the value vectors.
        """
        return torch.zeros()
        
        


class MultiHeadAttention:

    def __init__(self, embedding_dimension: int, number_heads: int) -> None:
        """
        Initialises multi-head attention by creating N parallel attention heads
        and an output projection matrix W0.

        Args:
            embedding_dimension:  Dimensionality of the input token embeddings.
            number_heads:         Number of parallel attention heads to run.
                                  Must evenly divide embedding_dimension.
        """


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Runs all attention heads in parallel, concatenates their outputs,
        and projects back to embedding_dimension via W0.

        Args:
            X:  Input tensor of shape (context_length, embedding_dimension).

        Returns:
            A tensor of shape (context_length, embedding_dimension) after
            concatenating all head outputs and applying the W0 projection.
        """
        return torch.zeros()



X = torch.randn(CONTEXT_LENGTH, EMBEDDING_DIMENSION, generator=generator)

mha = MultiHeadAttention(embedding_dimension=EMBEDDING_DIMENSION, number_heads=3)
output = mha.forward(X)

print("Input shape: ", X.shape)
print("Output shape:", output)