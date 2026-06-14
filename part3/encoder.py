import torch
import torch.nn as nn

CONTEXT_LENGTH = 4
EMBEDDING_DIMENSION = 6
NUMBER_HEADS = 3
FFN_HIDDEN_DIMENSION = 24  # typically 4x embedding dimension
VOCAB_SIZE = 1000  # size of output vocabulary


# ---------------------------------------------------------
# Encoder components (from part 2)
# ---------------------------------------------------------

class AttentionHead(nn.Module):
    def __init__(self, embedding_dimension: int, head_dimension: int, masked: bool = False) -> None:
        super().__init__()
        self.head_dimension = head_dimension
        self.masked = masked
        self.W_Q = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.W_K = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.W_V = nn.Linear(embedding_dimension, head_dimension, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        similarity = (Q @ K.transpose(-2, -1)) / (self.head_dimension ** 0.5)

# Difference
        if self.masked:
            seq_len = X.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            similarity = similarity.masked_fill(mask, float('-inf'))

    
        scaled_similarity = similarity.softmax(dim=-1)
        if self.masked:
            print("Masked Simlarity: ",similarity)
            print("Masked Simlarity after softmax : ",scaled_similarity)

        return scaled_similarity @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, masked: bool = False) -> None:
        super().__init__()
        self.head_dimension = embedding_dimension // number_heads
        self.heads = nn.ModuleList([
            AttentionHead(embedding_dimension, self.head_dimension, masked=masked)
            for _ in range(number_heads)
        ])
        self.W0 = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([head(X) for head in self.heads], dim=-1)
        return self.W0(concatenated)


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, embedding_dimension)
        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(X)))


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, ffn_hidden_dimension: int) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dimension, number_heads)
        self.feed_forward = FeedForward(embedding_dimension, ffn_hidden_dimension)
        self.layer_norm1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm2 = nn.LayerNorm(embedding_dimension)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layer_norm1(X + self.attention(X))
        X = self.layer_norm2(X + self.feed_forward(X))
        return X


class Encoder(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, ffn_hidden_dimension: int, number_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dimension, number_heads, ffn_hidden_dimension)
            for _ in range(number_layers)
        ])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X)
        return X


