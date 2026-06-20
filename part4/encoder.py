import torch
import torch.nn as nn

# ---------------------------------------------------------
# Self-attention. Masking is now ALWAYS provided externally via `mask`
# rather than computed inside the head. This is what lets the same
# AttentionHead/MultiHeadAttention serve three different jobs:
#   - encoder self-attention   -> mask = padding-only
#   - decoder self-attention   -> mask = causal AND padding
#   - (cross-attention has its own head class in decoder.py, same idea)
# Convention: mask is bool, True = allowed to attend, False = blocked.
# ---------------------------------------------------------


class AttentionHead(nn.Module):
    def __init__(self, embedding_dimension: int, head_dimension: int) -> None:
        super().__init__()
        self.head_dimension = head_dimension
        self.W_Q = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.W_K = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.W_V = nn.Linear(embedding_dimension, head_dimension, bias=False)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: #type:ignore
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        similarity = (Q @ K.transpose(-2, -1)) / (self.head_dimension ** 0.5)

        if mask is not None:
            # mask broadcasts: (B,1,S_k) for padding-only, or (B,S_q,S_k) for causal+padding
            similarity = similarity.masked_fill(~mask, float('-inf'))

        scaled_similarity = similarity.softmax(dim=-1)
        return scaled_similarity @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int) -> None:
        super().__init__()
        self.head_dimension = embedding_dimension // number_heads
        self.heads = nn.ModuleList([
            AttentionHead(embedding_dimension, self.head_dimension)
            for _ in range(number_heads)
        ])
        self.W0 = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: #type:ignore
        concatenated = torch.cat([head(X, mask) for head in self.heads], dim=-1)
        return self.W0(concatenated)


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, embedding_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.relu(self.fc1(X))))


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, ffn_hidden_dimension: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dimension, number_heads)
        self.feed_forward = FeedForward(embedding_dimension, ffn_hidden_dimension, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm2 = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: #type:ignore
        # sublayer 1: self-attention + residual + layer norm (post-LN, as in the paper)
        X = self.layer_norm1(X + self.dropout(self.attention(X, mask)))
        # sublayer 2: feed forward + residual + layer norm
        X = self.layer_norm2(X + self.dropout(self.feed_forward(X)))
        return X


class Encoder(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, ffn_hidden_dimension: int, number_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dimension, number_heads, ffn_hidden_dimension, dropout)
            for _ in range(number_layers)
        ])

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: #type:ignore
        for layer in self.layers:
            X = layer(X, mask)
        return X