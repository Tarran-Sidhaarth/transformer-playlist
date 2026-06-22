import torch
import torch.nn as nn
from encoder import MultiHeadAttention, FeedForward

# ---------------------------------------------------------
# Cross-attention (decoder attends to encoder output)
# ---------------------------------------------------------

class CrossAttentionHead(nn.Module):
    def __init__(self, embedding_dimension: int, head_dimension: int) -> None:
        super().__init__()
        self.head_dimension = head_dimension
        self.W_Q = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.W_K = nn.Linear(embedding_dimension, head_dimension, bias=False)
        self.W_V = nn.Linear(embedding_dimension, head_dimension, bias=False)

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: #type:ignore
        Q = self.W_Q(X)                 # queries from decoder
        K = self.W_K(encoder_output)    # keys from encoder
        V = self.W_V(encoder_output)    # values from encoder

        similarity = (Q @ K.transpose(-2, -1)) / (self.head_dimension ** 0.5)

        if mask is not None:
            # mask: (B,1,S_enc) - blocks attending to PAD positions in the encoder output
            similarity = similarity.masked_fill(~mask, float('-inf'))

        scaled_similarity = similarity.softmax(dim=-1)
        return scaled_similarity @ V


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int) -> None:
        super().__init__()
        head_dimension = embedding_dimension // number_heads
        self.heads = nn.ModuleList([
            CrossAttentionHead(embedding_dimension, head_dimension)
            for _ in range(number_heads)
        ])
        self.W0 = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: #type:ignore
        concatenated = torch.cat([head(X, encoder_output, mask) for head in self.heads], dim=-1)
        return self.W0(concatenated)


# ---------------------------------------------------------
# Decoder
# ---------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, ffn_hidden_dimension: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.masked_attention = MultiHeadAttention(embedding_dimension, number_heads)
        self.cross_attention = CrossMultiHeadAttention(embedding_dimension, number_heads)
        self.feed_forward = FeedForward(embedding_dimension, ffn_hidden_dimension, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm2 = nn.LayerNorm(embedding_dimension)
        self.layer_norm3 = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: torch.Tensor = None, cross_attn_mask: torch.Tensor = None) -> torch.Tensor: #type:ignore
        # sublayer 1: masked self-attention (causal + padding) + residual + layer norm
        X = self.layer_norm1(X + self.dropout(self.masked_attention(X, self_attn_mask)))

        # sublayer 2: cross-attention (Q from decoder, K/V from encoder) + residual + layer norm
        X = self.layer_norm2(X + self.dropout(self.cross_attention(X, encoder_output, cross_attn_mask)))

        # sublayer 3: feed forward + residual + layer norm
        X = self.layer_norm3(X + self.dropout(self.feed_forward(X)))

        return X


class Decoder(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, ffn_hidden_dimension: int, number_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(embedding_dimension, number_heads, ffn_hidden_dimension, dropout)
            for _ in range(number_layers)
        ])

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: torch.Tensor = None, cross_attn_mask: torch.Tensor = None) -> torch.Tensor: #type:ignore
        for layer in self.layers:
            X = layer(X, encoder_output, self_attn_mask, cross_attn_mask)
        return X


# ---------------------------------------------------------
# Output layer / LM head
# ---------------------------------------------------------

class OutputLayer(nn.Module):
    """Final linear projection to vocab logits. No softmax here -
    nn.CrossEntropyLoss expects raw logits and applies log-softmax
    internally; softmaxing before computing the loss is unstable and
    double-applies the normalization. Softmax/argmax only happen at
    inference time (see Seq2SeqTransformer.generate in model.py)."""
    def __init__(self, embedding_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dimension, vocab_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)