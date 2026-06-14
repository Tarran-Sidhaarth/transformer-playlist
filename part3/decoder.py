import torch 
import torch.nn as nn
from encoder import MultiHeadAttention,FeedForward,Encoder, CONTEXT_LENGTH,EMBEDDING_DIMENSION,NUMBER_HEADS,FFN_HIDDEN_DIMENSION, VOCAB_SIZE

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

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(X)                 # queries from decoder
        K = self.W_K(encoder_output)    # keys from encoder
        V = self.W_V(encoder_output)    # values from encoder

        similarity = (Q @ K.transpose(-2, -1)) / (self.head_dimension ** 0.5)
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

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([head(X, encoder_output) for head in self.heads], dim=-1)
        return self.W0(concatenated)


# ---------------------------------------------------------
# Decoder
# ---------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, ffn_hidden_dimension: int) -> None:
        super().__init__()
        self.masked_attention = MultiHeadAttention(embedding_dimension, number_heads, masked=True)
        self.cross_attention = CrossMultiHeadAttention(embedding_dimension, number_heads)
        self.feed_forward = FeedForward(embedding_dimension, ffn_hidden_dimension)
        self.layer_norm1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm2 = nn.LayerNorm(embedding_dimension)
        self.layer_norm3 = nn.LayerNorm(embedding_dimension)

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # sublayer 1: masked self-attention + residual + layer norm
        X = self.layer_norm1(X + self.masked_attention(X))

        # sublayer 2: cross-attention (Q from decoder, K/V from encoder) + residual + layer norm
        X = self.layer_norm2(X + self.cross_attention(X, encoder_output))

        # sublayer 3: feed forward + residual + layer norm
        X = self.layer_norm3(X + self.feed_forward(X))

        return X


class Decoder(nn.Module):
    def __init__(self, embedding_dimension: int, number_heads: int, ffn_hidden_dimension: int, number_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(embedding_dimension, number_heads, ffn_hidden_dimension)
            for _ in range(number_layers)
        ])

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, encoder_output)
        return X


# ---------------------------------------------------------
# Output layer
# ---------------------------------------------------------

class OutputLayer(nn.Module):
    """Final linear projection + softmax to get token probabilities."""
    def __init__(self, embedding_dimension: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dimension, vocab_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.linear(X)
        probabilities = logits.softmax(dim=-1)
        return probabilities


# ---------------------------------------------------------
# Putting it all together
# ---------------------------------------------------------

batch_size = 2

# encoder input: input embeddings (e.g. source sentence) + positional embedding (from part 1)
encoder_input = torch.randn(batch_size, CONTEXT_LENGTH, EMBEDDING_DIMENSION)

# decoder input: output embeddings (shifted-right target sentence) + positional embedding
decoder_input = torch.randn(batch_size, CONTEXT_LENGTH, EMBEDDING_DIMENSION)

encoder = Encoder(
    embedding_dimension=EMBEDDING_DIMENSION,
    number_heads=NUMBER_HEADS,
    ffn_hidden_dimension=FFN_HIDDEN_DIMENSION,
    number_layers=2
)

decoder = Decoder(
    embedding_dimension=EMBEDDING_DIMENSION,
    number_heads=NUMBER_HEADS,
    ffn_hidden_dimension=FFN_HIDDEN_DIMENSION,
    number_layers=2
)

output_layer = OutputLayer(EMBEDDING_DIMENSION, VOCAB_SIZE)

# 1. encoder processes the source sequence
encoder_output = encoder(encoder_input)

# 2. decoder processes the target sequence, using encoder_output for cross-attention
decoder_output = decoder(decoder_input, encoder_output)

# 3. project to vocabulary size and turn into probabilities
probabilities = output_layer(decoder_output)

# 4. pick the most likely token at each position
predicted_token_ids = probabilities.argmax(dim=-1)

print("Encoder input shape: ", encoder_input.shape)     # [2, 4, 6]
print("Encoder output shape:", encoder_output.shape)    # [2, 4, 6]
print("Decoder input shape: ", decoder_input.shape)      # [2, 4, 6]
print("Decoder output shape:", decoder_output.shape)    # [2, 4, 6]
print("Probabilities shape: ", probabilities.shape)      # [2, 4, 1000]
print("Predicted token IDs shape:", predicted_token_ids.shape)  # [2, 4]
print("Predicted token IDs:\n", predicted_token_ids)

total_params = (
    sum(p.numel() for p in encoder.parameters())
    + sum(p.numel() for p in decoder.parameters())
    + sum(p.numel() for p in output_layer.parameters())
)
print("Total params:", total_params)