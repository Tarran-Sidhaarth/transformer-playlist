import math
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder, OutputLayer
from data_preparation import TOTAL_VOCAB_SIZE, BOS_ID, EOS_ID, PAD_ID, MAX_SRC_LEN, MAX_TGT_LEN


class PositionalEmbedding(nn.Module):
    """Token embedding (scaled by sqrt(d_model), per the paper) + the fixed
    sinusoidal positional encoding from 'Attention Is All You Need' sec 3.5.
    If your part 1 already implemented positional encoding, swap that
    module in here instead - this is the standard paper version."""

    def __init__(self, vocab_size: int, embedding_dimension: int, max_len: int,
                 dropout: float = 0.1, pad_id: int = PAD_ID) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=pad_id)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dimension, 2).float()
                              * (-math.log(10000.0) / embedding_dimension))
        pe = torch.zeros(max_len, embedding_dimension)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe, persistent=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[1]
        x = self.token_embedding(token_ids) * math.sqrt(self.embedding_dimension)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0) #type:ignore
        return self.dropout(x)


def make_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """(seq_len, seq_len) bool, True = allowed to attend (key position <= query position)."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


class Transformer(nn.Module):
    def __init__(self, vocab_size: int = TOTAL_VOCAB_SIZE, embedding_dimension: int = 512,
                 number_heads: int = 8, ffn_hidden_dimension: int = 2048,
                 number_encoder_layers: int = 6, number_decoder_layers: int = 6,
                 max_src_len: int = MAX_SRC_LEN, max_tgt_len: int = MAX_TGT_LEN,
                 dropout: float = 0.1, pad_id: int = PAD_ID, tie_weights: bool = True) -> None:
        super().__init__()
        self.pad_id = pad_id
        max_len = max(max_src_len, max_tgt_len)

        self.embedding = PositionalEmbedding(vocab_size, embedding_dimension, max_len, dropout, pad_id)
        self.encoder = Encoder(embedding_dimension, number_heads, ffn_hidden_dimension, number_encoder_layers, dropout)
        self.decoder = Decoder(embedding_dimension, number_heads, ffn_hidden_dimension, number_decoder_layers, dropout)
        self.output_layer = OutputLayer(embedding_dimension, vocab_size)

        if tie_weights:
            # Share the input embedding matrix with the output projection (sec 3.4 of the
            # paper). Cuts ~vocab_size * embedding_dimension params and tends to help
            # generalization, at the cost of requiring embedding_dimension to match
            # the output layer's input size (already true here).
            self.output_layer.linear.weight = self.embedding.token_embedding.weight

    def make_key_padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, S) bool, True = real token, False = PAD."""
        return ids != self.pad_id

    def forward(self, encoder_input_ids: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        device = encoder_input_ids.device

        enc_key_mask = self.make_key_padding_mask(encoder_input_ids).unsqueeze(1)   # (B,1,S_enc)
        dec_key_mask = self.make_key_padding_mask(decoder_input_ids)                # (B,S_dec)

        causal = make_causal_mask(decoder_input_ids.shape[1], device=device)        # (S_dec,S_dec)
        decoder_self_attn_mask = causal.unsqueeze(0) & dec_key_mask.unsqueeze(1)    # (B,S_dec,S_dec)

        encoder_output = self.encoder(self.embedding(encoder_input_ids), enc_key_mask)
        decoder_output = self.decoder(self.embedding(decoder_input_ids), encoder_output,
                                       decoder_self_attn_mask, enc_key_mask)

        return self.output_layer(decoder_output)  # raw logits: (B, S_dec, vocab_size)

    @torch.no_grad()
    def generate(self, encoder_input_ids: torch.Tensor, max_len: int = MAX_TGT_LEN,
                 bos_id: int = BOS_ID, eos_id: int = EOS_ID) -> torch.Tensor:
        """Greedy autoregressive decoding, one token at a time. No KV-cache,
        so it recomputes the full prefix every step - simple and easy to
        follow, but slow for large-scale evaluation. Swap in KV-caching or
        beam search later if you want faster/better generation."""
        self.eval()
        device = encoder_input_ids.device
        batch_size = encoder_input_ids.shape[0]

        enc_key_mask = self.make_key_padding_mask(encoder_input_ids).unsqueeze(1)
        encoder_output = self.encoder(self.embedding(encoder_input_ids), enc_key_mask)

        decoder_input_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            dec_key_mask = self.make_key_padding_mask(decoder_input_ids)
            causal = make_causal_mask(decoder_input_ids.shape[1], device=device)
            self_attn_mask = causal.unsqueeze(0) & dec_key_mask.unsqueeze(1)

            decoder_output = self.decoder(self.embedding(decoder_input_ids), encoder_output,
                                           self_attn_mask, enc_key_mask)
            next_token_logits = self.output_layer(decoder_output[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            next_token[finished] = self.pad_id  # freeze already-finished sequences
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return decoder_input_ids