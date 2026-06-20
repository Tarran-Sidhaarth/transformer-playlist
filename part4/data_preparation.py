"""
Part 4 — Data preparation for the from-scratch Transformer summarizer.

This file ONLY handles: text -> token IDs -> framed encoder/decoder tensors + masks.
It does NOT build embeddings (that's your part 1/3 modules) and does NOT train anything.

Pipeline per example:
    article  -> [BOS] + tokens + [EOS] -> pad to MAX_SRC_LEN          -> encoder input
    abstract -> [BOS] + tokens (shifted right)                       -> decoder input
             -> tokens + [EOS] (shifted left, the thing we predict)  -> target labels
"""

import torch
from torch.utils.data import Dataset
import tiktoken

# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------
# r50k_base = the GPT-2/GPT-3 BPE tokenizer, 50,257 tokens. Chosen over the
# larger cl100k_base (100k vocab) to keep the embedding table and final
# lm_head projection (vocab_size x d_model) smaller, since that layer's size
# directly affects training speed and how much signal each rare token gets.
enc = tiktoken.get_encoding("r50k_base")

# tiktoken's base encodings ship with NO special tokens at all. We reserve
# three extra IDs right after the existing vocab for BOS / EOS / PAD.
# IMPORTANT: when you build your embedding layer and lm_head in part 1/3,
# their vocab dimension must be TOTAL_VOCAB_SIZE, not enc.n_vocab, or these
# IDs will be out of range.
BASE_VOCAB_SIZE = enc.n_vocab          # 50257
BOS_ID = BASE_VOCAB_SIZE               # 50257
EOS_ID = BASE_VOCAB_SIZE + 1           # 50258
PAD_ID = BASE_VOCAB_SIZE + 2           # 50259
TOTAL_VOCAB_SIZE = BASE_VOCAB_SIZE + 3  # 50260  <- use this in your model

# ---------------------------------------------------------------------------
# 2. Sequence length budgets
# ---------------------------------------------------------------------------
# PubMed articles are long full-text papers; abstracts are short. These caps
# control truncation. Raise MAX_SRC_LEN if your hardware/attention impl can
# handle it (cost grows quadratically with sequence length for self-attention).
MAX_SRC_LEN = 1024   # encoder (article) length
MAX_TGT_LEN = 256    # decoder (abstract) length


def encode_article(text: str, max_len: int = MAX_SRC_LEN):
    """
    Tokenize + frame an article for the ENCODER.

    Returns:
        ids:  list[int], length == max_len, BOS-wrapped, EOS-wrapped, PAD-filled
        mask: list[int], 1 for real tokens, 0 for padding
              (this is what self-attention uses to ignore PAD positions)
    """
    token_ids = enc.encode(text)
    token_ids = [BOS_ID] + token_ids[: max_len - 2] + [EOS_ID]  # -2 reserves room for BOS/EOS

    pad_len = max_len - len(token_ids)
    mask = [1] * len(token_ids) + [0] * pad_len
    token_ids = token_ids + [PAD_ID] * pad_len
    return token_ids, mask


def encode_abstract(text: str, max_len: int = MAX_TGT_LEN):
    """
    Tokenize + frame an abstract for the DECODER using teacher forcing.

    decoder_input = [BOS, t1, t2, ..., t_{n-1}]   <- what the decoder SEES
    target        = [t1, t2, ..., t_{n-1}, EOS]   <- what the decoder must PREDICT

    At position i, decoder_input[i] is the token the model has access to,
    and target[i] is the next correct token. This is the "shift" you
    mentioned: input is the sequence shifted right (BOS prepended), target
    is the same sequence shifted left relative to it (EOS appended), so
    they're offset by exactly one position.

    Returns:
        decoder_input: list[int], length == max_len
        target:        list[int], length == max_len
        mask:          list[int], 1 for real tokens, 0 for padding
    """
    token_ids = enc.encode(text)
    token_ids = token_ids[: max_len - 1]  # leave room for the BOS/EOS that get added below

    decoder_input = [BOS_ID] + token_ids
    target = token_ids + [EOS_ID]

    pad_len = max_len - len(decoder_input)
    mask = [1] * len(decoder_input) + [0] * pad_len
    decoder_input = decoder_input + [PAD_ID] * pad_len
    # Target is padded with PAD_ID too. In part 5, set your loss function's
    # ignore_index=PAD_ID so these positions contribute zero gradient.
    target = target + [PAD_ID] * pad_len

    return decoder_input, target, mask


def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Lower-triangular boolean mask, shape (seq_len, seq_len).
    mask[i, j] = True  if position i is allowed to attend to position j (j <= i)
    mask[i, j] = False if j is in the future relative to i (blocked)

    Used inside decoder self-attention, combined with the decoder padding
    mask (AND'd together) so the model can neither see future tokens nor
    attend to padding.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


# ---------------------------------------------------------------------------
# 3. Dataset wrapper
# ---------------------------------------------------------------------------
class PubMedSummarizationDataset(Dataset):
    """
    Wraps one HF dataset split (e.g. dataset["train"]) and returns fully
    framed tensors ready for the embedding layer -> encoder/decoder.
    """

    def __init__(self, hf_split, max_src_len: int = MAX_SRC_LEN, max_tgt_len: int = MAX_TGT_LEN):
        self.data = hf_split
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        src_ids, src_mask = encode_article(example["article"], self.max_src_len)
        dec_in_ids, tgt_ids, tgt_mask = encode_abstract(example["abstract"], self.max_tgt_len)

        return {
            "encoder_input_ids": torch.tensor(src_ids, dtype=torch.long),
            "encoder_padding_mask": torch.tensor(src_mask, dtype=torch.bool),
            "decoder_input_ids": torch.tensor(dec_in_ids, dtype=torch.long),
            "decoder_padding_mask": torch.tensor(tgt_mask, dtype=torch.bool),
            "target_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 4. Example usage (uncomment once you load the real dataset)
# ---------------------------------------------------------------------------
# from datasets import load_dataset
# from torch.utils.data import DataLoader
#
# raw = load_dataset("ccdv/pubmed-summarization")
# train_dataset = PubMedSummarizationDataset(raw["train"])
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#
# batch = next(iter(train_loader))
# print(batch["encoder_input_ids"].shape)   # (8, MAX_SRC_LEN)
# print(batch["decoder_input_ids"].shape)   # (8, MAX_TGT_LEN)


if __name__ == "__main__":
    # Quick sanity check with dummy text (no real dataset needed)
    fake_example = {
        "article": "Background: lung cancer remains a leading cause of death. " * 5,
        "abstract": "We studied lung cancer outcomes in a large cohort.",
    }

    src_ids, src_mask = encode_article(fake_example["article"], max_len=32)
    dec_in, tgt, tgt_mask = encode_abstract(fake_example["abstract"], max_len=16)

    print("encoder ids: ", src_ids)
    print("encoder mask:", src_mask)
    print("decoder input:", dec_in)
    print("target:       ", tgt)
    print("decoder mask: ", tgt_mask)
    print("vocab size for your embedding/lm_head layers:", TOTAL_VOCAB_SIZE)

    # decoder_input and target should be offset by exactly one real token
    assert dec_in[1] == tgt[0], "decoder_input[1] should equal target[0] (shift check)"
    print("\nShift check passed: decoder_input[1] == target[0]")