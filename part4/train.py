"""
Part 4/5 — Training loop for the from-scratch Transformer summarizer.

Run as: python train.py

Sized for an RTX 5060 Ti (16GB) + 32GB RAM. If you have the 8GB variant:
halve BATCH_SIZE and double GRAD_ACCUM_STEPS (keeps the same effective
batch size, just splits it across more forward/backward passes).
"""

import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data_preparation import PubMedSummarizationDataset, PAD_ID, MAX_TGT_LEN
from model_architecture import Transformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_DIMENSION = 512
NUMBER_HEADS = 8
FFN_HIDDEN_DIMENSION = 2048
NUMBER_ENCODER_LAYERS = 3
NUMBER_DECODER_LAYERS = 3
DROPOUT = 0.1

BATCH_SIZE = 16

GRAD_ACCUM_STEPS = 2 # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 32
NUM_EPOCHS = 20
WARMUP_STEPS = 4000            # matches the paper's schedule
LABEL_SMOOTHING = 0.1
CLIP_GRAD_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "best_model.pt"
LOG_EVERY = 100


def noam_lr_lambda(step: int, d_model: int = EMBEDDING_DIMENSION, warmup_steps: int = WARMUP_STEPS) -> float:
    """LR schedule from the paper (sec 5.3): ramps up linearly for
    `warmup_steps`, then decays as step^-0.5. Used with optimizer lr=1.0
    so this function IS the actual learning rate at each step."""
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


def build_dataloaders():
    from datasets import load_from_disk

    raw = load_from_disk("/home/tarran/compsci/tuts/encoder-decoder/dataset")
    raw["train"] = raw["train"].select(range(1000)) #type:ignore
    raw["validation"] = raw["validation"].select(range(200)) #type:ignore
    raw["test"] = raw["test"].select(range(200)) #type:ignore

    train_ds = PubMedSummarizationDataset(raw["train"])
    val_ds = PubMedSummarizationDataset(raw["validation"])
    test_ds = PubMedSummarizationDataset(raw["test"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, scheduler, criterion, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        encoder_input_ids = batch["encoder_input_ids"].to(DEVICE, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(DEVICE, non_blocking=True)
        target_ids = batch["target_ids"].to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits = model(encoder_input_ids, decoder_input_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
            loss = loss / GRAD_ACCUM_STEPS

        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS

        if step % LOG_EVERY == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"epoch {epoch} step {step}/{len(loader)} "
                  f"loss {loss.item()*GRAD_ACCUM_STEPS:.4f} lr {current_lr:.2e}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        encoder_input_ids = batch["encoder_input_ids"].to(DEVICE, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(DEVICE, non_blocking=True)
        target_ids = batch["target_ids"].to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits = model(encoder_input_ids, decoder_input_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow early in training
    return avg_loss, perplexity


@torch.no_grad()
def test_with_rouge(model, loader, max_examples: int = 200):
    """Generates summaries and scores them with ROUGE-1/2/L F1 - the
    standard metric for summarization (measures n-gram / longest-common-
    subsequence overlap with the reference abstract). Capped at
    max_examples by default since greedy decoding without a KV-cache is
    slow; raise or remove the cap for a final full-test-set number."""
    from rouge_score import rouge_scorer
    from data_preparation import enc as tokenizer, BOS_ID, EOS_ID

    def decode_ids(ids):
        return tokenizer.decode([i for i in ids if i not in (BOS_ID, EOS_ID, PAD_ID)])

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    count = 0

    model.eval()
    for batch in loader:
        encoder_input_ids = batch["encoder_input_ids"].to(DEVICE)
        target_ids = batch["target_ids"]

        generated = model.generate(encoder_input_ids, max_len=MAX_TGT_LEN)

        for i in range(generated.shape[0]):
            if count >= max_examples:
                break
            pred_text = decode_ids(generated[i].tolist())
            ref_text = decode_ids(target_ids[i].tolist())
            scores = scorer.score(ref_text, pred_text)
            for k in totals:
                totals[k] += scores[k].fmeasure
            count += 1
        if count >= max_examples:
            break

    return {k: v / count for k, v in totals.items()}


def main():
    train_loader, val_loader, test_loader = build_dataloaders()

    model = Transformer(
        embedding_dimension=EMBEDDING_DIMENSION,
        number_heads=NUMBER_HEADS,
        ffn_hidden_dimension=FFN_HIDDEN_DIMENSION,
        number_encoder_layers=NUMBER_ENCODER_LAYERS,
        number_decoder_layers=NUMBER_DECODER_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    print("Total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=noam_lr_lambda)

    best_val_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, epoch) # add scheduler
        val_loss, val_ppl = validate(model, val_loader, criterion)
        print(f"epoch {epoch} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f} | time {time.time()-start:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")

    # Final test: reload best checkpoint, generate summaries, score with ROUGE
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    rouge_scores = test_with_rouge(model, test_loader, max_examples=200)
    print("Test ROUGE (200 examples):", rouge_scores)


if __name__ == "__main__":
    main()