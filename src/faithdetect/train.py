"""Seeded training loop (AdamW + linear warmup, early stopping) for all three variants."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data.collate import Collator
from .data.maide_up import TextLabelDataset
from .function_words import FunctionWordSet
from .models import ModelConfig, ReviewDetector, compute_loss
from .utils.seeding import set_seed, seed_worker


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 2          # early-stop patience on val macro-F1
    num_workers: int = 0
    log_every: int = 0         # 0 = silent batch logging


def make_collator(
    model_cfg: ModelConfig,
    tokenizer,
    func_id: int,
    fw_set: FunctionWordSet,
) -> Collator:
    """Collator consistent with the variant. baseline/softreg -> no masking; hardmask ->
    [FUNC] placeholder; deletion -> drop FW tokens; random -> random-token placeholder."""
    mode = {"hardmask": "hardmask", "deletion": "deletion", "random": "random"}.get(
        model_cfg.variant, "none"
    )
    return Collator(
        tokenizer=tokenizer,
        fw_set=fw_set,
        max_length=model_cfg.max_length,
        hard_mask=(model_cfg.variant == "hardmask"),
        placeholder_id=func_id,
        padding=True,
        mask_mode=mode,
    )


def _move(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def _val_macro_f1(model, loader, device) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = _move(batch, device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())
    return float(f1_score(trues, preds, average="macro"))


def train_model(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    splits,
    tokenizer,
    func_id: int,
    fw_set: FunctionWordSet,
    device,
    seed: int,
) -> tuple[ReviewDetector, dict]:
    """Train one model and return (best_model, history)."""
    set_seed(seed)
    model = ReviewDetector(model_cfg, vocab_size=len(tokenizer)).to(device)
    collator = make_collator(model_cfg, tokenizer, func_id, fw_set)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        TextLabelDataset.from_frame(splits.train),
        batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collator,
        num_workers=train_cfg.num_workers, worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        TextLabelDataset.from_frame(splits.val),
        batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collator,
        num_workers=train_cfg.num_workers,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": train_cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=train_cfg.lr)
    total_steps = max(1, len(train_loader) * train_cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(train_cfg.warmup_ratio * total_steps), total_steps
    )

    history = {"train_loss": [], "val_macro_f1": []}
    best_f1, best_state, patience_left = -1.0, None, train_cfg.patience

    for epoch in range(train_cfg.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = _move(batch, device)
            optimizer.zero_grad()
            loss, _ = compute_loss(model, batch, model_cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            if train_cfg.log_every and step % train_cfg.log_every == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        avg_loss = epoch_loss / max(1, n_batches)
        val_f1 = _val_macro_f1(model, val_loader, device)
        history["train_loss"].append(avg_loss)
        history["val_macro_f1"].append(val_f1)
        print(f"[seed {seed}|{model_cfg.variant}] epoch {epoch+1}/{train_cfg.epochs} "
              f"loss={avg_loss:.4f} val_macroF1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            patience_left = train_cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[seed {seed}|{model_cfg.variant}] early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_macro_f1"] = best_f1
    return model, history
