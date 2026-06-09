"""Model variants for FAITH-Detect.

All three variants share ONE architecture (a transformer encoder + linear head over the
[CLS]/<s> representation); they differ only in *how data is fed* and *what loss is used*:

  * ``baseline`` : full text, cross-entropy.   -> shows shortcut learning.
  * ``hardmask`` : function-word token ids replaced by the ``[FUNC]`` placeholder (done in
                   the Collator), cross-entropy. The decision is PROVABLY invariant to
                   function-word identity (the encoder never sees which function word it was).
  * ``softreg``  : full text, cross-entropy + lambda * (saliency mass on function-word
                   tokens). A "right-for-the-right-reasons" input-gradient penalty that
                   discourages, but does not forbid, reliance on function words.

A single clean architecture also fixes flaw F11 (architecture/feature-dim drift across the
old scripts). The ``[FUNC]`` token is added for every variant (unused by baseline/softreg)
so all checkpoints share one vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .function_words import add_func_token

VARIANTS = ("baseline", "hardmask", "softreg")


@dataclass
class ModelConfig:
    encoder_name: str = "roberta-base"
    variant: str = "baseline"          # one of VARIANTS
    num_labels: int = 2
    dropout: float = 0.1
    max_length: int = 256
    softreg_lambda: float = 1.0        # weight of the attribution penalty (softreg only)
    pooling: str = "cls"               # "cls" or "mean"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")


def build_tokenizer(encoder_name: str = "roberta-base"):
    """Return (tokenizer, func_token_id) with the ``[FUNC]`` placeholder registered."""
    tok = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    func_id = add_func_token(tok)
    return tok, func_id


class ReviewDetector(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int | None = None):
        super().__init__()
        self.config = config
        # Force eager attention: the fused SDPA kernel lacks a CPU double-backward (needed by
        # the soft-reg attribution penalty) and bus-errors under Captum IG on CPU. Eager
        # attention is correct on CPU/MPS/CUDA and only marginally slower for these sizes.
        try:
            self.encoder = AutoModel.from_pretrained(config.encoder_name, attn_implementation="eager")
        except (TypeError, ValueError):
            self.encoder = AutoModel.from_pretrained(config.encoder_name)
        if vocab_size is not None and vocab_size != self.encoder.get_input_embeddings().weight.shape[0]:
            self._resize_embeddings(vocab_size)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden, config.num_labels)

    def _resize_embeddings(self, vocab_size: int) -> None:
        old = self.encoder.get_input_embeddings().weight.data
        self.encoder.resize_token_embeddings(vocab_size)
        # Initialise any new rows (e.g. [FUNC]) to the mean of existing embeddings for stability.
        new = self.encoder.get_input_embeddings().weight.data
        if vocab_size > old.shape[0]:
            new[old.shape[0]:] = old.mean(dim=0, keepdim=True)

    def word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_input_embeddings()(input_ids)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp_min(1e-9)
        return summed / counts

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None) -> torch.Tensor:
        kwargs = {"attention_mask": attention_mask}
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        else:
            kwargs["input_ids"] = input_ids
        out = self.encoder(**kwargs)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))


def saliency_on_function_words(
    model: ReviewDetector,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    fw_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, penalty) for the soft-reg variant.

    penalty = mean over batch of (saliency mass on function-word tokens / total saliency),
    where per-token saliency = |gradient(true-class logit) . input embedding| (gradient x
    input). ``create_graph=True`` lets the penalty be backpropagated into model weights
    (Ross et al., 2017, "Right for the Right Reasons").
    """
    emb = model.word_embeddings(input_ids).detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=emb, attention_mask=attention_mask)
    target = logits.gather(1, labels.view(-1, 1)).sum()
    grads = torch.autograd.grad(target, emb, create_graph=True)[0]  # [B, L, H]
    token_saliency = (grads * emb).sum(-1).abs()                    # [B, L]
    real = attention_mask.float()
    fw = fw_mask.float() * real
    num = (token_saliency * fw).sum(1)
    den = (token_saliency * real).sum(1).clamp_min(1e-8)
    penalty = (num / den).mean()
    return logits, penalty


def compute_loss(
    model: ReviewDetector,
    batch: dict,
    config: ModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, logits) for the configured variant."""
    labels = batch["labels"]
    if config.variant == "softreg":
        logits, penalty = saliency_on_function_words(
            model, batch["input_ids"], batch["attention_mask"], batch["fw_mask"], labels
        )
        loss = F.cross_entropy(logits, labels) + config.softreg_lambda * penalty
    else:
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = F.cross_entropy(logits, labels)
    return loss, logits
