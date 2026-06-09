"""Batch collation with function-word masking.

A single Collator serves all three model variants:
  * baseline    -> hard_mask=False; fw_mask still returned (unused by the model).
  * hard-mask   -> hard_mask=True;  function-word token ids replaced by the placeholder.
  * soft-reg    -> hard_mask=False; the model/training loop consumes fw_mask for the
                   attribution-regularisation penalty.
"""
from __future__ import annotations

import numpy as np
import torch

from ..function_words import FunctionWordSet, function_word_token_mask, apply_hard_mask


class Collator:
    def __init__(
        self,
        tokenizer,
        fw_set: FunctionWordSet,
        max_length: int = 256,
        hard_mask: bool = False,
        placeholder_id: int | None = None,
        padding: str | bool = True,
    ):
        self.tokenizer = tokenizer
        self.fw_set = fw_set
        self.max_length = max_length
        self.hard_mask = hard_mask
        self.placeholder_id = placeholder_id
        self.padding = padding
        if hard_mask and placeholder_id is None:
            raise ValueError("hard_mask=True requires a placeholder_id")

    def __call__(self, batch: list[dict]) -> dict:
        texts = [b["text"] for b in batch]
        labels = [b["label"] for b in batch]
        indices = [b.get("index", -1) for b in batch]
        enc, fw_mask = function_word_token_mask(
            texts, self.tokenizer, self.fw_set, self.max_length, padding=self.padding
        )
        input_ids = np.array(enc["input_ids"])
        if self.hard_mask:
            input_ids = apply_hard_mask(input_ids, fw_mask, self.placeholder_id)
        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(np.array(enc["attention_mask"]), dtype=torch.long),
            "fw_mask": torch.as_tensor(fw_mask, dtype=torch.bool),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "index": torch.as_tensor(indices, dtype=torch.long),
        }
