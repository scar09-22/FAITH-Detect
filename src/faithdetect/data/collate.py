"""Batch collation with function-word masking.

A single Collator serves every model variant via ``mask_mode``:
  * baseline / softreg  -> mask_mode='none'  (fw_mask still returned; softreg consumes it).
  * hardmask            -> mask_mode='hardmask': FW token ids replaced by a single shared
                           placeholder ([FUNC]) -> identity erased, slot position+count kept.
  * deletion            -> mask_mode='deletion': FW tokens removed entirely -> identity AND
                           slot removed (sequence shortens). "Truly removes function words."
  * random              -> mask_mode='random': each FW token replaced by a content-hash-seeded
                           random vocabulary id -> identity erased and the consistent [FUNC]
                           slot-marker removed, while the perturbed positions/count are kept.
The last two are controls that separate identity, slot-marker and position/syntax effects.
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
        mask_mode: str = "auto",
    ):
        self.tokenizer = tokenizer
        self.fw_set = fw_set
        self.max_length = max_length
        self.hard_mask = hard_mask
        self.placeholder_id = placeholder_id
        self.padding = padding
        # back-compat: 'auto' derives the mode from the hard_mask flag.
        self.mask_mode = ("hardmask" if hard_mask else "none") if mask_mode == "auto" else mask_mode
        if self.mask_mode in ("hardmask",) and placeholder_id is None:
            raise ValueError("hardmask requires a placeholder_id")
        self._pad_id = tokenizer.pad_token_id
        self._special = set(tokenizer.all_special_ids)
        self._vocab = len(tokenizer)

    def _random_replace(self, input_ids: np.ndarray, fw_mask: np.ndarray) -> np.ndarray:
        out = input_ids.copy()
        for r in range(out.shape[0]):
            cols = np.where(fw_mask[r])[0]
            if cols.size == 0:
                continue
            rng = np.random.RandomState(abs(hash(tuple(int(x) for x in input_ids[r]))) % (2**32))
            for c in cols:
                tid = int(rng.randint(0, self._vocab))
                while tid in self._special or tid == self.placeholder_id:
                    tid = int(rng.randint(0, self._vocab))
                out[r, c] = tid
        return out

    def _delete(self, input_ids: np.ndarray, attn: np.ndarray, fw_mask: np.ndarray):
        rows_ids, rows_len = [], []
        for r in range(input_ids.shape[0]):
            keep = (attn[r] == 1) & (~fw_mask[r])
            kept = input_ids[r][keep]
            rows_ids.append(kept)
            rows_len.append(len(kept))
        width = max(rows_len) if rows_len else 1
        new_ids = np.full((input_ids.shape[0], width), self._pad_id, dtype=input_ids.dtype)
        new_attn = np.zeros((input_ids.shape[0], width), dtype=attn.dtype)
        for r, kept in enumerate(rows_ids):
            new_ids[r, : len(kept)] = kept
            new_attn[r, : len(kept)] = 1
        return new_ids, new_attn

    def __call__(self, batch: list[dict]) -> dict:
        texts = [b["text"] for b in batch]
        labels = [b["label"] for b in batch]
        indices = [b.get("index", -1) for b in batch]
        enc, fw_mask = function_word_token_mask(
            texts, self.tokenizer, self.fw_set, self.max_length, padding=self.padding
        )
        input_ids = np.array(enc["input_ids"])
        attn = np.array(enc["attention_mask"])
        fw_mask = np.array(fw_mask)

        if self.mask_mode == "hardmask":
            input_ids = apply_hard_mask(input_ids, fw_mask, self.placeholder_id)
        elif self.mask_mode == "random":
            input_ids = self._random_replace(input_ids, fw_mask)
        elif self.mask_mode == "deletion":
            input_ids, attn = self._delete(input_ids, attn, fw_mask)
            fw_mask = np.zeros_like(input_ids, dtype=bool)

        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(attn, dtype=torch.long),
            "fw_mask": torch.as_tensor(fw_mask, dtype=torch.bool),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "index": torch.as_tensor(indices, dtype=torch.long),
        }
