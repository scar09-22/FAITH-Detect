"""Faithful, content-focused attributions over the REAL model.

This fixes the central integrity flaws of the original project:
  * F1: the old LIME/SHAP fed ``dummy_features = zeros`` and so explained a DIFFERENT model
    than the one evaluated. Here every attribution runs the actual deployed model, including
    the variant's input transform (hard masking for the ``hardmask`` variant).
  * F4: a home-grown "LIME" is replaced by Captum Integrated Gradients and genuine
    leave-one-word-out occlusion.

Display contract (user requirement): function words are EXCLUDED from the returned
explanation. For the ``hardmask`` variant the decision is additionally invariant to
function-word identity by construction, and we measure the residual attribution mass on
function-word positions (see faithfulness.function_word_attribution_mass).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..function_words import FunctionWordSet, _normalize_surface, apply_hard_mask


@dataclass
class Alignment:
    text: str
    input_ids: np.ndarray            # [L]
    attention_mask: np.ndarray       # [L]
    fw_token_mask: np.ndarray        # [L] bool: token is part of a function word
    token_word_id: list              # [L] int word index or None (special token)
    words: list                      # surface words in order
    word_is_fw: list                 # bool per word
    word_token_indices: list         # list[list[int]]: token indices per word


def build_alignment(
    text: str,
    tokenizer,
    fw_set: FunctionWordSet,
    max_length: int = 256,
    hard_mask: bool = False,
    placeholder_id: int | None = None,
) -> Alignment:
    enc = tokenizer(
        text, truncation=True, max_length=max_length,
        return_offsets_mapping=True, return_tensors="np",
    )
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]
    offsets = enc["offset_mapping"][0]
    word_ids = enc.word_ids(batch_index=0)
    L = len(input_ids)

    groups: dict[int, list[int]] = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        groups.setdefault(wid, []).append(i)

    fw_token_mask = np.zeros(L, dtype=bool)
    words, word_is_fw, word_token_indices = [], [], []
    for wid in sorted(groups):
        toks = groups[wid]
        starts = [int(offsets[i][0]) for i in toks]
        ends = [int(offsets[i][1]) for i in toks]
        surface = text[min(starts): max(ends)]
        is_fw = _normalize_surface(surface) in fw_set.words
        words.append(surface)
        word_is_fw.append(is_fw)
        word_token_indices.append(toks)
        if is_fw:
            for i in toks:
                fw_token_mask[i] = True

    if hard_mask:
        if placeholder_id is None:
            raise ValueError("hard_mask=True requires placeholder_id")
        input_ids = apply_hard_mask(input_ids, fw_token_mask, placeholder_id)

    return Alignment(
        text=text, input_ids=input_ids, attention_mask=attention_mask,
        fw_token_mask=fw_token_mask, token_word_id=word_ids,
        words=words, word_is_fw=word_is_fw, word_token_indices=word_token_indices,
    )


@dataclass
class WordAttribution:
    text: str
    words: list                       # all surface words
    word_is_fw: list                  # function-word flags
    word_scores: list                 # attribution per word (AI class), aligned to words
    method: str
    predicted_label: int
    p_ai: float
    token_scores: np.ndarray = field(default=None, repr=False)  # [L] token attributions
    fw_token_mask: np.ndarray = field(default=None, repr=False)
    attention_mask: np.ndarray = field(default=None, repr=False)

    def content_explanation(self, top_k: int | None = None) -> list[tuple[str, float]]:
        """Function words (and punctuation-only tokens) EXCLUDED. Returns (word, score)
        sorted by |score| desc."""
        pairs = [
            (w, s)
            for w, fw, s in zip(self.words, self.word_is_fw, self.word_scores)
            if (not fw) and any(c.isalnum() for c in w)
        ]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:top_k] if top_k else pairs


class FaithfulExplainer:
    """Explains the real model. The variant transform is applied consistently."""

    def __init__(self, model, model_cfg, tokenizer, func_id, fw_set, device):
        self.model = model
        self.cfg = model_cfg
        self.tokenizer = tokenizer
        self.func_id = func_id
        self.fw_set = fw_set
        self.device = device
        self.hard_mask = model_cfg.variant == "hardmask"

    def _align(self, text: str) -> Alignment:
        return build_alignment(
            text, self.tokenizer, self.fw_set, self.cfg.max_length,
            hard_mask=self.hard_mask, placeholder_id=self.func_id,
        )

    @torch.no_grad()
    def proba(self, texts) -> np.ndarray:
        """p(AI) for each text, using the deployed variant's masking. Used by occlusion,
        faithfulness, LIME and SHAP — all on the real model."""
        if isinstance(texts, str):
            texts = [texts]
        self.model.eval()
        out = []
        for t in texts:
            a = self._align(t)
            ids = torch.as_tensor(a.input_ids[None, :], dtype=torch.long, device=self.device)
            am = torch.as_tensor(a.attention_mask[None, :], dtype=torch.long, device=self.device)
            logits = self.model(input_ids=ids, attention_mask=am)
            out.append(float(torch.softmax(logits, 1)[0, 1].item()))
        return np.array(out)

    def _aggregate_to_words(self, a: Alignment, token_scores: np.ndarray) -> list:
        scores = []
        for toks in a.word_token_indices:
            scores.append(float(np.sum([token_scores[i] for i in toks])))
        return scores

    def integrated_gradients(
        self, text: str, n_steps: int = 32, target: int = 1, internal_batch_size: int = 4
    ) -> WordAttribution:
        """Integrated Gradients on the word-embedding layer of the real model.

        Implemented manually (Riemann sum over single backward passes through eager
        attention) rather than via Captum's batched-forward path, which can SIGBUS on CPU/
        macOS for longer sequences. This is portable across CPU/MPS/CUDA and uses exactly the
        same op as training (which is robust). `internal_batch_size` is accepted for API
        compatibility but unused here. Baseline = pad everywhere except special tokens.
        """
        a = self._align(text)
        ids = torch.as_tensor(a.input_ids[None, :], dtype=torch.long, device=self.device)
        am = torch.as_tensor(a.attention_mask[None, :], dtype=torch.long, device=self.device)

        special = set(self.tokenizer.all_special_ids) - {self.tokenizer.pad_token_id}
        baseline = np.full_like(a.input_ids, self.tokenizer.pad_token_id)
        for i, tid in enumerate(a.input_ids):
            if int(tid) in special:
                baseline[i] = tid
        base_ids = torch.as_tensor(baseline[None, :], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            p_ai = float(torch.softmax(self.model(input_ids=ids, attention_mask=am), 1)[0, 1])
            emb_input = self.model.word_embeddings(ids).detach()      # [1, L, H]
            emb_base = self.model.word_embeddings(base_ids).detach()
        pred = int(p_ai >= 0.5)

        total_grad = torch.zeros_like(emb_input)
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            emb = (emb_base + alpha * (emb_input - emb_base)).detach().requires_grad_(True)
            logits = self.model(inputs_embeds=emb, attention_mask=am)
            grad = torch.autograd.grad(logits[0, target], emb)[0]    # single backward (robust)
            total_grad = total_grad + grad.detach()
        avg_grad = total_grad / n_steps
        ig = ((emb_input - emb_base) * avg_grad).sum(-1).squeeze(0)   # [L]
        token_scores = ig.detach().cpu().numpy()
        word_scores = self._aggregate_to_words(a, token_scores)
        return WordAttribution(
            text=text, words=a.words, word_is_fw=a.word_is_fw, word_scores=word_scores,
            method="integrated_gradients", predicted_label=pred, p_ai=p_ai,
            token_scores=token_scores, fw_token_mask=a.fw_token_mask,
            attention_mask=a.attention_mask,
        )

    def occlusion(self, text: str) -> WordAttribution:
        """Leave-one-word-out occlusion on the real model (word-level, content + FW)."""
        a = self._align(text)
        words = a.words
        p_full = float(self.proba(text)[0])
        pred = int(p_full >= 0.5)
        scores = []
        for j in range(len(words)):
            reduced = " ".join(words[:j] + words[j + 1:])
            if not reduced.strip():
                scores.append(0.0)
                continue
            p_wo = float(self.proba(reduced)[0])
            scores.append(p_full - p_wo)  # positive -> word pushes prediction toward AI
        return WordAttribution(
            text=text, words=words, word_is_fw=a.word_is_fw, word_scores=scores,
            method="occlusion", predicted_label=pred, p_ai=p_full,
            fw_token_mask=a.fw_token_mask, attention_mask=a.attention_mask,
        )
