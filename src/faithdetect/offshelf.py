"""Off-the-shelf AI-text detectors as zero-shot-transfer reference points.

Wraps publicly released sequence-classification detectors (e.g.
``Hello-SimpleAI/chatgpt-detector-roberta``, ``openai-community/roberta-base-openai-detector``)
behind a minimal "probability that the text is AI-generated" interface, so they can be
scored with the same metric stack as FAITH-Detect's own models.

These detectors were trained on OTHER text distributions (HC3 question answering for
Hello-SimpleAI; GPT-2 WebText generations for the OpenAI detector). We therefore apply
them with NO fine-tuning and report results in a zero-shot-transfer style. Released
detectors do not agree on label semantics (the AI class may be index 0 or 1, named
"ChatGPT", "Fake", "machine", ...), so the AI class index is resolved from each model's
own ``config.id2label`` rather than assumed.
"""
from __future__ import annotations

import numpy as np
import torch

# Label names that released detectors use for the machine-generated class. Exact matches
# are checked first; the substring pass catches composites like "machine-generated".
_AI_EXACT = frozenset({
    "ai", "gpt", "bot", "fake", "machine", "chatgpt", "generated", "synthetic", "llm",
})
_AI_SUBSTR = ("chatgpt", "machine", "fake", "generated", "synthetic", "gpt", "bot")


def resolve_ai_index(id2label: dict) -> int | None:
    """Locate the AI/machine class index in a model's ``id2label`` mapping.

    Matches normalised label names against known AI-class names (exact first, then
    substring). Returns None when no label looks like an AI class (e.g. the generic
    ``LABEL_0``/``LABEL_1``), in which case the caller should fall back with a warning.
    """
    norm = {int(i): str(lbl).strip().lower() for i, lbl in id2label.items()}
    for i, lbl in sorted(norm.items()):
        if lbl in _AI_EXACT:
            return i
    for i, lbl in sorted(norm.items()):
        if any(k in lbl for k in _AI_SUBSTR):
            return i
    return None


class OffShelfDetector:
    """Lazy wrapper around a HuggingFace sequence-classification AI-text detector.

    The model and tokenizer are loaded on first use (keeps import + construction cheap,
    e.g. when only listing detectors). Inference is batched and gradient-free.

    Parameters
    ----------
    model_name : HuggingFace hub id of an ``AutoModelForSequenceClassification`` detector.
    device     : torch device string ("cpu", "mps", "cuda").
    max_length : tokenizer truncation length (these detectors were trained at 512).
    """

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.model = None
        self.tokenizer = None
        self.ai_index: int | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        id2label = {int(i): str(lbl) for i, lbl in self.model.config.id2label.items()}
        idx = resolve_ai_index(id2label)
        if idx is None:
            idx = 1
            print(f"[offshelf] WARNING: could not identify the AI class in "
                  f"id2label={id2label} for {self.model_name}; falling back to index 1.")
        self.ai_index = idx
        print(f"[offshelf] {self.model_name}: id2label={id2label} -> "
              f"AI class index {idx} ({id2label.get(idx, '?')!r})")

    @torch.no_grad()
    def predict_proba_ai(self, texts, batch_size: int = 16) -> np.ndarray:
        """Return P(AI-generated) for each text as a float array of shape [len(texts)].

        Softmax over the detector's logits, taking the column of the resolved AI class.
        Texts are truncated to ``max_length`` tokens.
        """
        self._ensure_loaded()
        texts = list(texts)
        if not texts:
            return np.zeros(0, dtype=float)
        probs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = self.tokenizer(
                chunk, truncation=True, padding=True, max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            p_ai = torch.softmax(logits, dim=-1)[:, self.ai_index]
            probs.append(p_ai.cpu().numpy())
        return np.concatenate(probs).astype(float)
