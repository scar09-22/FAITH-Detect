"""Automated checks for the two correctness claims the paper rests on:

1. Function-word identification maps the right subword tokens (incl. 'the','a','an').
2. The hard-masked model's decision is INVARIANT to function-word identity: perturbing only
   function words leaves the logits bit-identical.

Run:  python tests/test_invariance.py   (or via pytest)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from faithdetect.function_words import (
    build_function_word_set, function_word_token_mask, apply_hard_mask, add_func_token,
)
from faithdetect.models import ModelConfig, build_tokenizer, ReviewDetector
from faithdetect.explain import build_alignment
from faithdetect.utils.seeding import set_seed


def test_fw_set_contains_core_examples():
    fw = build_function_word_set("union")
    for w in ["the", "a", "an", "of", "and", "is", "was"]:
        assert w in fw.words, f"{w} should be a function word"


def test_subword_masking_marks_function_words():
    fw = build_function_word_set("union")
    tok, _ = build_tokenizer("roberta-base")
    text = "The hotel was clean and the staff helpful."
    enc, mask = function_word_token_mask([text], tok, fw, max_length=32)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    masked = {t.replace("Ġ", "").lower() for t, m in zip(toks, mask[0]) if m}
    assert {"the", "was", "and"} <= masked
    assert "hotel" not in masked and "staff" not in masked


def test_hard_mask_invariance():
    """Swapping only function words must not change the model output at all."""
    set_seed(0)
    fw = build_function_word_set("union")
    tok, fid = build_tokenizer("roberta-base")
    cfg = ModelConfig(variant="hardmask", max_length=48)
    model = ReviewDetector(cfg, vocab_size=len(tok)).eval()

    def logits_for(text):
        a = build_alignment(text, tok, fw, cfg.max_length, hard_mask=True, placeholder_id=fid)
        ids = torch.as_tensor(a.input_ids[None, :], dtype=torch.long)
        am = torch.as_tensor(a.attention_mask[None, :], dtype=torch.long)
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=am).numpy()

    base = "The room was clean and the staff were nice"
    swapped = "A room was clean but a staff were nice"  # only function words changed
    assert np.allclose(logits_for(base), logits_for(swapped), atol=1e-5), "hard-mask not invariant!"


def test_apply_hard_mask_replaces_only_fw():
    fw = build_function_word_set("union")
    tok, fid = build_tokenizer("roberta-base")
    enc, mask = function_word_token_mask(["The clean room"], tok, fw, max_length=16)
    masked_ids = apply_hard_mask(np.array(enc["input_ids"]), mask, fid)
    assert (masked_ids[mask] == fid).all()
    assert (masked_ids[~mask] == np.array(enc["input_ids"])[~mask]).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All invariance tests passed.")
