"""Local, download-free robustness attacks on the test set.

These probe the central hypothesis: a function-word-invariant detector should be UNMOVED by
attacks that perturb function words, while an unconstrained baseline (which may lean on
function-word shortcuts) degrades.

Attacks
-------
* function_word_attack : delete / duplicate / swap function words only. A hard-masked model
  is provably invariant to function-word *identity*; deletion/duplication still change token
  positions, so this is a strong, targeted probe.
* synonym_attack       : replace a fraction of CONTENT words with a WordNet synonym
  (paraphrase-style; leaves function words intact).
* whitespace_attack    : insert extra spaces inside words (a light typo/obfuscation attack).
"""
from __future__ import annotations

import random
import re

import pandas as pd

from ..function_words import FunctionWordSet, _normalize_surface

_FW_SWAP_POOL = ["the", "a", "an", "this", "that", "of", "in", "on", "at", "and", "but"]


def function_word_attack(text: str, fw_set: FunctionWordSet, rate: float = 0.5, seed: int = 0) -> str:
    """Delete, duplicate, or swap a fraction `rate` of the function words in `text`."""
    rng = random.Random(seed)
    words = text.split()
    out: list[str] = []
    for w in words:
        is_fw = _normalize_surface(w) in fw_set.words
        if is_fw and rng.random() < rate:
            op = rng.choice(["delete", "duplicate", "swap"])
            if op == "delete":
                continue
            if op == "duplicate":
                out.extend([w, w])
                continue
            if op == "swap":
                out.append(rng.choice(_FW_SWAP_POOL))
                continue
        out.append(w)
    return " ".join(out) if out else text


def _wordnet_synonym(word: str, rng: random.Random) -> str | None:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and name.isalpha():
                syns.add(name)
    if not syns:
        return None
    return rng.choice(sorted(syns))


def synonym_attack(text: str, fw_set: FunctionWordSet, rate: float = 0.3, seed: int = 0) -> str:
    """Replace a fraction `rate` of content words with a WordNet synonym."""
    rng = random.Random(seed)
    words = text.split()
    out = []
    for w in words:
        core = re.sub(r"[^A-Za-z']", "", w)
        is_content = core and _normalize_surface(w) not in fw_set.words
        if is_content and rng.random() < rate:
            syn = _wordnet_synonym(core.lower(), rng)
            if syn:
                out.append(syn)
                continue
        out.append(w)
    return " ".join(out)


def whitespace_attack(text: str, fw_set: FunctionWordSet, rate: float = 0.2, seed: int = 0) -> str:
    """Insert a space into the middle of a fraction `rate` of long words."""
    rng = random.Random(seed)
    words = text.split()
    out = []
    for w in words:
        if len(w) > 5 and rng.random() < rate:
            k = rng.randint(2, len(w) - 2)
            out.append(w[:k] + " " + w[k:])
        else:
            out.append(w)
    return " ".join(out)


ATTACKS = {
    "function_word": function_word_attack,
    "synonym": synonym_attack,
    "whitespace": whitespace_attack,
}


def build_attack_set(
    df: pd.DataFrame,
    fw_set: FunctionWordSet,
    attacks=("function_word", "synonym", "whitespace"),
    rate: float = 0.5,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Return {attack_name: attacked copy of df} with labels preserved."""
    result = {}
    for name in attacks:
        fn = ATTACKS[name]
        attacked = df.copy()
        attacked["text"] = [
            fn(t, fw_set, rate, seed=seed + i) for i, t in enumerate(df["text"].tolist())
        ]
        result[name] = attacked.reset_index(drop=True)
    return result
