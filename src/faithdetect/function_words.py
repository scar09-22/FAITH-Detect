"""Function-word identification and masking — the core of FAITH-Detect's novelty.

The user requirement: function words ("the", "a", "an", ...) — high-frequency tokens common
to BOTH human and AI text — must be (1) excluded from the XAI explanation and (2) unable to
affect the model's final decision.

This module provides:
  * Construction of an English function-word (closed-class) set from several standard
    sources (NLTK / scikit-learn / spaCy stopword lists + a curated closed-class list,
    and optionally a part-of-speech definition). The chosen set is a documented
    methodological artifact (see SRS section 3).
  * Robust mapping from RoBERTa byte-level BPE subword tokens back to surface words, so we
    can mark exactly which subword tokens belong to function words.
  * `apply_hard_mask`, which replaces every function-word subword token id with a single
    shared placeholder id. Because the model then sees the *same* placeholder regardless of
    which function word was present, its decision is PROVABLY invariant to function-word
    identity (verified by tests/invariance_check).

Design choices
--------------
* Identification is by *surface form* (lower-cased, punctuation-stripped word) against a
  fixed set. This is fully deterministic at inference time (no POS tagger in the hot path)
  and gives the hard invariance guarantee its clean, defensible form. A POS-based set is
  also offered for the ablation that compares set definitions.
* The placeholder is a dedicated added special token ``[FUNC]`` (not the pretrained
  ``<mask>``), so "a function word occurred here" is signalled without leaking which one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

FUNC_TOKEN = "[FUNC]"

# Universal-Dependencies closed-class POS tags (used by the POS-based set definition and to
# document what "function word" means linguistically).
CLOSED_CLASS_POS = ("DET", "ADP", "CCONJ", "SCONJ", "AUX", "PRON", "PART")

# A curated, explicit closed-class English list. Kept separate so the set is auditable and
# does not depend on any library being installed. Includes the canonical examples the
# requirement calls out ("the", "a", "an") and the rest of the closed classes.
CURATED_CLOSED_CLASS: frozenset[str] = frozenset(
    {
        # articles / determiners
        "a", "an", "the", "this", "that", "these", "those", "such", "every", "each",
        "either", "neither", "another", "any", "some", "no", "all", "both", "half",
        # personal / possessive / reflexive / relative / interrogative pronouns
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "who", "whom", "whose", "which", "what", "whatever",
        "whoever", "whomever", "whichever", "one", "ones", "oneself",
        # prepositions
        "of", "in", "on", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to", "from",
        "up", "down", "over", "under", "again", "further", "then", "once", "out", "off",
        "onto", "upon", "within", "without", "along", "across", "behind", "beyond",
        "near", "around", "among", "amongst", "toward", "towards", "via", "per",
        # coordinating / subordinating conjunctions
        "and", "but", "or", "nor", "so", "yet", "because", "as", "until", "while",
        "although", "though", "since", "unless", "whereas", "whether", "if", "than",
        "albeit", "lest", "whilst",
        # auxiliaries / modals / copula
        "be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "will", "would", "shall", "should",
        "can", "could", "may", "might", "must", "ought", "need", "dare",
        # particles / negation / common adverbial function words
        "not", "no", "nor", "too", "very", "just", "only", "also", "even", "still",
        "there", "here", "when", "where", "why", "how", "all", "more", "most", "other",
        "s", "t", "re", "ve", "ll", "d", "m", "o",  # contraction remnants
        "n't", "'s", "'re", "'ve", "'ll", "'d", "'m",
    }
)


@dataclass(frozen=True)
class FunctionWordSet:
    """An immutable, named set of function-word surface forms."""

    name: str
    words: frozenset[str]
    sources: tuple[str, ...]

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def __len__(self) -> int:
        return len(self.words)


def _nltk_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return set()


def _sklearn_stopwords() -> set[str]:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return set(ENGLISH_STOP_WORDS)
    except Exception:
        return set()


def _spacy_stopwords() -> set[str]:
    try:
        from spacy.lang.en.stop_words import STOP_WORDS

        return set(STOP_WORDS)
    except Exception:
        return set()


def build_function_word_set(definition: str = "union") -> FunctionWordSet:
    """Build a named function-word set.

    definition:
      * ``"curated"``      -> only the curated closed-class list (most conservative).
      * ``"stopwords"``    -> union of NLTK + scikit-learn + spaCy English stop-word lists.
      * ``"union"``        -> curated + all stop-word lists (default; most comprehensive).
      * ``"pos"``          -> curated list treated as the POS proxy (closed-class surface
                              forms); used by the set-definition ablation.
    """
    sources: list[str] = []
    words: set[str] = set()
    if definition in ("curated", "union", "pos"):
        words |= set(CURATED_CLOSED_CLASS)
        sources.append("curated_closed_class")
    if definition in ("stopwords", "union"):
        for fn, name in [
            (_nltk_stopwords, "nltk"),
            (_sklearn_stopwords, "sklearn"),
            (_spacy_stopwords, "spacy"),
        ]:
            s = fn()
            if s:
                words |= s
                sources.append(name)
    words = {w.lower().strip() for w in words if w.strip()}
    return FunctionWordSet(name=definition, words=frozenset(words), sources=tuple(sources))


@lru_cache(maxsize=8)
def default_function_word_set(definition: str = "union") -> FunctionWordSet:
    return build_function_word_set(definition)


_WORD_STRIP = re.compile(r"^[^\w']+|[^\w']+$")


def _normalize_surface(s: str) -> str:
    """Lower-case and strip leading/trailing punctuation (keep internal apostrophes)."""
    return _WORD_STRIP.sub("", s.strip().lower())


def add_func_token(tokenizer) -> int:
    """Ensure the ``[FUNC]`` placeholder special token exists; return its id.

    Caller is responsible for ``model.resize_token_embeddings(len(tokenizer))`` afterwards.
    """
    if FUNC_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [FUNC_TOKEN]})
    return tokenizer.convert_tokens_to_ids(FUNC_TOKEN)


def function_word_token_mask(
    texts: Sequence[str],
    tokenizer,
    fw_set: FunctionWordSet,
    max_length: int = 256,
    padding: str | bool = "max_length",
    return_encoding: bool = True,
):
    """Tokenize `texts` and mark which subword tokens belong to function words.

    Parameters
    ----------
    padding:
        Passed to the tokenizer. ``"max_length"`` pads every example to `max_length`
        (handy for caching); ``True`` pads to the longest item in the batch (faster for
        training collation).

    Returns
    -------
    encoding : transformers.BatchEncoding (if return_encoding)
        Standard padded encoding (input_ids, attention_mask).
    fw_mask : np.ndarray [B, L] of bool
        True where the subword token is part of a function word. Special/pad tokens are
        always False.

    Implementation: uses the fast tokenizer's offset mapping + `word_ids()` to group
    subword tokens into surface words, reconstructs each word from the original string, and
    tests its normalized form against `fw_set`.
    """
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding=padding,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    input_ids = enc["input_ids"]
    B, L = input_ids.shape
    fw_mask = np.zeros((B, L), dtype=bool)

    for b in range(B):
        text = texts[b]
        offsets = enc["offset_mapping"][b]
        word_ids = enc.word_ids(batch_index=b)
        # Group token indices by their source word id.
        groups: dict[int, list[int]] = {}
        for tok_idx, wid in enumerate(word_ids):
            if wid is None:
                continue  # special token
            groups.setdefault(wid, []).append(tok_idx)
        for wid, tok_indices in groups.items():
            starts = [int(offsets[i][0]) for i in tok_indices]
            ends = [int(offsets[i][1]) for i in tok_indices]
            surface = text[min(starts): max(ends)]
            if _normalize_surface(surface) in fw_set.words:
                for i in tok_indices:
                    fw_mask[b, i] = True
    enc.pop("offset_mapping", None)
    if return_encoding:
        return enc, fw_mask
    return fw_mask


def apply_hard_mask(
    input_ids: np.ndarray,
    fw_mask: np.ndarray,
    placeholder_id: int,
) -> np.ndarray:
    """Replace every function-word subword token id with the shared placeholder id.

    After this, the encoder sees `placeholder_id` wherever a function word stood, regardless
    of which function word it was -> the downstream decision cannot depend on function-word
    identity. Returns a new array; does not mutate the input.
    """
    out = np.array(input_ids, copy=True)
    out[fw_mask] = placeholder_id
    return out


def word_function_flags(text: str, fw_set: FunctionWordSet) -> list[tuple[str, bool]]:
    """Whitespace-tokenize `text` and flag each word as function (True) or content (False).

    Used by the XAI display layer to grey out / drop function words from explanations.
    """
    flags = []
    for w in text.split():
        flags.append((w, _normalize_surface(w) in fw_set.words))
    return flags
