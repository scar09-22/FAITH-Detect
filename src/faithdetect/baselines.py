"""Classic, cheap baselines so the transformer's gains are contextualised (fixes flaw F12).

* tfidf_lr           : TF-IDF (1-2 grams) + Logistic Regression on the full text.
* tfidf_lr (content) : same but with function words removed from the vocabulary -> a
                       *content-only* linear baseline, the linear analogue of FAITH-Detect.

These also let us show that even a bag-of-words model leans on function words unless they are
removed, motivating the invariance idea.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .function_words import FunctionWordSet
from .evaluate import classification_metrics


def tfidf_lr(
    train_df,
    test_df,
    fw_set: FunctionWordSet | None = None,
    content_only: bool = False,
    ngram_max: int = 2,
    seed: int = 0,
) -> dict:
    """Fit TF-IDF + LogisticRegression; return metrics + p_ai on the test frame."""
    stop_words = sorted(fw_set.words) if (content_only and fw_set is not None) else None
    vec = TfidfVectorizer(
        ngram_range=(1, ngram_max), min_df=2, max_features=50_000,
        sublinear_tf=True, stop_words=stop_words,
    )
    Xtr = vec.fit_transform(train_df["text"].tolist())
    Xte = vec.transform(test_df["text"].tolist())
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(Xtr, train_df["label"].values)
    p_ai = clf.predict_proba(Xte)[:, 1]
    y_pred = (p_ai >= 0.5).astype(int)
    metrics = classification_metrics(test_df["label"].values, y_pred, p_ai)
    return {
        "metrics": metrics,
        "y_true": test_df["label"].values.tolist(),
        "y_pred": y_pred.tolist(),
        "p_ai": p_ai.tolist(),
        "name": "tfidf_lr_content" if content_only else "tfidf_lr",
    }
