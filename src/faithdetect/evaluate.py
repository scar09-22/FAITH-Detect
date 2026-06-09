"""Prediction + metric computation (with calibration). Significance/CI live in utils.stats."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from torch.utils.data import DataLoader

from .data.collate import Collator
from .data.maide_up import TextLabelDataset
from .models import ModelConfig, ReviewDetector
from .train import make_collator
from .utils.stats import expected_calibration_error


@torch.no_grad()
def predict(
    model: ReviewDetector,
    df,
    collator: Collator,
    device,
    batch_size: int = 32,
) -> dict:
    """Run inference over a frame. Returns y_true, y_pred, p_ai (prob of AI), indices."""
    model.eval()
    loader = DataLoader(
        TextLabelDataset.from_frame(df), batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    y_true, y_pred, p_ai = [], [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_pred.extend(logits.argmax(1).cpu().numpy())
        p_ai.extend(probs.cpu().numpy())
        y_true.extend(batch["labels"].cpu().numpy())
    return {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "p_ai": np.array(p_ai, dtype=float),
    }


@torch.no_grad()
def extract_embeddings(model, df, collator, device, batch_size: int = 32) -> np.ndarray:
    """Pooled encoder representations for each row (for UMAP/t-SNE figures)."""
    model.eval()
    loader = DataLoader(
        TextLabelDataset.from_frame(df), batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    embs = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        pooled = model._pool(out.last_hidden_state, batch["attention_mask"])
        embs.append(pooled.cpu().numpy())
    return np.concatenate(embs, axis=0)


def classification_metrics(y_true, y_pred, p_ai=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }
    if p_ai is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, p_ai))
        m["pr_auc"] = float(average_precision_score(y_true, p_ai))
        m["ece"] = float(expected_calibration_error(y_true, p_ai)["ece"])
    return m


def evaluate_split(
    model: ReviewDetector,
    df,
    model_cfg: ModelConfig,
    tokenizer,
    func_id: int,
    fw_set,
    device,
    batch_size: int = 32,
) -> dict:
    """Predict + metrics + raw arrays + confusion matrix for one evaluation frame."""
    collator = make_collator(model_cfg, tokenizer, func_id, fw_set)
    pred = predict(model, df, collator, device, batch_size)
    metrics = classification_metrics(pred["y_true"], pred["y_pred"], pred["p_ai"])
    metrics["confusion_matrix"] = confusion_matrix(pred["y_true"], pred["y_pred"]).tolist()
    return {"metrics": metrics, **{k: v.tolist() for k, v in pred.items()}}
