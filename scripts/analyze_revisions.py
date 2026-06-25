#!/usr/bin/env python3
"""Reviewer-revision analyses on the reference-seed (seed 0) checkpoints.

Produces two JSONs from the REAL trained models (no retraining):
  1. attack_split.json       -- F1 under swap / delete / duplicate / mixed FW attacks + synonym,
                                 isolating the identity-only perturbation the proof covers.
  2. placeholder_mass.json    -- token-level Integrated-Gradients attribution mass on
                                 function-word positions ([FUNC] slots for hardmask), i.e. whether
                                 the hard-masked model relies on function-word SLOTS.

Run:
  python scripts/analyze_revisions.py --mode attacks
  python scripts/analyze_revisions.py --mode attr --n-attr 60 --ig-steps 24
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
import torch

sys.path.insert(0, "src")
from faithdetect.models import ReviewDetector, ModelConfig, build_tokenizer
from faithdetect.function_words import build_function_word_set
from faithdetect.data import load_maide_up_english, make_splits, build_attack_set
from faithdetect.evaluate import evaluate_split
from faithdetect.explain.attributions import FaithfulExplainer

DATA = "data/all_data.csv"
ENCODER = "distilroberta-base"
MAXLEN = 112
FWDEF = "union"
SEED = 0
CKPT = {"baseline": "results/cells/baseline_s0_model.pt",
        "hardmask": "results/cells/hardmask_s0_model.pt",
        "softreg":  "results/cells/softreg_s0_model.pt"}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(variant, tokenizer, device):
    mcfg = ModelConfig(encoder_name=ENCODER, variant=variant, max_length=MAXLEN)
    model = ReviewDetector(mcfg, vocab_size=len(tokenizer))
    model.load_state_dict(torch.load(CKPT[variant], map_location="cpu"))
    model.to(device).eval()
    return model, mcfg


def run_attacks(device):
    fw_set = build_function_word_set(FWDEF)
    tok, func_id = build_tokenizer(ENCODER)
    df = load_maide_up_english(DATA)
    splits = make_splits(df, mode="grouped", seed=SEED)
    test = splits.test
    attack_names = ("fw_swap", "fw_delete", "fw_duplicate", "function_word", "synonym")
    frames = build_attack_set(test, fw_set, attacks=attack_names, rate=0.5, seed=SEED)
    out = {"meta": {"seed": SEED, "encoder": ENCODER, "n_test": int(len(test)),
                    "rate": 0.5, "note": "reference-seed (seed 0) checkpoints; F1 macro %"},
           "variants": {}}
    for v in ("baseline", "softreg", "hardmask"):
        model, mcfg = load(v, tok, device)
        row = {}
        m = evaluate_split(model, test, mcfg, tok, func_id, fw_set, device)["metrics"]
        row["clean"] = m
        for a in attack_names:
            mm = evaluate_split(model, frames[a], mcfg, tok, func_id, fw_set, device)["metrics"]
            row[a] = mm
        out["variants"][v] = row
        del model
        print(f"[attacks] {v}: clean F1={row['clean']['f1_macro']*100:.1f} "
              f"swap={row['fw_swap']['f1_macro']*100:.1f} del={row['fw_delete']['f1_macro']*100:.1f} "
              f"dup={row['fw_duplicate']['f1_macro']*100:.1f} mixed={row['function_word']['f1_macro']*100:.1f} "
              f"syn={row['synonym']['f1_macro']*100:.1f}", flush=True)
    json.dump(out, open("results/attack_split.json", "w"), indent=2)
    print("wrote results/attack_split.json", flush=True)


def _ig_padbaseline_fw_mass(expl, text, n_steps):
    """Integrated Gradients with a PAD baseline at every non-BOS/EOS position, including the
    [FUNC] placeholder. The library default keeps [FUNC] (an added special token) in the
    baseline, which forces its IG contribution to exactly zero; using a pad baseline instead
    lets us actually measure attribution on the [FUNC] slots, comparably to how FW-mass is
    measured for the baseline/softreg variants. Returns (fw_position_mass, content_mass)."""
    import torch as T
    a = expl._align(text)
    ids = T.as_tensor(a.input_ids[None, :], dtype=T.long, device=expl.device)
    am = T.as_tensor(a.attention_mask[None, :], dtype=T.long, device=expl.device)
    keep = {expl.tokenizer.bos_token_id, expl.tokenizer.eos_token_id,
            expl.tokenizer.cls_token_id, expl.tokenizer.sep_token_id} - {None}
    baseline = np.full_like(a.input_ids, expl.tokenizer.pad_token_id)
    for i, tid in enumerate(a.input_ids):
        if int(tid) in keep:
            baseline[i] = tid
    base_ids = T.as_tensor(baseline[None, :], dtype=T.long, device=expl.device)
    expl.model.eval()
    with T.no_grad():
        emb_in = expl.model.word_embeddings(ids).detach()
        emb_bs = expl.model.word_embeddings(base_ids).detach()
    total_grad = T.zeros_like(emb_in)
    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        emb = (emb_bs + alpha * (emb_in - emb_bs)).detach().requires_grad_(True)
        logits = expl.model(inputs_embeds=emb, attention_mask=am)
        grad = T.autograd.grad(logits[0, 1], emb)[0]
        total_grad = total_grad + grad.detach()
    ig = ((emb_in - emb_bs) * (total_grad / n_steps)).sum(-1).squeeze(0)
    ts = np.abs(ig.detach().cpu().numpy())
    attended = np.asarray(a.attention_mask, dtype=bool)
    fw = np.asarray(a.fw_token_mask, dtype=bool)
    wid = np.array([-1 if x is None else x for x in a.token_word_id])
    is_special = (wid == -1)
    denom = ts[attended & ~is_special].sum()
    if denom <= 0:
        return None, None
    return ts[attended & fw].sum() / denom, ts[attended & ~is_special & ~fw].sum() / denom


def run_attr(device, n_attr, ig_steps):
    fw_set = build_function_word_set(FWDEF)
    tok, func_id = build_tokenizer(ENCODER)
    df = load_maide_up_english(DATA)
    splits = make_splits(df, mode="grouped", seed=SEED)
    test = splits.test.reset_index(drop=True)
    ai = test[test["label"] == 1].head(n_attr // 2)
    hu = test[test["label"] == 0].head(n_attr - n_attr // 2)
    sample = list(ai["text"]) + list(hu["text"])
    out = {"meta": {"seed": SEED, "encoder": ENCODER, "n_texts": len(sample), "ig_steps": ig_steps,
                    "note": "PILOT distilroberta-base seed 0. IG uses a pad baseline incl. [FUNC] so "
                            "placeholder-slot attribution is actually measured; occlusion is forward-only. "
                            "Mass = share of |attribution| on FW/[FUNC] positions."},
           "variants": {}}
    for v in ("baseline", "softreg", "hardmask"):
        model, mcfg = load(v, tok, device)
        expl = FaithfulExplainer(model, mcfg, tok, func_id, fw_set, device)
        ig_fw, occ_fw = [], []
        t0 = time.time()
        for text in sample:
            m_fw, _ = _ig_padbaseline_fw_mass(expl, text, ig_steps)
            if m_fw is not None:
                ig_fw.append(m_fw)
            wa = expl.occlusion(text)
            ws = np.abs(np.asarray(wa.word_scores, dtype=float))
            isfw = np.asarray(wa.word_is_fw, dtype=bool)
            tot = ws.sum()
            if tot > 0:
                occ_fw.append(ws[isfw].sum() / tot)
        ig_arr, occ_arr = np.array(ig_fw), np.array(occ_fw)
        out["variants"][v] = {
            "ig_fw_position_mass_mean": float(ig_arr.mean()),
            "ig_fw_position_mass_sd": float(ig_arr.std(ddof=1)) if len(ig_arr) > 1 else 0.0,
            "occlusion_fw_mass_mean": float(occ_arr.mean()),
            "occlusion_fw_mass_sd": float(occ_arr.std(ddof=1)) if len(occ_arr) > 1 else 0.0,
            "n": int(len(ig_arr)),
        }
        print(f"[attr] {v}: IG FW/[FUNC]-slot mass={ig_arr.mean()*100:.2f}%  "
              f"occlusion FW/[FUNC] mass={occ_arr.mean()*100:.2f}%  "
              f"(n={len(ig_arr)}, {time.time()-t0:.0f}s)", flush=True)
        del model, expl
    json.dump(out, open("results/placeholder_mass.json", "w"), indent=2)
    print("wrote results/placeholder_mass.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["attacks", "attr", "both"], default="both")
    ap.add_argument("--n-attr", type=int, default=60)
    ap.add_argument("--ig-steps", type=int, default=24)
    args = ap.parse_args()
    dev = get_device()
    print(f"device={dev}", flush=True)
    if args.mode in ("attacks", "both"):
        run_attacks(dev)
    if args.mode in ("attr", "both"):
        run_attr(dev, args.n_attr, args.ig_steps)
