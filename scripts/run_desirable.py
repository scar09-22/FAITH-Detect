#!/usr/bin/env python3
"""Desirable reviewer revisions, trained at PILOT scale (distilroberta-base, 600-subsample).

A. Masking-variant comparison -> results/variant_comparison.json
   baseline / hardmask([FUNC]) / deletion(remove FW) / random(random-token placeholder),
   to separate function-word IDENTITY, the slot-MARKER, and POSITION/COUNT effects.
   (SoftReg is in the main run; category-specific masks are in category_ablation.json.)

B. SoftReg lambda sweep -> results/lambda_sweep.json
   in-domain F1 and identity-sensitivity across lambda, tracing the reliance--accuracy frontier.
"""
from __future__ import annotations
import argparse, gc, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from faithdetect.models import ReviewDetector, ModelConfig, build_tokenizer  # noqa
from faithdetect.function_words import build_function_word_set
from faithdetect.data import load_maide_up_english, make_splits
from faithdetect.train import TrainConfig, train_model
from faithdetect.evaluate import evaluate_split
from faithdetect.experiment import _subsample
from faithdetect.explain.attributions import FaithfulExplainer
from faithdetect.explain.faithfulness import function_word_identity_sensitivity

# Defaults are the local PILOT config; the Colab notebook overrides them via CLI for the full run.
DATA, ENC, MAXLEN, FWDEF, SUB, EPOCHS = "data/all_data.csv", "distilroberta-base", 112, "union", 600, 3
OOD = "results/cache/raid_ood.parquet"
DEVICE_OVERRIDE = None      # set from --device
PENALTY_BATCH = 2           # softreg 2nd-order sub-batch (raise on GPU)


def dev():
    if DEVICE_OVERRIDE:
        return torch.device(DEVICE_OVERRIDE)
    forced = os.environ.get("FD_DEVICE")   # FD_DEVICE=cpu avoids MPS OOM on the 8 GB box
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def ci(vals):
    a = np.array(vals, dtype=float); n = len(a); m = float(a.mean())
    if n < 2:
        return {"mean": m, "lo": m, "hi": m, "n": n}
    sd = float(a.std(ddof=1)); t = 12.706 if n == 2 else (4.303 if n == 3 else 2.776)
    h = t * sd / math.sqrt(n)
    return {"mean": m, "lo": m - h, "hi": m + h, "sd": sd, "n": n}


def _prep(seed):
    fw = build_function_word_set(FWDEF)
    tok, fid = build_tokenizer(ENC)
    df = load_maide_up_english(DATA)
    sp = make_splits(df, mode="grouped", seed=seed)
    if SUB and SUB > 0:                 # SUB=0/None -> full training set (Colab full scale)
        sp.train = _subsample(sp.train, SUB, seed)
    return fw, tok, fid, sp


def _train_eval(variant, seed, lam=1.0, need_ood=True):
    device = dev()
    fw, tok, fid, sp = _prep(seed)
    # softreg_penalty_batch bounds the second-order (create_graph) penalty memory (small on the
    # 8 GB box; the Colab notebook raises it on GPU). batch_size shrinks for softreg too.
    mcfg = ModelConfig(encoder_name=ENC, variant=variant, max_length=MAXLEN,
                       softreg_lambda=lam, softreg_penalty_batch=PENALTY_BATCH)
    bs = 8 if variant == "softreg" else 16
    tcfg = TrainConfig(epochs=EPOCHS, batch_size=bs, lr=2e-5)
    model, hist = train_model(mcfg, tcfg, sp, tok, fid, fw, device, seed)
    ind = evaluate_split(model, sp.test, mcfg, tok, fid, fw, device)["metrics"]
    ood = None
    if need_ood and OOD and os.path.exists(OOD):
        ood_df = pd.read_parquet(OOD)[["text", "label"]]
        ood = evaluate_split(model, ood_df, mcfg, tok, fid, fw, device)["metrics"]
    return model, mcfg, tok, fid, fw, device, sp, ind, ood


def run_variants(seeds):
    variants = ["baseline", "hardmask", "deletion", "random"]
    per = {v: {"indomain_f1": [], "ood_f1": []} for v in variants}
    for v in variants:
        for s in seeds:
            t0 = time.time()
            res = _train_eval(v, s)
            model, ind, ood = res[0], res[7], res[8]
            oodf1 = ood["f1_macro"] * 100 if ood else float("nan")
            per[v]["indomain_f1"].append(ind["f1_macro"] * 100)
            per[v]["ood_f1"].append(oodf1)
            print(f"[var] {v} seed{s}: in={ind['f1_macro']*100:.1f} ood={oodf1:.1f} ({time.time()-t0:.0f}s)", flush=True)
            del model, res
            _cleanup()
        # incremental write so a crash mid-run is recoverable
        json.dump({"variants": {vv: {"indomain_f1": ci(per[vv]["indomain_f1"]),
                                     "ood_f1": ci(per[vv]["ood_f1"])}
                                for vv in variants if per[vv]["indomain_f1"]}},
                  open("results/variant_comparison_partial.json", "w"), indent=2)
    out = {"meta": {"encoder": ENC, "train_subsample": SUB, "epochs": EPOCHS, "seeds": seeds,
                    "ood": OOD,
                    "note": "baseline=identity+slots+content; hardmask=identity removed slot kept; "
                            "deletion=identity+slot removed; random=identity removed, consistent slot-marker removed."},
           "variants": {v: {"indomain_f1": ci(per[v]["indomain_f1"]),
                            "ood_f1": ci(per[v]["ood_f1"])} for v in variants}}
    json.dump(out, open("results/variant_comparison.json", "w"), indent=2)
    print("wrote results/variant_comparison.json", flush=True)


def run_lambda(lambdas, seeds, n_sens=40):
    rows = []
    for lam in lambdas:
        f1s, sens = [], []
        for s in seeds:
            model, mcfg, tok, fid, fw, device, sp, ind, _ = _train_eval("softreg", s, lam=lam, need_ood=False)
            f1s.append(ind["f1_macro"] * 100)
            expl = FaithfulExplainer(model, mcfg, tok, fid, fw, device)
            texts = list(sp.test["text"].head(n_sens))
            vals = function_word_identity_sensitivity(expl, texts, seed=s, max_texts=n_sens)
            sens.append(float(np.mean([abs(x) for x in vals])) * 100)
            del model, expl
            _cleanup()
        rows.append({"lambda": lam, "indomain_f1": ci(f1s), "identity_sensitivity": ci(sens)})
        print(f"[lam] λ={lam}: F1={np.mean(f1s):.1f} identity-sens={np.mean(sens):.2f}%", flush=True)
    out = {"meta": {"encoder": ENC, "train_subsample": SUB, "epochs": EPOCHS, "seeds": seeds,
                    "note": "SoftReg lambda sweep; lambda=0 ~ baseline."}, "sweep": rows}
    json.dump(out, open("results/lambda_sweep.json", "w"), indent=2)
    print("wrote results/lambda_sweep.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["variants", "lambda", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lambda-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 5.0])
    # full-scale overrides (Colab): --encoder roberta-base --subsample 0 --device cuda ...
    ap.add_argument("--encoder", default=ENC)
    ap.add_argument("--max-length", type=int, default=MAXLEN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--subsample", type=int, default=SUB, help="0 = full training set")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    ap.add_argument("--ood-parquet", default=OOD, help="OOD parquet for the variant comparison")
    ap.add_argument("--penalty-batch", type=int, default=PENALTY_BATCH)
    a = ap.parse_args()
    ENC, MAXLEN, EPOCHS, SUB = a.encoder, a.max_length, a.epochs, a.subsample
    OOD, DEVICE_OVERRIDE, PENALTY_BATCH = a.ood_parquet, a.device, a.penalty_batch
    print(f"device={dev()} encoder={ENC} subsample={SUB} epochs={EPOCHS} penalty_batch={PENALTY_BATCH}", flush=True)
    if a.mode in ("variants", "both"):
        run_variants(a.seeds)
    if a.mode in ("lambda", "both"):
        run_lambda(a.lambdas, a.lambda_seeds)
