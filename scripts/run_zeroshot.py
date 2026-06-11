"""Run the training-free zero-shot detector baselines (log-likelihood, log-rank, entropy,
Fast-DetectGPT) on the same evaluation frames as the supervised models:

  * the grouped in-domain test split,
  * the function-word attack version of that split (do the statistical detectors survive
    the attack the Hard-Mask model is invariant to?),
  * optionally an OOD frame (e.g. the RAID reviews pool) — zero-shot methods carry no
    domain-specific training, so this is an informative reference point for the transfer
    discussion.

Thresholds are fit on the training split only (see faithdetect.zeroshot).

Example:
  python scripts/run_zeroshot.py --ood_parquet results/cache/raid_reviews_pool.parquet
"""
import argparse
import os

import _bootstrap  # noqa: F401
from faithdetect.data import load_maide_up_english, make_splits, build_attack_set
from faithdetect.function_words import build_function_word_set
from faithdetect.zeroshot import evaluate_zeroshot, ZEROSHOT_METHODS
from faithdetect.utils.logging import save_json, env_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=400)
    ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--attack_rate", type=float, default=0.5)
    ap.add_argument("--ood_parquet", default="results/cache/raid_reviews_pool.parquet")
    ap.add_argument("--max_ood", type=int, default=600)
    ap.add_argument("--out", default="results/zeroshot.json")
    args = ap.parse_args()

    df = load_maide_up_english(args.data_csv)
    sp = make_splits(df, mode="grouped", seed=args.seed)
    fw = build_function_word_set("union")

    results = {"meta": {"args": vars(args), "env": env_info(), "methods": list(ZEROSHOT_METHODS)}}

    print("== zero-shot: in-domain test ==", flush=True)
    results["indomain"] = evaluate_zeroshot(
        sp.train, sp.test, model_name=args.model_name, device=args.device,
        max_train=args.max_train, max_test=args.max_test, seed=args.seed,
    )
    save_json(args.out, results)

    print("== zero-shot: function-word attack ==", flush=True)
    attacked = build_attack_set(sp.test, fw, attacks=("function_word",),
                                rate=args.attack_rate, seed=0)["function_word"]
    results["fw_attack"] = evaluate_zeroshot(
        sp.train, attacked, model_name=args.model_name, device=args.device,
        max_train=args.max_train, max_test=args.max_test, seed=args.seed,
    )
    save_json(args.out, results)

    if args.ood_parquet and os.path.exists(args.ood_parquet):
        import pandas as pd
        print("== zero-shot: OOD ==", flush=True)
        ood = pd.read_parquet(args.ood_parquet)
        results["ood"] = evaluate_zeroshot(
            sp.train, ood, model_name=args.model_name, device=args.device,
            max_train=args.max_train, max_test=args.max_ood, seed=args.seed,
        )
        save_json(args.out, results)

    print(f"\nSaved {args.out}")
    for axis in ("indomain", "fw_attack", "ood"):
        if axis not in results:
            continue
        print(f"--- {axis} ---")
        for m, d in results[axis].items():
            print(f"  {m:16s} f1_macro={d['metrics']['f1_macro']*100:.1f}%  auroc={d['auroc']:.3f}")


if __name__ == "__main__":
    main()
