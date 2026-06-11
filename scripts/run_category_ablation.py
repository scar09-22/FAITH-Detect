"""Per-category function-word masking ablation.

Trains a Hard-Mask detector that masks ONLY one grammatical category of function words
(determiners, pronouns, prepositions, conjunctions, auxiliaries, adverbial) at a time, plus
the full-union reference, and compares in-domain F1 and OOD transfer per category. The
question: which categories carry domain-general signal (masking them should hurt OOD) and
which are safe to remove (masking costs nothing anywhere)?

Each (category, seed) runs in an isolated subprocess via scripts/run_cell.py (same machinery
as run_grid.py), so a crash loses one cell only.

Example:
  python scripts/run_category_ablation.py --seeds 0 1 --epochs 3 --train_subsample 600 \
      --ood_parquet results/cache/raid_reviews_pool.parquet
"""
import argparse
import os
from dataclasses import asdict

import _bootstrap  # noqa: F401
from faithdetect.experiment import ExperimentConfig
from faithdetect.function_words import FW_CATEGORIES
from faithdetect.utils.logging import save_json
from faithdetect.utils.stats import aggregate_seeds

from run_grid import run_subproc  # noqa: E402  (sibling script; sys.path[0] is scripts/)

CELL_DIR = "results/cells/catablation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variant", default="hardmask")
    ap.add_argument("--categories", nargs="+",
                    default=sorted(FW_CATEGORIES) + ["union"],
                    help="categories to ablate; 'union' = mask all function words (reference)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--train_subsample", type=int, default=600)
    ap.add_argument("--ood_parquet", default="results/cache/raid_reviews_pool.parquet",
                    help="OOD eval frame (text,label); used as ood_cache for every cell")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out", default="results/category_ablation.json")
    args = ap.parse_args()

    os.makedirs(CELL_DIR, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    results = {"meta": {"args": vars(args)}, "categories": {}}
    for cat in args.categories:
        fw_def = cat if cat == "union" else f"cat:{cat}"
        cfg = ExperimentConfig(
            name=f"catablation_{cat}", data_csv=args.data_csv, encoder_name=args.encoder,
            variants=(args.variant,), seeds=tuple(args.seeds),
            epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length,
            train_subsample=args.train_subsample, fw_definition=fw_def,
            attack_types=("function_word",), measure_leakage=False,
            ood_cache=args.ood_parquet if os.path.exists(args.ood_parquet) else None,
            device=args.device,
        )
        cfg_path = os.path.join(CELL_DIR, f"cfg_{cat}.json")
        save_json(cfg_path, asdict(cfg))
        per_seed = []
        for s in args.seeds:
            out = os.path.join(CELL_DIR, f"{cat}_s{s}.json")
            print(f"== category {cat} seed {s} (fw={fw_def}) ==", flush=True)
            cell = run_subproc(
                ["--config_json", cfg_path, "--variant", args.variant, "--seed", str(s),
                 "--xai", "0", "--device", args.device, "--out", out],
                out, retries=args.retries, env=env,
            )
            if cell:
                rec = {"indomain": cell["indomain"]}
                if cell.get("ood"):
                    rec["ood"] = cell["ood"]
                if cell.get("attacks"):
                    rec["fw_attack"] = cell["attacks"].get("function_word")
                per_seed.append(rec)
        results["categories"][cat] = {
            "fw_definition": fw_def,
            "per_seed": per_seed,
            "indomain": aggregate_seeds([r["indomain"] for r in per_seed]),
            "ood": aggregate_seeds([r["ood"] for r in per_seed if "ood" in r]),
            "fw_attack": aggregate_seeds([r["fw_attack"] for r in per_seed if r.get("fw_attack")]),
        }
        save_json(args.out, results)  # checkpoint after every category

    try:
        from faithdetect.viz.figures import fig_category_ablation, PUBLICATION_STYLE
        import matplotlib.pyplot as plt
        plt.rcParams.update(PUBLICATION_STYLE)
        fig_category_ablation(results, "figures/16_category_ablation.png")
    except Exception as e:
        print(f"[warn] ablation figure failed: {type(e).__name__}: {e}")

    print(f"\nSaved {args.out}")
    print(f"{'category':<14}{'in-domain F1':>14}{'OOD F1':>10}")
    for cat, d in results["categories"].items():
        f1 = d["indomain"].get("f1_macro", {}).get("mean", float("nan"))
        oo = d["ood"].get("f1_macro", {}).get("mean", float("nan")) if d["ood"] else float("nan")
        print(f"{cat:<14}{f1*100:>13.1f}%{oo*100:>9.1f}%")


if __name__ == "__main__":
    main()
