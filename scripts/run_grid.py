"""Process-isolated grid driver: runs each (variant, seed) cell in its own subprocess with
retries, then merges partials into the consolidated results structure and renders figures.

This is the robust path on memory-constrained / flaky machines (e.g. an 8 GB Mac where a long
single process OOMs on MPS or intermittently SIGBUSes on CPU). Each cell is short-lived with a
bounded footprint; a crash is isolated and retried.

Example:
  python scripts/run_grid.py --encoder distilroberta-base --device cpu --seeds 0 1 2 \
      --epochs 3 --batch_size 8 --max_length 128 --train_subsample 600 --xai_method ig
"""
import argparse
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np

import _bootstrap  # noqa: F401
from faithdetect.experiment import ExperimentConfig
from faithdetect.function_words import build_function_word_set
from faithdetect.data import load_raid_sample
from faithdetect.utils.logging import save_json, load_json, env_info
from faithdetect.utils.stats import aggregate_seeds, bootstrap_ci, mcnemar_test
from faithdetect.viz import make_all_figures

CELL_DIR = "results/cells"
PY = sys.executable


def _ci(values):
    arr = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return bootstrap_ci(arr).as_dict() if arr else None


def run_subproc(cell_args, out, retries=2, env=None):
    for attempt in range(retries + 1):
        if os.path.exists(out):
            os.remove(out)
        proc = subprocess.run([PY, "-u", "scripts/run_cell.py"] + cell_args, env=env)
        if proc.returncode == 0 and os.path.exists(out):
            return load_json(out)
        print(f"   [retry] cell failed (exit {proc.returncode}); attempt {attempt+1}/{retries+1}")
    return None


def build_cfg(args) -> ExperimentConfig:
    return ExperimentConfig(
        name=args.name, data_csv=args.data_csv, encoder_name=args.encoder,
        variants=tuple(args.variants), seeds=tuple(args.seeds), max_length=args.max_length,
        epochs=args.epochs, batch_size=args.batch_size, train_subsample=args.train_subsample,
        softreg_lambda=args.softreg_lambda, xai_method=args.xai_method,
        fw_definition=args.fw_definition,
        faithfulness_n_texts=args.faithfulness_n_texts, ig_steps=args.ig_steps,
        ood_domains=tuple(args.ood_domains), device=args.device,
        train_mix_parquet=args.train_mix_parquet,
        train_mix_generators=tuple(args.train_mix_generators),
        heldout_generators=tuple(args.heldout_generators),
        measure_leakage=not args.no_leakage,
    )


def merge(cfg, cells, base, leak) -> dict:
    seeds = list(cfg.seeds)
    _fw = build_function_word_set(cfg.fw_definition)
    results = {
        "meta": {"config": asdict(cfg), "env": env_info(),
                 "fw_set_size": len(_fw), "fw_sources": list(_fw.sources)},
        "variants": {}, "baselines": {}, "fw_mass": {}, "fw_identity_sensitivity": {},
        "embeddings": {}, "leakage": {}, "example_explanations": [], "significance": {},
        "ref_preds": {},
    }
    if base:
        results["baselines"] = base.get("baselines", {})
        results["split_describe"] = base.get("split_describe", {})

    for v in cfg.variants:
        cs = [cells[(v, s)] for s in seeds if cells.get((v, s))]
        if not cs:
            continue
        per_in = [c["indomain"] for c in cs]
        per_ood = [c["ood"] for c in cs if c.get("ood")]
        per_ho = [c["heldout_gen"] for c in cs if c.get("heldout_gen")]
        attacks = list(cs[0].get("attacks", {}).keys())
        vres = {
            "indomain": {"per_seed": per_in, "aggregated": aggregate_seeds(per_in)},
            "histories": [c["history"] for c in cs],
            "attacks": {
                a: {"per_seed": [c["attacks"][a] for c in cs if a in c.get("attacks", {})],
                    "aggregated": aggregate_seeds([c["attacks"][a] for c in cs if a in c.get("attacks", {})])}
                for a in attacks
            },
        }
        if per_ood:
            vres["ood"] = {"per_seed": per_ood, "aggregated": aggregate_seeds(per_ood)}
        if per_ho:
            vres["heldout_gen"] = {"per_seed": per_ho, "aggregated": aggregate_seeds(per_ho)}
            ref_ho = next((c for c in cs if c.get("heldout_per_generator")), None)
            if ref_ho:
                results.setdefault("heldout_per_generator", {})[v] = ref_ho["heldout_per_generator"]
        ref = next((c for c in cs if c["seed"] == seeds[0]), cs[0])
        results["ref_preds"][v] = {"indomain": ref.get("indomain_preds")}
        if ref.get("ood_preds"):
            results["ref_preds"][v]["ood"] = ref["ood_preds"]
        if ref.get("faithfulness"):
            vres["faithfulness"] = ref["faithfulness"]
        if ref.get("fw_mass"):
            results["fw_mass"][v] = ref["fw_mass"]
        if ref.get("fw_identity_sensitivity"):
            results["fw_identity_sensitivity"][v] = ref["fw_identity_sensitivity"]
        if ref.get("examples"):
            results["example_explanations"].extend(ref["examples"])
        if ref.get("embeddings"):
            results["embeddings"][v] = ref["embeddings"]
        results["variants"][v] = vres

    # Cross-generator breakdown from the cached OOD frame (has a 'model' column).
    if cfg.ood_cache and os.path.exists(cfg.ood_cache):
        try:
            import pandas as pd
            from faithdetect.experiment import _cross_generator
            ood_df = pd.read_parquet(cfg.ood_cache)
            if "model" in ood_df.columns:
                cg = _cross_generator(ood_df, results["ref_preds"], list(cfg.variants))
                if cg:
                    results["cross_generator"] = cg
        except Exception as e:
            print(f"   [warn] cross-generator skipped: {type(e).__name__}: {str(e)[:80]}")

    # Leakage: grouped = baseline ref-seed indomain; random = dedicated cell.
    if "baseline" in results["variants"] and leak:
        results["leakage"] = {
            "grouped": results["variants"]["baseline"]["indomain"]["per_seed"][0],
            "random": leak["leakage_random"],
        }
    # Significance (baseline vs hardmask, ref seed).
    rp = results["ref_preds"]
    if "baseline" in rp and "hardmask" in rp and rp["baseline"].get("indomain") and rp["hardmask"].get("indomain"):
        a, b = rp["baseline"]["indomain"], rp["hardmask"]["indomain"]
        results["significance"]["indomain_baseline_vs_hardmask"] = mcnemar_test(
            a["y_true"], a["y_pred"], b["y_pred"]
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="grid")
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants", nargs="+", default=["baseline", "hardmask", "softreg"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--train_subsample", type=int, default=600)
    ap.add_argument("--softreg_lambda", type=float, default=0.5)
    ap.add_argument("--xai_method", default="ig")
    ap.add_argument("--fw_definition", default="union",
                    help="union | curated | stopwords | cat:<category> (per-category ablation)")
    ap.add_argument("--train_mix_parquet", default=None,
                    help="RAID reviews pool parquet for mixed-generator training")
    ap.add_argument("--train_mix_generators", nargs="*", default=[],
                    help="held-IN generators added to training")
    ap.add_argument("--heldout_generators", nargs="*", default=[],
                    help="held-OUT generators for same-domain cross-generator eval")
    ap.add_argument("--no_leakage", action="store_true",
                    help="skip the random-split leakage cell (saves one training run)")
    ap.add_argument("--faithfulness_n_texts", type=int, default=16)
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--ood_domains", nargs="+", default=["abstracts"])
    # Device routing: train-only cells are robust on CPU; only XAI cells need MPS (the macOS
    # Accelerate attribution crash is CPU-only, and a single XAI cell fits MPS memory).
    ap.add_argument("--train_device", default=None, help="device for train-only cells (default = --device)")
    ap.add_argument("--xai_device", default=None, help="device for XAI cells (default = --device)")
    ap.add_argument("--out", default="results/smoke_results.json")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    cfg = build_cfg(args)
    cell_dir = os.path.join(CELL_DIR, args.name)
    os.makedirs(cell_dir, exist_ok=True)

    # Pre-build the RAID OOD cache once (so each cell just reads the parquet).
    if cfg.ood_cache:
        try:
            load_raid_sample(domains=cfg.ood_domains, cap_per_group=cfg.ood_cap_per_group,
                             max_scan=cfg.ood_max_scan, seed=cfg.seeds[0], cache_path=cfg.ood_cache)
            print(f"OOD cache ready: {cfg.ood_cache}")
        except Exception as e:
            print(f"[warn] OOD cache build failed: {type(e).__name__}: {str(e)[:100]}")

    cfg_path = os.path.join(cell_dir, "cfg.json")
    save_json(cfg_path, asdict(cfg))

    env = dict(os.environ)
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    train_dev = args.train_device or cfg.device
    xai_dev = args.xai_device or cfg.device

    cells = {}
    for v in cfg.variants:
        for s in cfg.seeds:
            ref = (s == cfg.seeds[0])
            out = os.path.join(cell_dir, f"{v}_s{s}.json")
            # Train+eval ALL cells on the train device (CPU is robust; no attribution here).
            # Ref-seed cells also save their weights so a separate XAI cell can attribute them.
            cell_args = ["--config_json", cfg_path, "--variant", v, "--seed", str(s),
                         "--xai", "0", "--device", train_dev, "--out", out]
            model_path = os.path.join(cell_dir, f"{v}_s{s}_model.pt")
            if ref:
                cell_args += ["--model_out", model_path]
            print(f"== train cell {v} seed {s} (device={train_dev}{', +save' if ref else ''}) ==")
            res = run_subproc(cell_args, out, retries=args.retries, env=env)
            cells[(v, s)] = res

            # Attribute the ref-seed model on the XAI device (small footprint; fits MPS).
            if ref and res is not None and os.path.exists(model_path):
                xout = os.path.join(cell_dir, f"{v}_xai.json")
                print(f"== xai cell {v} (device={xai_dev}, load {os.path.basename(model_path)}) ==")
                xres = run_subproc(
                    ["--config_json", cfg_path, "--mode", "xai", "--variant", v,
                     "--model_path", model_path, "--device", xai_dev, "--out", xout],
                    xout, retries=args.retries, env=env,
                )
                if xres:
                    cells[(v, s)].update({k: xres.get(k) for k in
                                          ("faithfulness", "fw_mass", "fw_identity_sensitivity",
                                           "examples", "embeddings")})
                else:
                    print(f"   [warn] XAI never succeeded for {v}; metrics kept, XAI figures skipped")

    base = run_subproc(["--config_json", cfg_path, "--kind", "tfidf", "--device", train_dev,
                        "--out", os.path.join(cell_dir, "tfidf.json")],
                       os.path.join(cell_dir, "tfidf.json"), retries=args.retries, env=env)
    leak = None
    if cfg.measure_leakage:
        leak = run_subproc(["--config_json", cfg_path, "--kind", "leakage_random", "--device", train_dev,
                            "--out", os.path.join(cell_dir, "leak.json")],
                           os.path.join(cell_dir, "leak.json"), retries=args.retries, env=env)

    results = merge(cfg, cells, base, leak)
    save_json(args.out, results)
    print(f"\nMerged results -> {args.out}")
    figs = make_all_figures(results, args.figdir)
    print(f"Rendered {len(figs)} figures -> {args.figdir}/")


if __name__ == "__main__":
    main()
