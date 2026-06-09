"""End-to-end experiment orchestration shared by the local smoke run and the Colab grid.

`run_full_experiment(cfg)` produces ONE consolidated results dict (also the schema the figure
code consumes). The only difference between local and Colab is the config (seeds, epochs,
OOD domain, sample caps) -- the logic is identical, so local results are a faithful preview
of the full run.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from .data import (
    load_maide_up_english, make_splits, load_raid_sample, build_attack_set,
)
from .function_words import build_function_word_set
from .models import ModelConfig, build_tokenizer, ReviewDetector
from .train import TrainConfig, train_model
from .evaluate import evaluate_split, classification_metrics
from .baselines import tfidf_lr
from .explain import (
    FaithfulExplainer, faithfulness_report, function_word_attribution_mass,
    function_word_identity_sensitivity,
)
from .utils.seeding import get_device, set_seed
from .utils.stats import aggregate_seeds, mcnemar_test, bootstrap_ci
from .utils.logging import env_info


@dataclass
class ExperimentConfig:
    name: str = "smoke"
    data_csv: str = "/Users/shiva/Detection+XAI/all_data.csv"
    encoder_name: str = "roberta-base"
    variants: tuple = ("baseline", "hardmask", "softreg")
    seeds: tuple = (0, 1, 2)
    fw_definition: str = "union"
    max_length: int = 192
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    softreg_lambda: float = 1.0
    train_subsample: int | None = None   # cap train rows (per seed) for fast local smoke runs
    # OOD (RAID). Local: 'abstracts' (cheap, cross-domain). Colab: e.g. ['reviews','news',...].
    ood_domains: tuple = ("abstracts",)
    ood_cap_per_group: int = 120
    ood_max_scan: int = 60_000
    ood_cache: str = "results/cache/raid_ood.parquet"
    # If set, OOD is built from a pre-downloaded RAID CSV via chunked reads (use this to reach
    # the 'reviews' domain on Colab). When the OOD frame carries a 'model' column,
    # ood_per_generator adds a cross-generator detection breakdown + figure.
    ood_csv_path: str | None = None
    ood_per_generator: bool = True
    # robustness
    attack_types: tuple = ("function_word", "synonym")
    attack_rate: float = 0.5
    # XAI. method="ig" (Captum Integrated Gradients) is the default for GPU/Colab; "occlusion"
    # (leave-one-word-out, forward-only) is robust on CPU/macOS where Captum IG can SIGBUS.
    xai_method: str = "ig"
    faithfulness_n_texts: int = 24
    ig_steps: int = 24
    n_example_explanations: int = 6
    # leakage demonstration (train baseline on a random split too, seed 0 only)
    measure_leakage: bool = True
    # Persist the reference-seed checkpoint per variant (for deployment / release / re-analysis).
    save_models: bool = False
    models_dir: str = "models"
    device: str | None = None
    out_dir: str = "results"


def _ci(values):
    arr = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return bootstrap_ci(arr).as_dict() if arr else None


def run_one_cell(cfg: "ExperimentConfig", variant: str, seed: int, compute_xai: bool,
                 model_out: str | None = None) -> dict:
    """Train+evaluate ONE (variant, seed) and return a partial-result dict.

    Designed to run in its own subprocess (see scripts/run_grid.py) so memory never
    accumulates across the grid and any crash is isolated to a single cell. If `model_out`
    is given, the trained weights are saved there so a separate (small-footprint) XAI cell can
    load them on another device — this lets us train on CPU and attribute on MPS.
    """
    import os
    device = torch.device(cfg.device) if cfg.device else get_device()
    fw_set = build_function_word_set(cfg.fw_definition)
    tokenizer, func_id = build_tokenizer(cfg.encoder_name)
    df = load_maide_up_english(cfg.data_csv)
    splits = make_splits(df, mode="grouped", seed=seed)
    if cfg.train_subsample:
        splits.train = _subsample(splits.train, cfg.train_subsample, seed)
    mcfg = ModelConfig(encoder_name=cfg.encoder_name, variant=variant,
                       max_length=cfg.max_length, softreg_lambda=cfg.softreg_lambda)
    tcfg = TrainConfig(epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)
    set_seed(seed)
    model, history = train_model(mcfg, tcfg, splits, tokenizer, func_id, fw_set, device, seed)

    cell = {"variant": variant, "seed": seed, "history": history}
    ind = evaluate_split(model, splits.test, mcfg, tokenizer, func_id, fw_set, device)
    cell["indomain"] = ind["metrics"]
    cell["indomain_preds"] = {k: ind[k] for k in ("y_true", "y_pred", "p_ai")}

    if cfg.ood_cache and os.path.exists(cfg.ood_cache):
        import pandas as pd
        ood_df = pd.read_parquet(cfg.ood_cache)
        oo = evaluate_split(model, ood_df, mcfg, tokenizer, func_id, fw_set, device)
        cell["ood"] = oo["metrics"]
        cell["ood_preds"] = {k: oo[k] for k in ("y_true", "y_pred", "p_ai")}

    attack_frames = build_attack_set(splits.test, fw_set, attacks=cfg.attack_types, rate=cfg.attack_rate, seed=0)
    cell["attacks"] = {
        a: evaluate_split(model, attack_frames[a], mcfg, tokenizer, func_id, fw_set, device)["metrics"]
        for a in cfg.attack_types
    }

    if model_out:
        import torch as _t
        _t.save(model.state_dict(), model_out)
        cell["model_out"] = model_out
    if compute_xai:
        cell.update(_xai_payload(model, mcfg, tokenizer, func_id, fw_set, splits, cfg, device, variant))
    return cell


def _xai_payload(model, mcfg, tokenizer, func_id, fw_set, splits, cfg, device, variant) -> dict:
    mini = {"variants": {}, "fw_mass": {}, "example_explanations": [],
            "embeddings": {}, "fw_identity_sensitivity": {}}
    _compute_xai_for_variant(model, mcfg, tokenizer, func_id, fw_set, splits, cfg, device, variant, mini)
    return {
        "faithfulness": mini["variants"].get(variant, {}).get("faithfulness"),
        "fw_mass": mini["fw_mass"].get(variant),
        "fw_identity_sensitivity": mini["fw_identity_sensitivity"].get(variant),
        "examples": mini["example_explanations"],
        "embeddings": mini["embeddings"].get(variant),
    }


def run_xai_cell(cfg: "ExperimentConfig", variant: str, model_path: str) -> dict:
    """Load a trained checkpoint and compute ONLY the XAI payload (no training/optimizer).

    Small footprint (model + attribution activations, no Adam states) so it fits MPS even when
    most of the unified memory is taken by other apps. Returns the XAI fields to merge into the
    ref-seed cell.
    """
    import torch as _t
    device = _t.device(cfg.device) if cfg.device else get_device()
    fw_set = build_function_word_set(cfg.fw_definition)
    tokenizer, func_id = build_tokenizer(cfg.encoder_name)
    df = load_maide_up_english(cfg.data_csv)
    splits = make_splits(df, mode="grouped", seed=cfg.seeds[0])
    mcfg = ModelConfig(encoder_name=cfg.encoder_name, variant=variant,
                       max_length=cfg.max_length, softreg_lambda=cfg.softreg_lambda)
    model = ReviewDetector(mcfg, vocab_size=len(tokenizer))
    model.load_state_dict(_t.load(model_path, map_location="cpu"))
    model.to(device).eval()
    return _xai_payload(model, mcfg, tokenizer, func_id, fw_set, splits, cfg, device, variant)


def run_baseline_cell(cfg: "ExperimentConfig", kind: str) -> dict:
    """A classic-baseline or leakage cell (cheap; also isolated in a subprocess)."""
    device = torch.device(cfg.device) if cfg.device else get_device()
    fw_set = build_function_word_set(cfg.fw_definition)
    df = load_maide_up_english(cfg.data_csv)
    splits = make_splits(df, mode="grouped", seed=cfg.seeds[0])
    if kind == "tfidf":
        import os
        out = {}
        for content_only in (False, True):
            b = tfidf_lr(splits.train, splits.test, fw_set=fw_set, content_only=content_only, seed=cfg.seeds[0])
            if cfg.ood_cache and os.path.exists(cfg.ood_cache):
                import pandas as pd
                ood_df = pd.read_parquet(cfg.ood_cache)
                b["ood_metrics"] = tfidf_lr(splits.train, ood_df, fw_set=fw_set, content_only=content_only,
                                            seed=cfg.seeds[0])["metrics"]
            out[b["name"]] = b
        return {"baselines": out, "split_describe": splits.describe()}
    if kind == "leakage_random":
        tokenizer, func_id = build_tokenizer(cfg.encoder_name)
        m = _train_eval_random_baseline(df, cfg, tokenizer, func_id, fw_set, device)
        return {"leakage_random": m}
    raise ValueError(kind)


def _cross_generator(ood_df, ref_preds, variants) -> dict:
    """Per-generator detection macro-F1 (humans vs each RAID generator), ref seed.

    Predictions are in OOD-frame order (DataLoader shuffle=False), so they align with
    ood_df['model']. For each generator g, score on (human rows + g rows)."""
    from sklearn.metrics import f1_score
    models = ood_df["model"].values
    human_mask = models == "human"
    gens = sorted(set(models) - {"human"})
    out = {}
    for v in variants:
        if v not in ref_preds or "ood" not in ref_preds[v]:
            continue
        yt = np.array(ref_preds[v]["ood"]["y_true"])
        yp = np.array(ref_preds[v]["ood"]["y_pred"])
        if len(yt) != len(models):
            continue
        per = {}
        for g in gens:
            sel = human_mask | (models == g)
            if sel.sum() >= 10 and len(set(yt[sel])) == 2:
                per[g] = float(f1_score(yt[sel], yp[sel], average="macro"))
        out[v] = per
    return out


def _subsample(train_df, n, seed):
    """Stratified-by-label subsample of the training frame (for fast smoke runs)."""
    if n >= len(train_df):
        return train_df
    frac = n / len(train_df)
    out = train_df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed)
    )
    return out.reset_index(drop=True)


def run_full_experiment(cfg: ExperimentConfig) -> dict:
    device = torch.device(cfg.device) if cfg.device else get_device()
    print(f"== Experiment '{cfg.name}' on {device} ==")
    fw_set = build_function_word_set(cfg.fw_definition)
    tokenizer, func_id = build_tokenizer(cfg.encoder_name)

    df = load_maide_up_english(cfg.data_csv)
    splits0 = make_splits(df, mode="grouped", seed=cfg.seeds[0])

    # OOD frame (shared across variants/seeds). Prefer a pre-downloaded CSV (reaches 'reviews'
    # quickly via chunked reads); otherwise stream (cheap only for early domains).
    ood_df = None
    try:
        if cfg.ood_csv_path:
            from .data import load_raid_from_csv
            ood_df = load_raid_from_csv(
                cfg.ood_csv_path, domains=cfg.ood_domains, attacks=("none",),
                cap_per_group=cfg.ood_cap_per_group, seed=cfg.seeds[0], cache_path=cfg.ood_cache,
            )
        else:
            ood_df = load_raid_sample(
                domains=cfg.ood_domains, attacks=("none",),
                cap_per_group=cfg.ood_cap_per_group, max_scan=cfg.ood_max_scan,
                seed=cfg.seeds[0], cache_path=cfg.ood_cache,
            )
        print(f"   OOD ({cfg.ood_domains}): {len(ood_df)} rows, "
              f"models={sorted(ood_df['model'].unique())[:8]}")
    except Exception as e:
        print(f"   [warn] OOD load skipped: {type(e).__name__}: {str(e)[:120]}")

    # Attack frames built from seed-0 grouped test split.
    attack_frames = build_attack_set(
        splits0.test, fw_set, attacks=cfg.attack_types, rate=cfg.attack_rate, seed=0
    )

    results: dict = {
        "meta": {"config": asdict(cfg), "env": env_info(),
                 "fw_set_size": len(fw_set), "fw_sources": list(fw_set.sources)},
        "split_describe": splits0.describe(),
        "variants": {}, "baselines": {}, "fw_mass": {}, "leakage": {},
        "example_explanations": [], "significance": {},
    }

    # Reference predictions kept for figures (seed 0).
    ref_preds = {}

    for variant in cfg.variants:
        print(f"\n--- variant: {variant} ---")
        mcfg = ModelConfig(encoder_name=cfg.encoder_name, variant=variant,
                           max_length=cfg.max_length, softreg_lambda=cfg.softreg_lambda)
        tcfg = TrainConfig(epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)
        per_seed_in, per_seed_ood = [], []
        per_seed_attacks = {a: [] for a in cfg.attack_types}
        histories = []
        for seed in cfg.seeds:
            set_seed(seed)
            splits = make_splits(df, mode="grouped", seed=seed)
            if cfg.train_subsample:
                splits.train = _subsample(splits.train, cfg.train_subsample, seed)
            model, history = train_model(
                mcfg, tcfg, splits, tokenizer, func_id, fw_set, device, seed
            )
            histories.append(history)
            print(f"   [{variant}|seed {seed}] eval in-domain", flush=True)
            ind = evaluate_split(model, splits.test, mcfg, tokenizer, func_id, fw_set, device)
            per_seed_in.append(ind["metrics"])
            if seed == cfg.seeds[0]:
                ref_preds[variant] = {"indomain": {k: ind[k] for k in ("y_true", "y_pred", "p_ai")}}
            if ood_df is not None:
                print(f"   [{variant}|seed {seed}] eval OOD", flush=True)
                oo = evaluate_split(model, ood_df, mcfg, tokenizer, func_id, fw_set, device)
                per_seed_ood.append(oo["metrics"])
                if seed == cfg.seeds[0]:
                    ref_preds[variant]["ood"] = {k: oo[k] for k in ("y_true", "y_pred", "p_ai")}
            print(f"   [{variant}|seed {seed}] eval attacks", flush=True)
            for a in cfg.attack_types:
                at = evaluate_split(model, attack_frames[a], mcfg, tokenizer, func_id, fw_set, device)
                per_seed_attacks[a].append(at["metrics"])

            # XAI on the reference seed only (cost). Captum IG can hit unsupported MPS ops,
            # so fall back to CPU rather than letting it abort the whole experiment.
            if seed == cfg.seeds[0]:
                print(f"   [{variant}|seed {seed}] compute XAI", flush=True)
                _compute_xai_for_variant(
                    model, mcfg, tokenizer, func_id, fw_set, splits, cfg, device, variant, results
                )
                if cfg.save_models:
                    os.makedirs(cfg.models_dir, exist_ok=True)
                    mpath = os.path.join(cfg.models_dir, f"{variant}_seed{seed}.pt")
                    torch.save(model.state_dict(), mpath)
                    results.setdefault("model_paths", {})[variant] = mpath
                    print(f"   [{variant}|seed {seed}] saved checkpoint -> {mpath}", flush=True)
            del model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        vres = results["variants"].setdefault(variant, {})
        vres["indomain"] = {"per_seed": per_seed_in, "aggregated": aggregate_seeds(per_seed_in)}
        vres["histories"] = histories
        if per_seed_ood:
            vres["ood"] = {"per_seed": per_seed_ood, "aggregated": aggregate_seeds(per_seed_ood)}
        vres["attacks"] = {
            a: {"per_seed": per_seed_attacks[a], "aggregated": aggregate_seeds(per_seed_attacks[a])}
            for a in cfg.attack_types
        }

    # Cross-generator breakdown (ref seed): detection macro-F1, humans vs each RAID generator.
    if ood_df is not None and cfg.ood_per_generator and "model" in ood_df.columns:
        results["cross_generator"] = _cross_generator(ood_df, ref_preds, list(cfg.variants))

    # Classic baselines (seed-0 grouped split).
    for content_only in (False, True):
        b = tfidf_lr(splits0.train, splits0.test, fw_set=fw_set, content_only=content_only, seed=cfg.seeds[0])
        results["baselines"][b["name"]] = b
        if ood_df is not None:
            bo = tfidf_lr(splits0.train, ood_df, fw_set=fw_set, content_only=content_only, seed=cfg.seeds[0])
            results["baselines"][b["name"]]["ood_metrics"] = bo["metrics"]

    results["ref_preds"] = ref_preds

    # Significance: McNemar baseline vs hardmask on in-domain (ref seed).
    if "baseline" in ref_preds and "hardmask" in ref_preds:
        a = ref_preds["baseline"]["indomain"]; b = ref_preds["hardmask"]["indomain"]
        results["significance"]["indomain_baseline_vs_hardmask"] = mcnemar_test(
            a["y_true"], a["y_pred"], b["y_pred"]
        )

    # Leakage demonstration: baseline on random vs grouped split (seed 0).
    # The grouped number is the already-trained baseline ref seed; we only train the random one.
    if cfg.measure_leakage and "baseline" in results["variants"]:
        grouped_metrics = results["variants"]["baseline"]["indomain"]["per_seed"][0]
        random_metrics = _train_eval_random_baseline(df, cfg, tokenizer, func_id, fw_set, device)
        results["leakage"] = {"grouped": grouped_metrics, "random": random_metrics}

    return results


def _project_2d(emb: np.ndarray, labels: list) -> dict:
    """2D projection of embeddings (UMAP, PCA fallback) for the scatter figure."""
    try:
        import umap
        coords = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(emb)
        method = "umap"
    except Exception:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=0).fit_transform(emb)
        method = "pca"
    return {"coords": coords.tolist(), "labels": list(labels), "method": method}


def _compute_xai_for_variant(model, mcfg, tokenizer, func_id, fw_set, splits, cfg, device, variant, results):
    """Faithfulness + FW-mass + example explanations + embedding projection, with CPU fallback."""
    from .evaluate import extract_embeddings
    from .train import make_collator

    # Free training gradients before the (memory-heavy) attribution pass.
    model.zero_grad(set_to_none=True)
    gc.collect()
    # macOS Accelerate/BLAS SIGBUSes on single-example forwards AFTER multi-threaded training;
    # running the XAI phase single-threaded avoids it (training stays multi-threaded).
    _prev_threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(1)
    try:
        return _compute_xai_inner(
            model, mcfg, tokenizer, func_id, fw_set, splits, cfg, device, variant, results,
            extract_embeddings, make_collator,
        )
    finally:
        torch.set_num_threads(_prev_threads)


def _compute_xai_inner(model, mcfg, tokenizer, func_id, fw_set, splits, cfg, device, variant,
                       results, extract_embeddings, make_collator):
    for attempt in (device, torch.device("cpu")):
        try:
            if attempt != device:
                print(f"   [info] retrying XAI for '{variant}' on CPU")
                model.to(attempt)
            explainer = FaithfulExplainer(model, mcfg, tokenizer, func_id, fw_set, attempt)
            fr = faithfulness_report(
                explainer, splits.test["text"].tolist(),
                method=cfg.xai_method, n_steps=cfg.ig_steps, max_texts=cfg.faithfulness_n_texts,
            )
            results["variants"].setdefault(variant, {})["faithfulness"] = fr
            fw_masses = [r["fw_mass"] for r in fr["per_text"]]
            results["fw_mass"][variant] = {
                "values": fw_masses, "mean": float(np.mean(fw_masses)), "ci": _ci(fw_masses),
                "method": cfg.xai_method,
            }
            # Direct test of the requirement: does swapping function-word identity move the
            # decision? (Exactly 0 for Hard-Mask.) Forward-only -> robust on every backend.
            sens = function_word_identity_sensitivity(
                explainer, splits.test["text"].tolist(), max_texts=cfg.faithfulness_n_texts
            )
            results.setdefault("fw_identity_sensitivity", {})[variant] = {
                "values": sens, "mean": float(np.mean(sens)), "ci": _ci(sens),
            }
            results["example_explanations"].extend(
                _collect_examples(explainer, splits.test, variant, cfg.n_example_explanations, cfg.xai_method)
            )
            coll = make_collator(mcfg, tokenizer, func_id, fw_set)
            emb = extract_embeddings(model, splits.test, coll, attempt)
            results.setdefault("embeddings", {})[variant] = _project_2d(
                emb, splits.test["label"].tolist()
            )
            if attempt != device:
                model.to(device)
            return
        except Exception as e:
            print(f"   [warn] XAI on {attempt} failed: {type(e).__name__}: {str(e)[:100]}")
    print(f"   [warn] XAI skipped for '{variant}'")


def _collect_examples(explainer, test_df, variant, n, method="ig"):
    texts = test_df["text"].tolist()[:n]
    out = []
    for t in texts:
        attr = explainer.occlusion(t) if method == "occlusion" else explainer.integrated_gradients(t, n_steps=24)
        out.append({
            "variant": variant, "text": t, "words": attr.words,
            "word_is_fw": attr.word_is_fw, "word_scores": attr.word_scores,
            "p_ai": attr.p_ai, "predicted_label": attr.predicted_label,
            "fw_mass": function_word_attribution_mass(attr)["fw_mass"],
        })
    return out


def _train_eval_random_baseline(df, cfg, tokenizer, func_id, fw_set, device):
    """Train a baseline on a RANDOM (leaky) split to contrast with the grouped split (F10)."""
    mcfg = ModelConfig(encoder_name=cfg.encoder_name, variant="baseline", max_length=cfg.max_length)
    tcfg = TrainConfig(epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr)
    set_seed(cfg.seeds[0])
    sp = make_splits(df, mode="random", seed=cfg.seeds[0])
    if cfg.train_subsample:
        sp.train = _subsample(sp.train, cfg.train_subsample, cfg.seeds[0])
    model, _ = train_model(mcfg, tcfg, sp, tokenizer, func_id, fw_set, device, cfg.seeds[0])
    ev = evaluate_split(model, sp.test, mcfg, tokenizer, func_id, fw_set, device)
    del model; gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return ev["metrics"]
