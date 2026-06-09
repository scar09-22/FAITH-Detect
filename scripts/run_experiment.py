"""Run a single experiment from a YAML config (used for full/ablation runs and Colab).

Example:
    python scripts/run_experiment.py --config configs/base.yaml --out results/base.json
"""
import argparse
from dataclasses import fields

import yaml

import _bootstrap  # noqa: F401
from faithdetect.experiment import ExperimentConfig, run_full_experiment
from faithdetect.utils.logging import save_json
from faithdetect.viz import make_all_figures


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    valid = {f.name for f in fields(ExperimentConfig)}
    kwargs = {}
    for k, v in raw.items():
        if k not in valid:
            continue
        # tuples for sequence fields
        kwargs[k] = tuple(v) if isinstance(v, list) else v
    return ExperimentConfig(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    results = run_full_experiment(cfg)
    out = args.out or f"results/{cfg.name}_results.json"
    save_json(out, results)
    print(f"\nSaved results -> {out}")
    figs = make_all_figures(results, args.figdir)
    print(f"Rendered {len(figs)} figures -> {args.figdir}/")


if __name__ == "__main__":
    main()
