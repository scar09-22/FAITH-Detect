"""Run ONE experiment cell (a variant/seed, or a baseline) in an isolated process.

Invoked by scripts/run_grid.py. Writes a partial-result JSON. Crashing here only loses one
cell, which the grid driver retries.
"""
import argparse
import faulthandler
faulthandler.enable()
import _bootstrap  # noqa: F401

from faithdetect.experiment import ExperimentConfig, run_one_cell, run_baseline_cell, run_xai_cell
from faithdetect.utils.logging import save_json, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_json", required=True)
    ap.add_argument("--mode", default="cell", choices=["cell", "xai", "baseline"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--xai", type=int, default=0)
    ap.add_argument("--kind", default=None, help="tfidf | leakage_random (baseline cells)")
    ap.add_argument("--device", default=None, help="override cfg.device for this cell")
    ap.add_argument("--model_out", default=None, help="(cell) save trained weights here")
    ap.add_argument("--model_path", default=None, help="(xai) load weights from here")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = load_json(args.config_json)
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    cfg = ExperimentConfig(**d)
    if args.device:
        cfg.device = args.device

    if args.mode == "baseline" or args.kind:
        result = run_baseline_cell(cfg, args.kind)
    elif args.mode == "xai":
        result = run_xai_cell(cfg, args.variant, args.model_path)
    else:
        result = run_one_cell(cfg, args.variant, args.seed, bool(args.xai), model_out=args.model_out)
    save_json(args.out, result)
    print(f"cell OK -> {args.out}")


if __name__ == "__main__":
    main()
