"""Genuine, reduced-scale local smoke run (real training on MPS, real CIs over seeds).

Produces results/smoke_results.json and renders all figures. The full grid (5 seeds, more
epochs, full RAID OOD + attacks) runs from notebooks/colab_full_run.ipynb with the SAME
`run_full_experiment`, so these results are a faithful preview.
"""
import argparse
import faulthandler
faulthandler.enable()  # dump a stack trace on hard faults (segfaults) instead of vanishing
import _bootstrap  # noqa: F401

from faithdetect.experiment import ExperimentConfig, run_full_experiment
from faithdetect.utils.logging import save_json
from faithdetect.viz import make_all_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="/Users/shiva/Detection+XAI/all_data.csv")
    ap.add_argument("--encoder", default="roberta-base",
                    help="encoder (use distilroberta-base for a fast, light local smoke)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--train_subsample", type=int, default=None)
    ap.add_argument("--softreg_lambda", type=float, default=0.5)
    ap.add_argument("--ig_steps", type=int, default=24)
    ap.add_argument("--xai_method", default="ig", choices=["occlusion", "ig"],
                    help="ig = manual Integrated Gradients (robust on CPU/MPS/CUDA); occlusion = forward-only")
    ap.add_argument("--ood_domains", nargs="+", default=["abstracts"])
    ap.add_argument("--faithfulness_n_texts", type=int, default=16)
    ap.add_argument("--out", default="results/smoke_results.json")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig(
        name="smoke",
        data_csv=args.data_csv,
        encoder_name=args.encoder,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        train_subsample=args.train_subsample,
        softreg_lambda=args.softreg_lambda,
        ig_steps=args.ig_steps,
        xai_method=args.xai_method,
        ood_domains=tuple(args.ood_domains),
        faithfulness_n_texts=args.faithfulness_n_texts,
        device=args.device,
    )
    results = run_full_experiment(cfg)
    save_json(args.out, results)
    print(f"\nSaved results -> {args.out}")
    figs = make_all_figures(results, args.figdir)
    print(f"Rendered {len(figs)} figures -> {args.figdir}/")


if __name__ == "__main__":
    main()
