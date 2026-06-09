"""Render all publication figures from a results JSON (no recomputation, no hard-coded data)."""
import argparse
import _bootstrap  # noqa: F401

from faithdetect.utils.logging import load_json
from faithdetect.viz import make_all_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/smoke_results.json")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()
    results = load_json(args.results)
    figs = make_all_figures(results, args.figdir)
    print(f"Rendered {len(figs)} figures -> {args.figdir}/")


if __name__ == "__main__":
    main()
