"""Print a clean markdown summary (tables) from a results JSON — for paper writing & review."""
import argparse
import _bootstrap  # noqa: F401

from faithdetect.utils.logging import load_json
from faithdetect.utils.stats import fmt_mean_ci

VARIANTS = ["baseline", "softreg", "hardmask"]
LBL = {"baseline": "Baseline", "softreg": "SoftReg", "hardmask": "Hard-Mask (FAITH)"}


def _cell(agg, metric):
    it = agg.get(metric)
    return fmt_mean_ci(it) if it else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/smoke_results.json")
    args = ap.parse_args()
    r = load_json(args.results)
    meta = r.get("meta", {}).get("config", {})

    print(f"# FAITH-Detect results — '{r.get('meta',{}).get('config',{}).get('name','?')}'\n")
    print(f"- encoder: `{meta.get('encoder_name')}`  device: `{r.get('meta',{}).get('env',{}).get('torch','')}`")
    print(f"- seeds: {meta.get('seeds')}  epochs: {meta.get('epochs')}  "
          f"batch: {meta.get('batch_size')}  max_len: {meta.get('max_length')}  "
          f"train_subsample: {meta.get('train_subsample')}")
    print(f"- function-word set: {r.get('meta',{}).get('fw_set_size')} words "
          f"({', '.join(r.get('meta',{}).get('fw_sources',[]))})\n")

    sd = r.get("split_describe", {})
    if sd:
        print(f"**Split** ({sd.get('mode')}): train {sd['train']['n']}, val {sd['val']['n']}, "
              f"test {sd['test']['n']}; hotel leakage train↔test = {sd['hotel_leakage_train_test']}\n")

    # In-domain + OOD table
    print("## In-domain (grouped) and OOD (RAID) — mean ± 95% CI over seeds\n")
    print("| Variant | In-domain F1 | In-domain ROC-AUC | OOD F1 | Held-out-gen F1 |")
    print("|---|---|---|---|---|")
    for v in VARIANTS:
        if v not in r.get("variants", {}):
            continue
        ind = r["variants"][v].get("indomain", {}).get("aggregated", {})
        ood = r["variants"][v].get("ood", {}).get("aggregated", {})
        ho = r["variants"][v].get("heldout_gen", {}).get("aggregated", {})
        print(f"| {LBL[v]} | {_cell(ind,'f1_macro')} | {_cell(ind,'roc_auc')} | "
              f"{_cell(ood,'f1_macro') if ood else '—'} | {_cell(ho,'f1_macro') if ho else '—'} |")
    hp = r.get("heldout_per_generator", {})
    if hp:
        gens = sorted(set().union(*[set(d) for d in hp.values()]))
        print("\n## Held-out generators (same domain, mixed-generator training) — ref-seed F1\n")
        print("| Generator | " + " | ".join(LBL[v] for v in VARIANTS if v in hp) + " |")
        print("|---|" + "---|" * len([v for v in VARIANTS if v in hp]))
        for g in gens:
            cells = [f"{hp[v].get(g, float('nan'))*100:.1f}%" for v in VARIANTS if v in hp]
            print(f"| {g} | " + " | ".join(cells) + " |")
    for bname, b in r.get("baselines", {}).items():
        m = b["metrics"]; om = b.get("ood_metrics", {})
        print(f"| {bname} | {m['f1_macro']*100:.1f}% | {m.get('roc_auc',0)*100:.1f}% | "
              f"{om.get('f1_macro',0)*100:.1f}% |" if om else
              f"| {bname} | {m['f1_macro']*100:.1f}% | {m.get('roc_auc',0)*100:.1f}% | — |")

    # Function-word attribution mass (the core metric)
    print("\n## Function-word attribution mass (IG) — lower = less reliance on function words\n")
    print("| Variant | FW-mass (mean) |")
    print("|---|---|")
    for v in VARIANTS:
        d = r.get("fw_mass", {}).get(v)
        if d:
            print(f"| {LBL[v]} | {d['mean']*100:.2f}% |")

    # Robustness
    attacks = list(r.get("variants", {}).get(VARIANTS[0], {}).get("attacks", {}).keys())
    if attacks:
        print("\n## Robustness — F1 (macro) under attack (mean over seeds)\n")
        print("| Variant | clean | " + " | ".join(a.replace("_", " ") for a in attacks) + " |")
        print("|---|" + "---|" * (len(attacks) + 1))
        for v in VARIANTS:
            if v not in r.get("variants", {}):
                continue
            clean = r["variants"][v]["indomain"]["aggregated"].get("f1_macro", {}).get("mean", 0)
            row = [f"{clean*100:.1f}%"]
            for a in attacks:
                m = r["variants"][v]["attacks"][a]["aggregated"].get("f1_macro", {}).get("mean", 0)
                row.append(f"{m*100:.1f}%")
            print(f"| {LBL[v]} | " + " | ".join(row) + " |")

    # Faithfulness
    print("\n## Explanation faithfulness (IG, content words)\n")
    print("| Variant | Comprehensiveness↑ | Sufficiency↓ | Deletion-AUC↓ | Insertion-AUC↑ |")
    print("|---|---|---|---|---|")
    for v in VARIANTS:
        f = r.get("variants", {}).get(v, {}).get("faithfulness", {}).get("summary")
        if f:
            print(f"| {LBL[v]} | {f.get('comprehensiveness',0):.3f} | {f.get('sufficiency',0):.3f} | "
                  f"{f.get('deletion_auc',0):.3f} | {f.get('insertion_auc',0):.3f} |")

    # Leakage + significance
    lk = r.get("leakage", {})
    if lk:
        print("\n## Leakage gap (baseline)\n")
        for mode in ("grouped", "random"):
            if mode in lk:
                print(f"- {mode}: F1={lk[mode].get('f1_macro',0)*100:.1f}%  "
                      f"acc={lk[mode].get('accuracy',0)*100:.1f}%  "
                      f"ROC-AUC={lk[mode].get('roc_auc',0)*100:.1f}%")
    sig = r.get("significance", {})
    for k, v in sig.items():
        print(f"\n**Significance** ({k}): McNemar {v.get('method')}, p = {v.get('pvalue'):.4f} "
              f"(A_right_B_wrong={v.get('n_a_right_b_wrong')}, A_wrong_B_right={v.get('n_a_wrong_b_right')})")


if __name__ == "__main__":
    main()
