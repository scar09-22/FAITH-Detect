# FAITH-Detect

**F**unction-word-**A**gnostic, fa**ith**fully-explained detection of AI-generated reviews.

A from-scratch, methodologically-corrected reimplementation of an AI-review detector + XAI.
Its defining property: **function words ("the", "a", "an", …) are excluded from the explanation
and cannot affect the model's decision** — turned into a testable scientific contribution.

> Full requirements and the prior-flaw → correction matrix are in [`SRS.md`](SRS.md).
> Paper plan in [`paper/outline.md`](paper/outline.md).

## Why
The predecessor project reported ~94–95% accuracy but: explained a *different* model than it
evaluated (XAI fed zeroed features), shipped a **hard-coded fabricated** explanation figure and
a **fabricated** error-type pie chart, computed "perplexity" with a masked LM, truncated reviews
to 100–200 characters, used a single seed with no confidence intervals, and never tested
generalization. FAITH-Detect fixes all of this (see `SRS.md` §3).

## Core idea
| Variant | Decision uses function words? | Guarantee |
|---|---|---|
| `baseline` | yes | — (shows the shortcut) |
| `hardmask` (FAITH) | **no** — FW tokens replaced by `[FUNC]` before encoding | **provable** invariance to FW identity |
| `softreg` | discouraged | learned (attribution penalty) |

Function-word invariance is **content-only by construction**: a **function-word attribution-mass**
metric (≈0 for Hard-Mask under Integrated Gradients) and a forward-only **identity-sensitivity**
metric (|Δp(AI)| when function words are swapped) make the guarantee measurable. Empirically
(see **Results**) it buys in-domain parity, the best **attack robustness**, and faithful
content-only explanations — with a characterised **trade-off** in cross-domain transfer.

## Results

Full run: **RoBERTa-base, 5 seeds**, grouped (leakage-free) splits on MAiDE-up English;
out-of-distribution = **RAID `reviews`** (cross-domain + 8 generators incl. GPT-3.5/4, Cohere,
Llama). Mean ± 95% CI over seeds. Figures: [`results_colab/figures/`](results_colab/figures);
table via `python scripts/summarize_results.py --results results_colab/results/full_results.json`.

| Metric | Baseline | SoftReg | **Hard-Mask (FAITH)** |
|---|---|---|---|
| In-domain F1 | 93.7 ±2.8 | 93.8 ±3.6 | **93.6 ±1.7** |
| In-domain ROC-AUC | 99.4 | 99.3 | 98.6 |
| **Function-word attribution mass** (IG) ↓ | 36.5% | 30.0% | **0.00%** |
| F1 under **function-word attack** ↑ | 93.7→87.2 | 93.8→87.0 | **93.6→95.8** |
| F1 under **synonym attack** ↑ | →88.4 | →86.4 | **→89.9** |
| Cross-domain OOD F1 (RAID reviews) | **69.5** | 68.2 | 47.4 |

**What holds (the wins).**
- **No in-domain cost.** All three variants are statistically tied (McNemar baseline-vs-Hard-Mask
  p = 0.15); Hard-Mask even has the tightest CI. Function-word invariance is *free* in-domain.
- **Provable + measurable invariance.** Hard-Mask puts **0.00%** attribution on function words
  (vs 36.5% baseline); swapping function words leaves its decision unchanged.
- **Best attack robustness.** Under a function-word attack the baseline drops ~6.5 pts while
  Hard-Mask is *unmoved* (slightly up); it's also most robust to synonym swaps.
- **Leakage is real.** A naive random split inflates baseline F1 to **96.7%** vs **93.2%** grouped
  — the prior project's ~95% was this inflated number ([`02_leakage_gap.png`](results_colab/figures/02_leakage_gap.png)).

**The honest trade-off.** On the cross-domain OOD (hotel → **movie** reviews), Hard-Mask transfers
*worse* (47.4% vs baseline 69.5%, non-overlapping CIs). When the content vocabulary shifts, the
content words Hard-Mask relies on (room, staff, breakfast) vanish, while the baseline's
function-word/style cues carry domain-general signal. So function-word invariance trades a slice of
cross-domain transfer for its guarantee + robustness. *Caveat:* this OOD conflates cross-domain and
cross-generator and is an extreme content shift — a same-domain/different-generator test (see the
README's next-steps) would isolate the cross-generator question more cleanly.

### XAI backends
Attributions run on the **real** model (never a surrogate). Two methods are provided:
- **Integrated Gradients** — a portable manual implementation (Riemann sum over single
  backward passes); used on Colab GPU. (Captum's batched IG can SIGBUS on macOS CPU, so it is
  avoided there.)
- **Occlusion** — leave-one-word-out, forward-only; the robust default for the local
  macOS-CPU smoke. The **identity-sensitivity** metric is forward-only and runs on any backend.

## Layout
```
src/faithdetect/        package: data/, function_words.py, models.py, train.py, evaluate.py,
                        explain/ (attributions, faithfulness), viz/figures.py, experiment.py
scripts/                smoke_test.py, run_experiment.py, make_figures.py
tests/                  test_invariance.py  (asserts the hard-mask guarantee)
configs/                base.yaml, colab_full.yaml
notebooks/              colab_full_run.ipynb  (T4 full grid)
results/  figures/      outputs (rendered only from results JSON)
```

## Setup
This project reuses the existing env at `/Users/shiva/Detection+XAI/roberta_env` (torch+MPS,
transformers, datasets, shap, lime, spacy, sklearn, scipy, seaborn, nltk) plus `captum`,
`statsmodels`, `umap-learn`, and the spaCy English model. To recreate elsewhere:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(x) for x in ['stopwords','punkt','wordnet','omw-1.4']]"
```

## Run
```bash
PY=/Users/shiva/Detection+XAI/roberta_env/bin/python

# 1) correctness checks (function-word masking + hard-mask invariance)
$PY tests/test_invariance.py

# 2) genuine local smoke run — PROCESS-ISOLATED grid (recommended on memory-limited Macs).
#    Each (variant, seed) runs in its own short-lived subprocess on MPS, with retries, then
#    results are merged and all figures rendered. This is robust to the 8 GB unified-memory
#    ceiling and to a macOS Accelerate quirk that crashes attribution passes on CPU.
$PY scripts/run_grid.py --device mps --seeds 0 1 2 --variants baseline hardmask softreg \
    --epochs 3 --train_subsample 600 --xai_method ig

#    (single-process alternative, fine on a GPU box / Colab):
#    $PY scripts/smoke_test.py --seeds 0 1 2 --epochs 3

# 3) (re)render figures from a results file, and print a results table
$PY scripts/make_figures.py --results results/smoke_results.json --figdir figures
$PY scripts/summarize_results.py --results results/smoke_results.json

# 4) a config-driven run (e.g. ablation)
$PY scripts/run_experiment.py --config configs/base.yaml
```

The **full grid** (5 seeds, more epochs, full RAID reviews/cross-generator/attacks, optional
roberta-large) runs on Colab T4 via `notebooks/colab_full_run.ipynb` — same `run_full_experiment`
code, just a heavier config, so the local smoke is a faithful preview.

## Datasets
- **MAiDE-up** (Ignat, Xu & Mihalcea, Findings of NAACL 2025) — English hotel reviews, in-domain.
- **RAID** (Dugan et al., ACL 2024, `liamdugan/raid`) — cross-domain/-generator/-attack OOD.

## Reproducibility & integrity
Every figure is rendered **only** from `results/*.json` (no hard-coded data). Seeds, configs and
environment are recorded in each results file. The central guarantee is asserted in CI-style by
`tests/test_invariance.py`.
