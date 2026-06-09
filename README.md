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

We show function-word invariance improves **cross-domain / cross-generator** transfer and
**attack robustness**, and yields **faithful, content-only** explanations (validated with
comprehensiveness/sufficiency, deletion/insertion AUC, a **function-word attribution-mass**
metric ≈0 for the hard-masked model, and a forward-only **function-word identity-sensitivity**
metric — |Δp(AI)| when function words are swapped — which is *exactly 0* for Hard-Mask).

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
