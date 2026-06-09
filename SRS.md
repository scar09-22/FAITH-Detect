# Software Requirements Specification (SRS)
## FAITH-Detect: Function-word-Agnostic, Faithfully-explained Detection of AI-Generated Reviews

**Version:** 1.0  **Status:** Baseline for implementation  **Author:** (project owner)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for **FAITH-Detect**, a methodologically-corrected,
publishable system for detecting AI-generated reviews with **explainable AI (XAI)**. It is a
from-scratch redesign of a prior project (`Detection+XAI`) whose methodology and result
integrity do not meet a publishable standard (see §3). The defining requirement, set by the
project owner, is that **function words** ("the", "a", "an", …) — high-frequency tokens
common to both human and machine text — must be **excluded from the explanation** and must
**not influence the model's final decision**. FAITH-Detect elevates this requirement into the
central, testable scientific contribution.

### 1.2 Scope
* **Task:** binary classification of a review as **human-written (0)** or **AI-generated (1)**.
* **Language:** English (per project decision).
* **In-domain data:** the English subset of **MAiDE-up** (Ignat, Xu & Mihalcea, *Findings of
  NAACL 2025*) — GPT-4-generated vs. human hotel reviews.
* **Out-of-distribution (OOD) data:** **RAID** (Dugan et al., *ACL 2024*) for cross-domain,
  cross-generator and adversarial-robustness evaluation.
* **Deliverables:** runnable codebase, this SRS, a reproducible local pipeline (Apple-Silicon
  MPS), a Colab T4 notebook for the full experimental grid, real multi-seed results with
  confidence intervals, ablations, statistical tests, and publication-quality figures.

### 1.3 Definitions
| Term | Meaning |
|---|---|
| **Function word** | A closed-class word (article, preposition, conjunction, auxiliary, pronoun, particle, determiner) or standard stop-word. The audited set is defined in `function_words.py`. |
| **Content word** | Any non-function, alphanumeric word. |
| **Hard-Mask variant** | Function-word subword tokens replaced by a single placeholder `[FUNC]` before encoding ⇒ decision provably invariant to function-word identity. |
| **SoftReg variant** | Full text + an attribution-regularisation penalty discouraging saliency on function words. |
| **Faithful explanation** | An attribution produced by running the *actual deployed model* (not a surrogate), validated by quantitative faithfulness metrics. |
| **Grouped split** | Train/val/test partition by hotel so no hotel appears in two folds (prevents content leakage). |

### 1.4 Contributions / Novelty
1. **Function-word-invariant detection** with a *constructive guarantee* (Hard-Mask) and a
   *learned alternative* (SoftReg), directly realising the owner's requirement.
2. **Faithful, content-focused XAI by construction**, replacing post-hoc explanations of a
   surrogate model; validated with comprehensiveness/sufficiency, deletion/insertion AUC and a
   new **function-word attribution-mass** metric.
3. A link between the function-word question and **shortcut learning / OOD generalization**:
   evidence that ignoring function words improves **cross-domain & cross-generator** transfer
   and **adversarial robustness**.
4. A **rigorous evaluation protocol** (leakage-free grouped splits, ≥5 seeds with CIs, McNemar
   & paired-bootstrap significance, classic + zero-shot baselines, full ablations).

---

## 2. Overall Description

### 2.1 Product perspective
FAITH-Detect is a research codebase (Python package `faithdetect` + scripts + Colab notebook).
It reuses the owner's existing MAiDE-up CSV and an existing virtual environment, adding only a
few libraries (Captum, statsmodels, spaCy model).

### 2.2 Constraints & assumptions
* **Compute:** local development on Apple-Silicon **MPS** (8.5 GB RAM, ~22 GB disk) for the
  smoke pipeline; **Colab T4 (CUDA)** for the full grid. No multi-GPU.
* **Data:** MAiDE-up English = 2,000 reviews (small) ⇒ contributions are framed around
  generalization & faithfulness, not leaderboard accuracy; CIs reported honestly.
* **Single generator in-domain:** MAiDE-up is GPT-4 only ⇒ cross-generator evidence comes
  from RAID.

### 2.3 Operating environment
* Python ≥3.11, PyTorch ≥2.2 (MPS or CUDA), HuggingFace Transformers/Datasets, scikit-learn,
  Captum, SHAP, spaCy (`en_core_web_sm`), NLTK, matplotlib/seaborn, statsmodels, umap-learn.

---

## 3. Methodology-Correction Matrix (prior flaws → requirements)

| ID | Flaw in prior `Detection+XAI` | Corrected requirement | Verified by |
|----|-------------------------------|-----------------------|-------------|
| **F1** | XAI fed `dummy_features = zeros`, explaining a *different* model than the one evaluated. | All attributions run the **real deployed model** with the real variant transform (`explain/attributions.py`). | Faithfulness metrics; code review |
| **F2** | Figure 7 hard-coded fabricated highlight phrases (`gen.py`). | Every figure rendered **only** from `results/*.json`; example explanations produced by the model. | `viz/figures.py`, `make_figures.py` |
| **F3** | "Error types" pie chart from toy keyword rules. | Real error analysis: confident-wrong mining + attribution inspection. | `experiment.py`, figures |
| **F4** | Home-grown "LIME" (occlusion + length/position fallback). | Captum **Integrated Gradients** + genuine **leave-one-word-out occlusion** (+ optional SHAP/LIME). | `explain/attributions.py` |
| **F5** | "Perplexity" from a masked-LM loss. | If used, perplexity uses a **causal LM**; interpretable features are an **ablation only**. | `features.py` (optional) |
| **F6** | Reviews truncated to 100–200 *chars*. | Token-level truncation at `max_length` (no char truncation); full review used. | `data/collate.py` |
| **F7** | Raw, unscaled features concatenated. | Encoder-based; any auxiliary features standardised. | models / features |
| **F8** | English-only NLP tools applied to 10 languages. | **English-only** scope; English-appropriate tooling. | scope decision |
| **F9** | Single seed, no CIs, no significance. | **≥5 seeds**, mean ± SD & 95% CI; **McNemar** + **paired bootstrap**. | `utils/stats.py` |
| **F10** | No leakage-free / OOD evaluation. | **Grouped (by-hotel) splits**; explicit **leakage gap**; **RAID** cross-domain/-generator/-attack. | `data/maide_up.py`, `data/raid.py` |
| **F11** | Architecture/feature-dim drift; `strict=False` loads. | **One** clean architecture across all variants. | `models.py` |
| **F12** | No real baselines / ablations. | TF-IDF+LR, content-only LR, zero-shot; full ablation suite. | `baselines.py`, configs |

---

## 4. Functional Requirements

* **FR1 Data.** Load MAiDE-up English; combine Upside+Downside; drop empties; produce
  **grouped** and **random** train/val/test splits (`data/maide_up.py`).
* **FR2 Function-word set.** Build an auditable set from curated closed-class + NLTK/sklearn/
  spaCy stop-words; expose set-definition variants for ablation (`function_words.py`).
* **FR3 Masking.** Map RoBERTa BPE subwords → surface words; mark & hard-mask function-word
  tokens with `[FUNC]`; expose per-token FW mask for SoftReg (`function_words.py`, `collate.py`).
* **FR4 Models.** One encoder+head architecture with three variants: **baseline**, **hardmask**,
  **softreg** (`models.py`).
* **FR5 Training.** Seeded AdamW + linear warmup, early stopping; MPS/CUDA (`train.py`).
* **FR6 Evaluation.** Accuracy, F1 (macro/binary), precision, recall, ROC-AUC, PR-AUC, ECE;
  confusion matrices; per-seed aggregation to CIs (`evaluate.py`, `utils/stats.py`).
* **FR7 Generalization.** Evaluate in-domain (grouped), RAID OOD (cross-domain/-generator) and
  local attacks (function-word, synonym, whitespace) (`data/raid.py`, `data/attacks.py`).
* **FR8 Faithful XAI.** IG + occlusion on the real model; content-only explanations; function
  words excluded from display (`explain/`).
* **FR9 Faithfulness metrics.** Comprehensiveness, sufficiency, deletion/insertion AUC, and
  **function-word attribution mass** (`explain/faithfulness.py`).
* **FR10 Statistics.** Multi-seed CIs; McNemar & paired bootstrap for model comparisons
  (`utils/stats.py`).
* **FR11 Visualization.** ≥14 publication figures rendered solely from results (`viz/figures.py`).
* **FR12 Reproducibility.** One consolidated results JSON capturing config + env + per-seed
  metrics + CIs (`utils/logging.py`).

---

## 5. Non-Functional Requirements

* **NFR1 Reproducibility.** Global seeding (python/numpy/torch + DataLoader workers); configs
  recorded with every run.
* **NFR2 Faithfulness guarantee.** Hard-Mask decision **provably** invariant to function-word
  identity — asserted by automated test (`tests/test_invariance.py`).
* **NFR3 Integrity.** No hard-coded results in any figure or table.
* **NFR4 Portability.** Identical experiment code runs on local MPS and Colab T4; only the
  config differs.
* **NFR5 Resource budget.** Local smoke completes on 8.5 GB RAM / 22 GB disk (RAID streamed &
  capped; models freed between runs).

---

## 6. External Interfaces / Datasets

* **MAiDE-up** — English subset of the owner's `all_data.csv` (columns `Upside_Review`,
  `Downside_Review`, `Review_Language`, `Hotel Name`, `City Name`, `source`).
* **RAID** — `liamdugan/raid` on HuggingFace (`model`, `domain`, `attack`, `generation`).
  Streamed with caps; `abstracts` domain used locally (cheap cross-domain), `reviews`/others on
  Colab.

---

## 7. Experimental Design

* **Seeds:** ≥5 (full); ≥3 (smoke). Report mean ± SD, 95% CI (t and bootstrap).
* **Splits:** grouped primary; random reported to expose leakage.
* **Baselines:** TF-IDF+LR (full vocab), TF-IDF+LR (content-only), zero-shot threshold.
* **Ablations:** {baseline, hardmask, softreg}; mask strategy {replace/remove/attention-zero};
  FW-set {curated/stopwords/union}; +interpretable features; train-data {MAiDE-up, +RAID};
  SoftReg λ sweep; encoder {roberta-base, roberta-large (Colab)}.
* **Axes:** in-domain, cross-domain (RAID), cross-generator (RAID models), robustness (RAID
  attacks + local attacks), calibration.

---

## 8. Acceptance Criteria (publishable bar)

1. Hard-Mask invariance test passes (logits bit-identical under function-word perturbation).
2. **Function-word identity sensitivity = 0** for Hard-Mask (|Δp(AI)| when function words are
   swapped) and **> 0** for baseline; **function-word attribution mass** also reported (≈0 for
   Hard-Mask under IG). The identity-sensitivity metric is forward-only and therefore portable
   across backends; Integrated Gradients (used for attribution mass) runs on GPU/Colab and via
   a portable manual implementation, while the macOS-CPU smoke uses occlusion for faithfulness.
3. Multi-seed CIs reported for every headline number; ≥1 significant comparison (McNemar) shown.
4. Leakage gap (grouped vs. random) quantified.
5. OOD and robustness results show Hard-Mask/SoftReg ≥ baseline on transfer and/or attacks.
6. Faithfulness metrics computed and reported with CIs.
7. Every figure regenerable from `results/*.json`; no fabricated artifacts remain.

---

## 9. Deliverables

`SRS.md` (this doc) · `faithdetect` package · `scripts/` (smoke, figures, single-run) ·
`tests/` (invariance) · `notebooks/colab_full_run.ipynb` · `results/*.json` · `figures/*.png` ·
`paper/outline.md` · `README.md`.
