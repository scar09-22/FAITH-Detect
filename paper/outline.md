# Paper outline — FAITH-Detect

**Working title:** *Function Words Are a Shortcut: Faithful, Function-word-Invariant
Detection of AI-Generated Reviews*

**Target venues:** ACL/EMNLP/NAACL (Findings or main); journal options: *Information Processing
& Management*, *Computational Linguistics*, *IEEE TASLP*, *Knowledge-Based Systems*.

---

## Abstract (claims to support with figures/tables)
AI-text detectors trained on a single domain/generator overfit to superficial cues; we show
that **function words** are a major such cue and that **removing them from the decision**
yields detectors that (i) match in-domain accuracy, (ii) generalize better across domains and
generators, (iii) are more robust to function-word attacks, and (iv) admit **faithful,
content-only explanations** by construction.

## 1. Introduction
- Problem: review platforms face AI-generated reviews; detection + trustworthy explanations.
- Gap: prior XAI-for-detection explains a *surrogate*; nobody enforces/measures that decisions
  ignore function words shared by human & machine text.
- Contributions (→ §1.4 of SRS): hard-mask guarantee, soft-reg alternative, faithfulness +
  FW-mass metric, generalization/robustness evidence, rigorous protocol.

## 2. Related work
AI-text detection (zero-shot: GLTR/DetectGPT; supervised: RoBERTa detectors; RAID benchmark);
shortcut learning & spurious correlations; XAI faithfulness (ERASER, IG, deletion/insertion);
"right for the right reasons" attribution regularization; review deception (MAiDE-up).

## 3. Method
- 3.1 Function-word set (auditable; curated ∪ stop-word lists). **[no figure]**
- 3.2 Hard-Mask variant + invariance proof sketch. **[Tab: invariance test result]**
- 3.3 SoftReg attribution-regularization loss.
- 3.4 Faithful explainer (IG/occlusion on the real model) + FW-attribution-mass metric.

## 4. Experimental setup
- Data: MAiDE-up English (grouped splits), RAID OOD. **[Fig 01 dataset_overview]**
- Baselines, seeds/CIs, significance tests, ablations (SRS §7).

## 5. Results
| Claim | Figure/Table |
|---|---|
| Leakage inflates naive scores | **Fig 02 leakage_gap** |
| In-domain parity vs. baseline & classic baselines | **Fig 03 indomain_performance**, **Fig 04 confusion**, **Fig 05 roc_pr** |
| Better cross-domain/-generator transfer | **Fig 06 ood_transfer** |
| Robustness to function-word & paraphrase attacks | **Fig 07 robustness** |
| **Function-word identity cannot move the decision (core claim)** | **Fig 08b fw_identity_sensitivity** |
| Function words carry ~0 attribution (IG) | **Fig 08 fw_attribution_mass** |
| Explanations are faithful | **Fig 09 faithfulness**, **Fig 10 deletion_insertion_curves** |
| Calibration | **Fig 11 calibration** |
| Training behavior | **Fig 12 learning_curves** |
| Representation structure | **Fig 13 embedding_projection** |
| Qualitative content-only explanations | **Fig 14 example_explanation** |

- Significance: McNemar (baseline vs hard-mask), paired bootstrap on OOD F1.
- Ablations: FW-set definition; mask strategy; SoftReg λ; +features; +RAID training; encoder size.

## 6. Discussion
Why function words are a shortcut (domain/generator artifacts); trade-offs of hard vs soft
invariance; limitations (English-only, small in-domain set, single in-domain generator).

## 7. Conclusion
Function-word invariance is a cheap, constructive route to detectors that are simultaneously
**accurate, transferable, robust, and faithfully explainable**.

## Reproducibility statement
All numbers from `results/*.json`; figures via `scripts/make_figures.py`; seeds/configs logged;
invariance asserted by `tests/test_invariance.py`.
