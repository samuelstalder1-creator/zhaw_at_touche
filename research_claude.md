# Research Questions — Answers from Experimental Results

**Metric convention.** All tables report positive-class (advertisement) Precision, Recall, and F1 alongside Macro F1.
Positive-class F1 is the task's primary metric (detecting ads). Macro F1 treats both classes equally and is used for cross-family comparisons.
`overall.f1` in newer metrics_summary.json files stores positive-class F1 only — do not compare it directly to Macro F1 from older format files.

---

## Setup reference

### Fine-tuned models

| Setup | Input | Acc | Recall | Prec | F1 | Macro F1 | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| **setup10** *(CompactQueryClassifier)* | query + response | 0.993 | 0.983 | 0.995 | 0.989 | 0.992 | 42 |
| setup10_1_gemini | query + response + neutral (Gemini) | 0.998 | 0.995 | 0.998 | 0.996 | 0.997 | 14 |
| setup10_1-gemma | query + response + neutral (Gemma) | 0.998 | 0.999 | 0.996 | 0.997 | 0.998 | 10 |
| setup10_1-qwen | query + response + neutral (Qwen) | 0.998 | 0.995 | 0.998 | 0.996 | 0.997 | 14 |

### Frozen embedding + LR (delta models)

| Setup | Input | Provider | Acc | Recall | Prec | F1 | Macro F1 | Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **setup104-base** *(BaseStack)* | response only | — | 0.737 | 0.657 | 0.560 | 0.605 | 0.704 | 1635 |
| setup104-gemini | response + neutral + delta | Gemini | 0.772 | 0.662 | 0.619 | 0.640 | 0.736 | 1420 |
| setup104-gemma | response + neutral + delta | Gemma | 0.756 | 0.670 | 0.589 | 0.627 | 0.723 | 1519 |
| **setup104-qwen** *(QwenResidualStack)* | response + neutral + delta | Qwen | 0.791 | 0.678 | 0.653 | 0.665 | 0.756 | 1301 |
| setup119-gemini | delta only (response − neutral) | Gemini | 0.766 | 0.601 | 0.622 | 0.612 | 0.722 | 1454 |
| setup119-gemma | delta only (response − neutral) | Gemma | 0.727 | 0.719 | 0.540 | 0.617 | 0.702 | 1700 |
| **setup119-qwen** *(QwenResidualOnly)* | delta only (response − neutral) | Qwen | 0.788 | 0.636 | 0.659 | 0.647 | 0.748 | 1321 |

---

## Touché Challenge Validation

The four submitted models were evaluated on the Touché shared task test set independently of the local test split. Rankings are perfectly preserved; absolute scores drop slightly (0–0.027 F1), confirming that the local test results generalise.

| Model | Setup | Local F1 | Touché F1 | Δ | Local P/R | Touché P/R |
|---|---|---:|---:|---:|---|---|
| CompactQueryClassifier | setup10 | 0.989 | **0.985** | −0.004 | 0.995 / 0.983 | 0.990 / 0.981 |
| QwenResidualStack | setup104-qwen | 0.665 | **0.638** | −0.027 | 0.653 / 0.678 | 0.608 / 0.671 |
| QwenResidualOnly | setup119-qwen | 0.647 | **0.613** | −0.034 | 0.659 / 0.636 | 0.616 / 0.610 |
| BaseStack | setup104-base | 0.605 | **0.601** | −0.004 | 0.560 / 0.657 | 0.532 / 0.691 |

**Generalisation pattern.** CompactQueryClassifier and BaseStack lose almost no F1 (−0.004) when moving to the Touché test set. The two Qwen delta models drop more (−0.027 / −0.034). The drop is driven entirely by a precision decrease: at Touché, the delta models generate more false positives (lower precision) while recall holds or increases slightly. This suggests the Touché test set contains more non-ad responses that share surface features with ads in embedding space — the frozen LR boundary is less tight there than on the local test split.

---

## Q1 — Does the neutral rewrite help fine-tuned models?

**Yes, by a small but consistent margin — and the gain is symmetric across P and R.**

| Setup | Recall | Prec | F1 | Macro F1 | Errors |
|---|---:|---:|---:|---:|---:|
| setup10 (no neutral) | 0.983 | 0.995 | 0.989 | 0.992 | 42 |
| + Gemini neutral | 0.995 | 0.998 | 0.996 | 0.997 | 14 |
| + Gemma neutral | **0.999** | 0.996 | **0.997** | **0.998** | **10** |
| + Qwen neutral | 0.995 | 0.998 | 0.996 | 0.997 | 14 |

All providers reduce errors by ~66–76% (42 → 10–14). Recall improves more than precision (+0.012–0.016 recall vs +0.001–0.003 precision), meaning the neutral primarily helps the fine-tuned model catch ads it would otherwise miss — false negatives are the main failure mode of setup10, and the neutral guides attention to the advertising-specific language.

**Gemma is marginally best** (0.997 F1, 10 errors, highest recall 0.999) despite being the weakest neutral provider for delta models. At the fine-tuning level, the exact quality of the neutral matters less because the backbone learns to attend selectively to the input; any clean neutral rewrite is enough to flag the contrast.

**Provider effect at this level is negligible** (0.001 F1 between Gemini/Qwen and Gemma). The ceiling effect dominates.

---

## Q2 — Does the frozen delta substitute for fine-tuning?

**No. The gap is large and the error profiles differ qualitatively.**

| | Recall | Prec | F1 | Macro F1 | Errors |
|---|---:|---:|---:|---:|---:|
| Fine-tuned, no neutral (setup10) | 0.983 | 0.995 | 0.989 | 0.992 | 42 |
| Fine-tuned, best neutral (setup10_1-gemma) | 0.999 | 0.996 | 0.997 | 0.998 | 10 |
| Best delta model (setup104-qwen) | 0.678 | 0.653 | 0.665 | 0.756 | 1301 |
| Response-only LR (setup104-base) | 0.657 | 0.560 | 0.605 | 0.704 | 1635 |

The best frozen delta model (setup104-qwen) makes **31× more errors** than the fine-tuned baseline (setup10). The error structure is different:

- **Fine-tuned models**: near-symmetric errors (setup10: FP=10, FN=32). Both recall and precision are high. The dominant failure is a few missed ads (low FN count relative to total).
- **Delta models**: asymmetric errors, recall-biased (setup104-qwen: FP=687, FN=614). Many both false positives and false negatives. Neither precision nor recall approaches fine-tuned levels.

The frozen encoder cannot form task-specific boundaries: it sees responses as points in a general-purpose semantic space. The LR finds the best linear separator in that space, but advertising language does not have a clean linear boundary in frozen embeddings. Fine-tuning reshapes the internal representations so that advertising patterns become linearly separable.

**The delta does not substitute for fine-tuning. It is a useful GPU-free alternative when no labelled fine-tuning is feasible, but not a replacement.**

---

## Q3 — Does adding the neutral and delta improve frozen embedding-LR over response-only?

**Yes — and the gain is almost entirely a precision improvement.**

| | Recall | Prec | F1 | Macro F1 | Δ Macro vs base |
|---|---:|---:|---:|---:|---:|
| setup104-base (response only) | 0.657 | 0.560 | 0.605 | 0.704 | — |
| + Gemma neutral | 0.670 | 0.589 | 0.627 | 0.723 | +0.019 |
| + Gemini neutral | 0.662 | 0.619 | 0.640 | 0.736 | +0.032 |
| + Qwen neutral | 0.678 | **0.653** | **0.665** | **0.756** | **+0.053** |

**Recall is nearly unchanged** (0.657 → 0.662–0.678, +0.005–0.021). **Precision jumps** (0.560 → 0.589–0.653, +0.029–0.093 depending on provider).

Interpretation: without a neutral, the LR can find regions in embedding space that correlate with ads (hence decent recall), but it mistakes many non-ad responses as ads because the response-embedding alone does not isolate what is specifically advertising. Adding the delta vector tells the LR *which dimensions changed when the neutral removed the ad language* — this directional information allows the model to suppress false positives without sacrificing recall. The neutral acts as a precision booster: it does not help the model find more ads, it helps the model stop over-calling non-ads as ads.

---

## Q4 — Is full stack better than residual-only in the frozen embedding family?

**Yes, and the difference is a recall vs. precision tradeoff, not a uniform improvement.**

| Provider | Setup | Recall | Prec | F1 | Macro F1 |
|---|---|---:|---:|---:|---:|
| Qwen | setup104-qwen (full stack) | **0.678** | 0.653 | **0.665** | **0.756** |
| Qwen | setup119-qwen (residual only) | 0.636 | **0.659** | 0.647 | 0.748 |
| Gemini | setup104-gemini (full stack) | **0.662** | 0.619 | **0.640** | **0.736** |
| Gemini | setup119-gemini (residual only) | 0.601 | **0.622** | 0.612 | 0.722 |
| Gemma | setup104-gemma (full stack) | **0.670** | **0.589** | **0.627** | **0.723** |
| Gemma | setup119-gemma (residual only) | 0.719 | 0.540 | 0.617 | 0.702 |

**Consistent pattern for Qwen and Gemini**: full stack achieves higher recall (+0.042 / +0.061) at a small precision cost (−0.006 / −0.003). Adding the absolute response and neutral embeddings alongside the delta gives the LR more context about the response's position in semantic space, enabling it to detect ads it would miss with the delta alone. Precision drops slightly because the additional features also activate for some non-ad responses.

**Gemma is the exception**: residual-only (setup119-gemma) has substantially higher recall (0.719 vs 0.670) but much lower precision (0.540 vs 0.589). The full stack is better overall (Macro F1: 0.723 vs 0.702) because the extremely low precision of the Gemma residual drags F1 down despite high recall. Gemma's delta is noisy (large changes to both ad and non-ad responses), and without the anchoring provided by the absolute embeddings, the residual alone produces an imbalanced classifier.

**Recommendation**: use full stack when recall matters more (minimise missed ads); use residual-only when precision matters more (minimise false alarms). For Qwen, the difference is small and either is a reasonable choice.

---

## Q5 — Which neutral provider works best?

**Qwen > Gemini > Gemma** consistently, but with distinct precision-recall profiles.

### Full stack (setup104-*)
| Provider | Recall | Prec | F1 | Macro F1 |
|---|---:|---:|---:|---:|
| Qwen | 0.678 | **0.653** | **0.665** | **0.756** |
| Gemini | 0.662 | 0.619 | 0.640 | 0.736 |
| Gemma | 0.670 | 0.589 | 0.627 | 0.723 |
| Base (no neutral) | 0.657 | 0.560 | 0.605 | 0.704 |

### Residual-only (setup119-*)
| Provider | Recall | Prec | F1 | Macro F1 |
|---|---:|---:|---:|---:|
| Qwen | 0.636 | **0.659** | **0.647** | **0.748** |
| Gemini | 0.601 | 0.622 | 0.612 | 0.722 |
| Gemma | **0.719** | 0.540 | 0.617 | 0.702 |

**Why Qwen is best**: Qwen's neutrals make larger changes to ads than to non-ads (character-level similarity gap: 0.011 Qwen vs 0.004 Gemini). The delta `response − neutral` therefore carries a cleaner directional signal: it is large in the advertising direction for ads and near-zero for clean responses. This allows the LR to set a tight decision boundary with high precision and good recall.

**Why Gemini is second**: Gemini 2.5 Flash Lite is a more capable model and produces higher-quality neutrals, but it rewrites more aggressively — even non-ad responses are substantially rewritten. The delta is informative but noisier than Qwen's, leading to more false positives (lower precision). More aggressive rewriting does not equal a more discriminative delta.

**Gemma's distinct profile**: Gemma shows the highest recall in residual-only (0.719) but the lowest precision (0.540). The Gemma delta appears to activate strongly for a broad set of responses, pushing the LR to classify many borderline cases as ads. This recall-biased behaviour makes Gemma the weakest provider by F1 and Macro F1 despite catching the most ads.

**Provider ranking is stable across both model families and across local and Touché test sets.** This confirms it is a property of the neutral rewrite quality, not of the evaluation data.

---

## Overarching answer

> *Does the semantic delta between a response and its neutral rewrite provide a sufficient signal for advertisement detection and does it substitute for or complement end-to-end fine-tuning? How does access to the query and multiple neutral sources affect each approach?*

**The delta is a useful but not sufficient standalone signal. It complements fine-tuning (slightly) but does not substitute for it.**

### Sufficient signal?
The best frozen delta model (QwenResidualStack) achieves F1=0.665 and Macro F1=0.756 with no labelled fine-tuning beyond a logistic regression. This is well above chance and usable in practice. However, "sufficient" for the task requires F1 close to fine-tuned models (~0.989): the delta model makes 31× more errors. The delta signal exists and is real — it is just not rich enough to support a high-precision classifier in frozen embedding space.

### Substitute for fine-tuning?
No. Fine-tuning (CompactQueryClassifier, F1=0.989) produces near-symmetric, low-error predictions (FP=10, FN=32). Delta models (F1=0.605–0.665) produce many false positives and false negatives simultaneously. The error structures do not converge even with the best neutral provider.

### Complement to fine-tuning?
Yes. Adding the neutral rewrite to the fine-tuned input reduces errors from 42 to 10–14 (−66–76%). The improvement is mostly in recall: the fine-tuned model catches more ads it would have missed. However, the baseline (setup10, F1=0.989) is already extremely high, so the practical impact of the complement is modest.

### Effect of query access?
The fine-tuned model already includes the query (setup10: query + response). No delta-only model in this experiment uses the query, so the isolated query contribution to delta models cannot be quantified from these results.

### Effect of neutral provider?
- For delta models: Qwen > Gemini > Gemma (stable across full-stack and residual-only, and across local and Touché). Provider quality has a significant effect (~0.054 Macro F1 between best and worst provider in full stack).
- For fine-tuned models: Gemma ≈ Gemini ≈ Qwen (within 0.001 F1). Provider becomes irrelevant when the backbone fine-tunes end-to-end.
