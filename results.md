# Results Overview

All metrics are on the `test` split. Confusion matrix columns: TN / FP / FN / TP (rows = gold, cols = predicted).

## Result Table

| Setup | Family | Eval source | Macro F1 | TN | FP | FN | TP | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `setup115` | classifier | Gemini | 0.9987 | 4315 | 1 | 6 | 1898 | best committed overall classifier; response-only |
| `setup6-qwen` | classifier | Qwen | 0.9985 | 4313 | 3 | 5 | 1899 | best committed Qwen-backed classifier |
| `setup12` | classifier | Gemini | 0.9977 | 4312 | 4 | 8 | 1896 | best committed Gemini-backed query-aware classifier |
| `setup6` | classifier | Gemini | 0.9975 | 4305 | 11 | 2 | 1902 | strongest Gemini recall |
| `setup105_1` | cross-encoder | Gemini | 0.9975 | 4315 | 1 | 12 | 1892 | stable RoBERTa cross-encoder retry |
| `setup7-qwen` | classifier | Qwen | 0.9964 | 4311 | 5 | 14 | 1890 | best committed prompted-neutral classifier |
| `setup116` | classifier | Gemini + Qwen | 0.9962 | 4312 | 4 | 16 | 1888 | best committed dual-neutral prompted classifier |
| `setup7` | classifier | Gemini | 0.9953 | 4298 | 18 | 7 | 1897 | Gemini neutral in prompt; below `setup7-qwen` and `setup116` |
| `setup10` | classifier | Gemini | 0.9920 | 4306 | 10 | 32 | 1872 | strong compact baseline (ALBERT) |
| `setup104` | learned embedding feature | Gemini | 0.9915 | 4306 | 10 | 35 | 1869 | best committed single-neutral embedding feature |
| `setup103` | learned embedding feature | Gemini | 0.9913 | 4293 | 23 | 23 | 1881 | residual-only Gemini baseline |
| `setup113` | learned embedding feature | Gemini + Qwen | 0.9857 | 4260 | 56 | 20 | 1884 | committed dual-neutral residual |
| `setup106` | descriptor-only sentence delta | Gemini | 0.9706 | 4196 | 120 | 37 | 1867 | historical artifact; backend not currently wired |
| `setup114` | learned embedding feature | Gemini + Qwen | 0.7527 | 3599 | 717 | 610 | 1294 | full dual-provider embedding stack |
| `setup119` | learned embedding feature | Qwen | 0.7475 | 3689 | 627 | 694 | 1210 | Qwen-only residual counterpart to `setup103` |
| `setup118` | learned embedding feature | Gemini + Qwen | 0.7443 | 3733 | 583 | 737 | 1167 | query + dual residual vectors |
| `setup117` | learned embedding feature | Gemini | 0.7224 | 3557 | 759 | 717 | 1187 | query + Gemini residual vector |
| `setup111` | scalar anchor baseline | Gemini + Qwen | 0.5669 | 3438 | 878 | 1269 | 635 | handcrafted `response_drift - anchor_cohesion` control |
| `setup110` | scalar anchor baseline | Gemini + Qwen | 0.5653 | 2729 | 1587 | 907 | 997 | learned weights over the same six scalars |
| `setup100` | embedding divergence | Gemini | 0.4532 | 2755 | 1561 | 1396 | 508 | balanced drift baseline, not competitive |
| `setup102` | embedding divergence | Gemini | 0.3533 | 582 | 3734 | 144 | 1760 | larger encoder, same failure mode as setup101 |
| `setup101` | embedding divergence | Gemini | 0.3509 | 559 | 3757 | 125 | 1779 | overpredicts positives |
| `setup105` | cross-encoder | Gemini | 0.2344 | 0 | 4316 | 0 | 1904 | collapsed to all-positive; DeBERTa instability + missing eval config |
| `setup8` | classifier | Gemini | 0.2344 | 0 | 4316 | 0 | 1904 | collapsed to all-positive predictions |

---

## Results by Research Question

Each sub-question groups only the setups directly relevant to answering it. Confusion matrix format: **TN / FP / FN / TP**.

---

### RQ1 — Is the semantic delta a sufficient signal?

Compares the full 768-dim delta vector against scalar summarisations of the same signal.

| Setup | Input | Macro F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|
| `setup103` | `response_emb − gemini_emb` | 0.9913 | 4293 | 23 | 23 | 1881 |
| `setup104` | `[response_emb \| gemini_emb \| delta]` | 0.9915 | 4306 | 10 | 35 | 1869 |
| `setup110` | 6 cosine scalars (learned LR) | 0.5653 | 2729 | 1587 | 907 | 997 |
| `setup111` | 6 cosine scalars (handcrafted) | 0.5669 | 3438 | 878 | 1269 | 635 |
| `setup100` | cosine threshold (balanced) | 0.4532 | 2755 | 1561 | 1396 | 508 |
| `setup101` | cosine threshold (overpredicts pos.) | 0.3509 | 559 | 3757 | 125 | 1779 |
| `setup102` | cosine threshold (larger encoder) | 0.3533 | 582 | 3734 | 144 | 1760 |

**Finding**: The 768-dim delta vector (setup103, 0.9913) is sufficient; scalar bottlenecks (setup100–102: 0.35–0.45; setup110–111: ~0.57) discard the directional structure that makes the delta work.

---

### RQ2 — Does the delta substitute for fine-tuning?

Compares the best delta-LR setups against the best plain fine-tuned classifiers (no neutral).

| Setup | Approach | Input | Macro F1 | TN | FP | FN | TP |
|---|---|---|---:|---:|---:|---:|---:|
| `setup115` | fine-tuned | response only | 0.9987 | 4315 | 1 | 6 | 1898 |
| `setup12` | fine-tuned | query + response | 0.9977 | 4312 | 4 | 8 | 1896 |
| `setup6` | fine-tuned | query + response | 0.9975 | 4305 | 11 | 2 | 1902 |
| `setup104` | delta-LR | `[response_emb \| gemini_emb \| delta]` | 0.9915 | 4306 | 10 | 35 | 1869 |
| `setup103` | delta-LR | `response_emb − gemini_emb` | 0.9913 | 4293 | 23 | 23 | 1881 |
| `setup113` | delta-LR | `[delta_gemini \| delta_qwen]` | 0.9857 | 4260 | 56 | 20 | 1884 |

**Finding**: Fine-tuned classifiers consistently outperform delta-LR by ~0.007 Macro F1 and 17–39 fewer errors. The delta does not substitute for fine-tuning, but is a strong GPU-free alternative.

---

### RQ3 — Does the delta complement fine-tuning?

Compares neutral-aware fine-tuned models against the best plain fine-tuned classifiers.

| Setup | Architecture | Input | Macro F1 | TN | FP | FN | TP |
|---|---|---|---:|---:|---:|---:|---:|
| `setup115` | classifier | response only | **0.9987** | 4315 | 1 | 6 | 1898 |
| `setup12` | classifier | query + response | 0.9977 | 4312 | 4 | 8 | 1896 |
| `setup6` | classifier | query + response | 0.9975 | 4305 | 11 | 2 | 1902 |
| `setup105_1` | cross-encoder | response + Gemini neutral | 0.9975 | 4315 | 1 | 12 | 1892 |
| `setup7-qwen` | classifier | query + Qwen neutral + response | 0.9964 | 4311 | 5 | 14 | 1890 |
| `setup116` | classifier | query + Gemini neutral + Qwen neutral + response | 0.9962 | 4312 | 4 | 16 | 1888 |
| `setup7` | classifier | query + Gemini neutral + response | 0.9953 | 4298 | 18 | 7 | 1897 |

**Finding**: No neutral-aware fine-tuned model surpasses the response-only baseline (setup115, 0.9987). The cross-encoder (setup105_1) ties setup6 but does not beat setup115. The neutral does not complement fine-tuning — the response alone already captures what the neutral would expose.

---

### RQ4 — How does query access affect each approach?

#### Fine-tuned classifiers — query vs no query

| Setup | Input | Macro F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|
| `setup115` | response only | **0.9987** | 4315 | 1 | 6 | 1898 |
| `setup12` | query + response | 0.9977 | 4312 | 4 | 8 | 1896 |
| `setup6` | query + response | 0.9975 | 4305 | 11 | 2 | 1902 |

**Finding**: Response alone outperforms every query-aware classifier. The query adds redundant or slightly noisy signal.

#### Delta-LR — query vs no query

| Setup | Input | Macro F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|
| `setup103` | `delta_gemini` | 0.9913 | 4293 | 23 | 23 | 1881 |
| `setup117` | `[query_emb \| delta_gemini]` | 0.7224 | 3557 | 759 | 717 | 1187 |
| `setup113` | `[delta_gemini \| delta_qwen]` | 0.9857 | 4260 | 56 | 20 | 1884 |
| `setup118` | `[query_emb \| delta_gemini \| delta_qwen]` | 0.7443 | 3733 | 583 | 737 | 1167 |

**Finding**: Adding the 768-dim query embedding drops Macro F1 by ~0.27 in both single- and dual-neutral configurations. The query introduces high-dimensional noise that competes with the residual signal in the logistic regression weight space.

---

### RQ5 — How does access to multiple neutral sources affect each approach?

#### Fine-tuned prompted classifiers — single vs dual neutral

| Setup | Neutral sources | Macro F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|
| `setup7-qwen` | Qwen only | **0.9964** | 4311 | 5 | 14 | 1890 |
| `setup116` | Gemini + Qwen | 0.9962 | 4312 | 4 | 16 | 1888 |
| `setup7` | Gemini only | 0.9953 | 4298 | 18 | 7 | 1897 |

**Finding**: The dual-neutral setup (setup116) marginally improves over Gemini-only (setup7) but does not surpass Qwen-only (setup7-qwen). Provider quality dominates — a second neutral does not deliver a step change.

#### Delta-LR — single vs dual neutral, and provider choice

| Setup | Neutral sources | Macro F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|
| `setup103` | Gemini only | **0.9913** | 4293 | 23 | 23 | 1881 |
| `setup104` | Gemini only (+ absolute pos.) | 0.9915 | 4306 | 10 | 35 | 1869 |
| `setup113` | `[delta_gemini \| delta_qwen]` | 0.9857 | 4260 | 56 | 20 | 1884 |
| `setup119` | Qwen only | 0.7475 | 3689 | 627 | 694 | 1210 |
| `setup114` | full dual stack (abs. + delta) | 0.7527 | 3599 | 717 | 610 | 1294 |

**Finding**: The Qwen-only residual (setup119, 0.7475) is far weaker than the Gemini residual (setup103, 0.9913). Concatenating both deltas (setup113) recovers most of the Gemini-only quality but still trails it — the weaker Qwen residual introduces noise. Provider quality matters far more than provider diversity.

---

## No Committed Results Yet

All core research-question runs now have committed results.

Secondary or backbone-comparison runs still missing:

- `setup4`
- `setup9`
- `setup11`

## Key Findings

- Best committed overall classifier is now `setup115`; it outperforms all
  query-aware committed classifiers despite using the response alone
- Best committed Qwen-backed classifier remains `setup6-qwen`
- Best committed Gemini-backed query-aware classifier remains `setup12`
- Prompted neutral-aware classifiers (`setup7`, `setup7-qwen`, `setup116`) are
  all strong, but none beat the best plain query-aware baselines; `setup116`
  improves over Gemini-only `setup7`, but still trails `setup7-qwen`,
  `setup12`, and `setup105_1`
- The stable cross-encoder retry `setup105_1` is competitive with the best
  plain classifiers, avoids the collapse seen in `setup105`, and is currently
  the strongest committed neutral-aware fine-tuned model
- Learned embedding features (`setup103`, `setup104`, `setup113`) are far
  stronger than raw cosine thresholding (`setup100`–`setup102`)
- The newly committed learned embedding variants `setup114`, `setup117`,
  `setup118`, and `setup119` remain far below the strongest classifier family
  and below the best earlier embedding-feature baselines
- `setup104` slightly beats `setup103` on Macro F1, but the tradeoff is mostly
  precision vs recall, not a step-change in quality
- `setup113` shows that a dual-neutral residual is viable, but it still trails
  the strongest single-neutral learned embedding setups; the second neutral has
  not yet shown a clear committed Macro-F1 gain
- `setup110` and `setup111` show that collapsing the comparison down to six
  cosine scalars loses too much directional information to compete with the
  vector-delta family; `setup111` only marginally exceeds `setup110`, so the
  main failure is the representation bottleneck, not the learned layer
- Both collapse cases (`setup8`, `setup105`) predict everything as positive;
  the shared theme is DeBERTa instability in this repo
