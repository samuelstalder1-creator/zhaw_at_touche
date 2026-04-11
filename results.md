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
| `setup7-qwen` | classifier | Qwen | 0.9964 | 4311 | 5 | 14 | 1890 | long-context with Qwen neutral reference |
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

## No Committed Results Yet

Core research-question runs still missing:

- `setup7`
- `setup116`

Secondary or backbone-comparison runs still missing:

- `setup4`
- `setup9`
- `setup11`

## Key Findings

- Best committed overall classifier is now `setup115`; it outperforms all
  query-aware committed classifiers despite using the response alone
- Best committed Qwen-backed classifier remains `setup6-qwen`
- Best committed Gemini-backed query-aware classifier remains `setup12`
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
