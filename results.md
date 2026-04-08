# Results Overview

All metrics are on the `test` split. Confusion matrix columns: TN / FP / FN / TP (rows = gold, cols = predicted).

## Result Table

| Setup | Family | Eval source | Macro F1 | TN | FP | FN | TP | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `setup6-qwen` | classifier | Qwen | 0.9985 | 4313 | 3 | 5 | 1899 | best committed Qwen-backed classifier |
| `setup12` | classifier | Gemini | 0.9977 | 4312 | 4 | 8 | 1896 | best committed Gemini-backed classifier |
| `setup6` | classifier | Gemini | 0.9975 | 4305 | 11 | 2 | 1902 | strongest Gemini recall |
| `setup105_1` | cross-encoder | Gemini | 0.9975 | 4315 | 1 | 12 | 1892 | stable RoBERTa cross-encoder retry |
| `setup7-qwen` | classifier | Qwen | 0.9964 | 4311 | 5 | 14 | 1890 | long-context with Qwen neutral reference |
| `setup10` | classifier | Gemini | 0.9920 | 4306 | 10 | 32 | 1872 | strong compact baseline (ALBERT) |
| `setup104` | learned embedding feature | Gemini | 0.9915 | 4306 | 10 | 35 | 1869 | best committed single-neutral embedding feature |
| `setup103` | learned embedding feature | Gemini | 0.9913 | 4293 | 23 | 23 | 1881 | residual-only Gemini baseline |
| `setup113` | learned embedding feature | Gemini + Qwen | 0.9857 | 4260 | 56 | 20 | 1884 | committed dual-neutral residual |
| `setup106` | descriptor-only sentence delta | Gemini | 0.9706 | 4196 | 120 | 37 | 1867 | historical artifact; backend not currently wired |
| `setup111` | scalar anchor baseline | Gemini + Qwen | 0.5669 | 3438 | 878 | 1269 | 635 | handcrafted `response_drift - anchor_cohesion` control |
| `setup110` | scalar anchor baseline | Gemini + Qwen | 0.5653 | 2729 | 1587 | 907 | 997 | learned weights over the same six scalars |
| `setup100` | embedding divergence | Gemini | 0.4532 | 2755 | 1561 | 1396 | 508 | balanced drift baseline, not competitive |
| `setup102` | embedding divergence | Gemini | 0.3533 | 582 | 3734 | 144 | 1760 | larger encoder, same failure mode as setup101 |
| `setup101` | embedding divergence | Gemini | 0.3509 | 559 | 3757 | 125 | 1779 | overpredicts positives |
| `setup105` | cross-encoder | Gemini | 0.2344 | 0 | 4316 | 0 | 1904 | collapsed to all-positive; DeBERTa instability + missing eval config |
| `setup8` | classifier | Gemini | 0.2344 | 0 | 4316 | 0 | 1904 | collapsed to all-positive predictions |

## No Committed Results Yet

Active but still uncommitted:

- `setup4`
- `setup7`
- `setup9`
- `setup11`
- `setup114`
- `setup115`
- `setup116`
- `setup117`
- `setup118`
- `setup119`

## Key Findings

- Best classifier overall remains `setup6-qwen` (Qwen) and `setup12` (Gemini)
- The stable cross-encoder retry `setup105_1` is competitive with the best
  plain classifiers and avoids the collapse seen in `setup105`
- Learned embedding features (`setup103`, `setup104`, `setup113`) are far
  stronger than raw cosine thresholding (`setup100`–`setup102`)
- `setup104` slightly beats `setup103` on Macro F1, but the tradeoff is mostly
  precision vs recall, not a step-change in quality
- `setup113` shows that a dual-neutral residual is viable, but it still trails
  the strongest single-neutral learned embedding setups
- `setup110` and `setup111` show that collapsing the comparison down to six
  cosine scalars loses too much directional information to compete with the
  vector-delta family
- Both collapse cases (`setup8`, `setup105`) predict everything as positive;
  the shared theme is DeBERTa instability in this repo
