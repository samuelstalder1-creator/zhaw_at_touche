# Results Overview

All metrics are on the `test` split. Confusion matrix columns: TN / FP / FN / TP (rows = gold, cols = predicted).

## Result Table

| Setup | Family | Eval source | Macro F1 | TN | FP | FN | TP | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `setup6-qwen` | classifier | Qwen | 0.9985 | 4313 | 3 | 5 | 1899 | best committed Qwen-backed classifier |
| `setup12` | classifier | Gemini | 0.9977 | 4312 | 4 | 8 | 1896 | best committed Gemini-backed classifier |
| `setup6` | classifier | Gemini | 0.9975 | 4305 | 11 | 2 | 1902 | strongest Gemini recall |
| `setup7-qwen` | classifier | Qwen | 0.9964 | 4311 | 5 | 14 | 1890 | long-context with Qwen neutral reference |
| `setup10` | classifier | Gemini | 0.9920 | 4306 | 10 | 32 | 1872 | strong compact baseline (ALBERT) |
| `setup104` | archived embedding-feature | Gemini | 0.9915 | 4306 | 10 | 35 | 1869 | best archived embedding-feature result |
| `setup103` | archived embedding-feature | Gemini | 0.9913 | 4293 | 23 | 23 | 1881 | residual-only archived result |
| `setup106` | archived sentence-delta | Gemini | 0.9706 | 4196 | 120 | 37 | 1867 | sentence-level delta with Hungarian matching |
| `setup110` | anchor-distance | Gemini + Qwen | 0.5653 | 2729 | 1587 | 907 | 997 | 6 cosine scalars, loses directional information |
| `setup100` | embedding divergence | Gemini | 0.4532 | 2755 | 1561 | 1396 | 508 | balanced drift baseline, not competitive |
| `setup102` | embedding divergence | Gemini | 0.3533 | 582 | 3734 | 144 | 1760 | larger encoder, same failure mode as setup101 |
| `setup101` | embedding divergence | Gemini | 0.3509 | 559 | 3757 | 125 | 1779 | overpredicts positives |
| `setup105` | cross-encoder | Gemini | 0.2344 | 0 | 4316 | 0 | 1904 | collapsed to all-positive; DeBERTa instability + missing eval config |
| `setup8` | classifier | Gemini | 0.2344 | 0 | 4316 | 0 | 1904 | collapsed to all-positive predictions |

## No Committed Results Yet

`setup4`, `setup7`, `setup9`, `setup11`, `setup105_1`, `setup111`, `setup113`, `setup114`

## Key Findings

- Best classifier overall: `setup6-qwen` (Qwen) and `setup12` (Gemini)
- Neutral reference in prompt (`setup7-qwen`) is competitive but does not beat plain `query_response` classifiers
- Learned embedding features (`setup103`, `setup104`) are far stronger than raw cosine thresholding (`setup100`–`setup102`)
- Both collapse cases (`setup8`, `setup105`) predict everything as positive; root cause is DeBERTa instability
- `setup110` fails because 6 cosine scalars discard the directional information that makes `setup103` work
- `setup111` is the no-classifier counterpart to `setup110`; it is now runnable
  but has no committed test artifact yet
