# Results Overview

This file summarizes the committed evaluation artifacts in `results/`.

The canonical source of truth is still the per-run `metrics_summary.json` file
inside each result directory.

## Scope

- All committed metrics here are on the `test` split.
- The main binary task labels are the same across runs.
- Some runs are Gemini-backed and some are Qwen-backed. That does not make the
  labels different, but it does mean the surrounding generated fields and setup
  defaults are not identical.

## Committed Result Table

| Setup | Family | Eval source | Accuracy | Macro F1 | Positive F1 | Positive precision | Positive recall | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `setup6-qwen` | classifier | Qwen-enriched file | 0.9987 | 0.9985 | 0.9979 | 0.9984 | 0.9974 | best committed Qwen-backed classifier |
| `setup12` | classifier | Gemini-enriched file | 0.9981 | 0.9977 | 0.9968 | 0.9979 | 0.9958 | best committed Gemini-backed classifier |
| `setup6` | classifier | Gemini-enriched file | 0.9979 | 0.9975 | 0.9966 | 0.9942 | 0.9989 | strongest committed Gemini recall |
| `setup10` | classifier | Gemini-enriched file | 0.9932 | 0.9920 | 0.9889 | 0.9947 | 0.9832 | strong lightweight baseline |
| `setup104` | archived embedding-feature classifier | Gemini-enriched file | 0.9928 | 0.9915 | 0.9881 | 0.9947 | 0.9816 | strongest committed archived embedding-feature result |
| `setup103` | archived embedding-feature classifier | Gemini-enriched file | 0.9926 | 0.9913 | 0.9879 | 0.9879 | 0.9879 | strong residual-only archived result |
| `setup100` | embedding divergence | Gemini-enriched file | 0.5246 | 0.4532 | 0.2557 | 0.2455 | 0.2668 | balanced drift baseline, not competitive |
| `setup101` | embedding divergence | Gemini-enriched file | 0.3759 | 0.3509 | 0.4782 | 0.3214 | 0.9343 | overpredicts positives |
| `setup102` | embedding divergence | Gemini-enriched file | 0.3765 | 0.3533 | 0.4758 | 0.3203 | 0.9244 | larger encoder, same failure mode |
| `setup8` | classifier | Gemini-enriched file | 0.3061 | 0.2344 | 0.4687 | 0.3061 | 1.0000 | committed run collapsed to all-positive predictions |

## Setups Without Committed Evaluation Artifacts

No committed evaluation artifacts currently exist for:

- `setup4`
- `setup7`
- `setup7-qwen`
- `setup9`
- `setup11`
- `setup105`
- `setup106`

## What The Numbers Say

### Best current production-style classifiers

- `setup12` is the best committed Gemini-backed classifier.
- `setup6` is nearly tied and has the best committed Gemini-backed positive
  recall.
- `setup6-qwen` is the strongest committed Qwen-backed run.

### Strong archived research ideas

- `setup103` and `setup104` show that learned neutral-vs-response embedding
  features are much stronger than raw cosine-distance thresholding.
- `setup104` is the best of the archived embedding-feature ideas.
- These results are important, but the corresponding training backends are not
  currently exposed by the present `touche-train` CLI.

### Weak family: semantic-drift thresholding

- `setup100` to `setup102` all underperform the classifier family badly.
- `setup100` is the most balanced of the three, but still weak.
- `setup101` and `setup102` push positive recall high by overpredicting the
  positive class.

### Failed or unstable run

- The committed `setup8` artifact is a collapse case: every example was
  predicted as positive.
- The current JSON config already contains the stabilization knobs that should
  be retried before judging the DeBERTa-v3 backbone itself.

## Practical Guidance

- If you need the strongest committed Gemini-backed model, start with
  `setup12`, then compare against `setup6`.
- If you need a committed Qwen-backed run, start with `setup6-qwen`.
- If you are exploring neutral-vs-response modeling ideas, read `setup103` and
  `setup104` in `setup.md` before spending more time on `setup100` to
  `setup102`.
- If you want to revisit DeBERTa-v3, rerun `setup8` or `setup9` with the
  current stabilized settings rather than relying on the old collapsed artifact.
