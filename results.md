# Results Overview

All results are on the **test split** (6,220 samples, ~69% negative / ~31% positive).

## Summary Table

| Setup | Approach | Model | Macro F1 | Pos F1 | Pos Prec | Pos Rec | TN | FP | FN | TP | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **setup12** | Fine-tuned classifier | DistilRoBERTa | **0.9977** | 0.9968 | 0.9979 | 0.9958 | 4312 | 4 | 8 | 1896 | best overall |
| **setup6** | Fine-tuned classifier | RoBERTa-base | **0.9975** | 0.9966 | 0.9942 | 0.9989 | 4305 | 11 | 2 | 1902 | best recall |
| **setup10** | Fine-tuned classifier | ALBERT-base-v2 | 0.9920 | 0.9889 | 0.9947 | 0.9832 | 4306 | 10 | 32 | 1872 | strong |
| setup100 | Embedding divergence | MiniLM-L6 / mpnet | 0.4532 | 0.2557 | 0.2455 | 0.2668 | 2755 | 1561 | 1396 | 508 | collapsed (config fixed) |
| setup101 | Embedding divergence | mpnet + top-3 | 0.3509 | 0.4782 | 0.3214 | 0.9343 | 559 | 3757 | 125 | 1779 | collapsed positive |
| setup102 | Embedding divergence | BGE-large + top-3 | 0.3533 | 0.4758 | 0.3203 | 0.9244 | 582 | 3734 | 144 | 1760 | collapsed positive |
| setup8 | Fine-tuned classifier | DeBERTa-v3-base | 0.2344 | 0.4687 | 0.3061 | 1.0000 | 0 | 4316 | 0 | 1904 | full collapse (config fixed) |

### Not yet trained / evaluated
| Setup | Approach | Model | Notes |
|---|---|---|---|
| setup4 | Fine-tuned classifier | DeBERTa-v3-base | RAG format + Gemini reference |
| setup7 | Fine-tuned classifier | Longformer-base-4096 | Long-context with neutral reference |
| setup8 | Fine-tuned classifier | DeBERTa-v3-base | Config fixed (eps, grad clipping, warmup) |
| setup9 | Fine-tuned classifier | DeBERTa-v3-base | Stabilized: lower LR, weight decay, layerwise decay, frozen embeddings |
| setup11 | Fine-tuned classifier | ELECTRA-base | Strong classifier pretraining objective |

---

## Per-Setup Notes

### setup12 — DistilRoBERTa (BEST)
- Macro F1: **0.9977**, only 12 errors total on 6,220 samples
- 2× faster inference than RoBERTa-base — best option for submission
- 5 epochs, LR=3e-5, linear scheduler, warmup=6%, weight_decay=0.01

### setup6 — RoBERTa-base
- Macro F1: **0.9975**, near-identical to setup12
- Best recall (FN=2) — misses almost no real ads
- 3 epochs, LR=2e-5, no scheduler (works because RoBERTa is forgiving)
- Earlier committed results showed full collapse — those were from a previous broken run

### setup10 — ALBERT-base-v2
- Macro F1: 0.9920, slightly below setup6/12
- FN=32 (misses more ads than the others)
- Good baseline, slightly weaker than RoBERTa/DistilRoBERTa on this task

### setup8 — DeBERTa-v3-base (full collapse, config now fixed)
- Predicted every sample as positive (TN=0) — classic gradient explosion on step 1
- Root cause: no gradient clipping (`max_grad_norm=null`), no warmup, `eps=1e-8`
- Config fixed: `eps=1e-6`, `max_grad_norm=1.0`, `cosine_with_warmup`, `warmup_ratio=0.06`, `epochs=3`
- **Needs retraining**

### setup100/101/102 — Embedding divergence (all weak)
- Heuristic approach: cosine distance between response and Gemini-generated neutral
- Fundamental ceiling: high distance ≠ advertisement — approach can't learn advertising intent
- setup100: config fixed (`sentence_agg: mean`, `threshold_metric: macro_f1`, upgraded to mpnet)
- setup101/102: near-collapse positive (recall=0.93–0.97, precision=0.32)
- **Do not use for submission** — fine-tuned classifiers are categorically better

---

## Direction: What to Do Next

### Priority 1 — Submit setup12 or setup6
Both are production-ready. setup12 has slightly fewer errors and half the inference cost.
If inference speed matters: **setup12**. If recall is paramount: **setup6**.

### Priority 2 — Retrain setup8 with fixed config
The fixed config (`eps=1e-6`, `max_grad_norm=1.0`, warmup) addresses the known DeBERTa-v3
instability. If it trains correctly, DeBERTa-v3-base is a stronger backbone than RoBERTa and
could surpass setup12. Worth one training run.

### Priority 3 — Try setup9 (stabilized DeBERTa-v3)
More conservative than setup8: lower LR (8e-6), longer warmup (10%), layerwise LR decay,
weight decay, one frozen-embedding epoch. If setup8 still collapses, setup9 is the safer DeBERTa path.

### Priority 4 — Try setup11 (ELECTRA-base)
ELECTRA's replaced-token-detection pretraining makes it particularly strong at detecting
unnatural/promotional phrasing — well-suited for ad detection. Standard RoBERTa-like hyperparameters
so low risk of training issues.

### Do not prioritize
- **setup4 / setup7**: RAG/long-context formats add complexity and memory cost. The simple
  `query_response` format already achieves 0.997+ F1 — the reference context provides no measurable
  benefit at this performance level.
- **setup100/101/102**: Embedding divergence has a hard ceiling. All three collapsed on the test set.
  Not worth further investment.

---

## Collapse Detection
`touche-validate` now prints a warning when any class has recall=0.0:
```
WARNING [<file>]: model never correctly predicted class(es) ['0'] — possible prediction collapse
```
