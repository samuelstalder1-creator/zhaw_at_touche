# Setup Reference

This file is the canonical explanation of the experiment matrix in this
repository. It explains what each setup is trying to do, what concepts its
configuration uses, which setups are runnable through the current CLI, and
which results are already committed.

The authoritative sources behind this summary are:

- `train_model/*.json`
- `validate_model/*.json`
- `results/*/metrics_summary.json`

## Setup Inventory

### Runnable Through The Current `touche-train` CLI

| Setup | Family | Trainer type | Main idea | Default data source |
| --- | --- | --- | --- | --- |
| `setup4` | reference-aware classifier | `classifier` | DeBERTa-v3 with explicit neutral reference + RAG response prompt | Gemini-enriched JSONL |
| `setup6` | baseline classifier | `classifier` | RoBERTa baseline on query + response only | Gemini-enriched JSONL |
| `setup6-qwen` | provider-aligned baseline | `classifier` | same RoBERTa recipe as `setup6`, but reads Qwen-enriched files | Qwen-enriched JSONL |
| `setup7` | long-context classifier | `classifier` | Longformer with neutral reference injected into the prompt | Gemini-enriched JSONL |
| `setup7-qwen` | provider-aligned long-context classifier | `classifier` | same Longformer recipe as `setup7`, but the neutral reference comes from Qwen | Qwen-enriched JSONL |
| `setup8` | DeBERTa retry | `classifier` | DeBERTa-v3 on query + response with the stabilization knobs that were missing from the first attempt | Gemini-enriched JSONL |
| `setup9` | stabilized DeBERTa | `classifier` | lower-risk DeBERTa-v3 variant with lower LR and stronger regularization | Gemini-enriched JSONL |
| `setup10` | lightweight classifier | `classifier` | ALBERT baseline with linear warmup/decay | Gemini-enriched JSONL |
| `setup11` | discriminator classifier | `classifier` | ELECTRA baseline with linear warmup/decay | Gemini-enriched JSONL |
| `setup12` | distilled classifier | `classifier` | DistilRoBERTa baseline with linear warmup/decay | Gemini-enriched JSONL |
| `setup100` | semantic-drift baseline | `embedding_divergence` | sentence-level cosine drift with mean aggregation | Gemini-enriched JSONL |
| `setup101` | higher-recall drift baseline | `embedding_divergence` | sentence-level cosine drift with top-3 aggregation | Gemini-enriched JSONL |
| `setup102` | larger-encoder drift baseline | `embedding_divergence` | same idea as `setup101`, but with BGE-large | Gemini-enriched JSONL |
| `setup110` | anchor-distance baseline | `anchor_distance_classifier` | logistic regression over six pairwise query/Gemini/Qwen/response cosine distances | merged Gemini + Qwen JSONL |

### Present In `train_model/`, But Not Currently Runnable Through The Current CLI

The current parser accepts `trainer_type=classifier`,
`trainer_type=embedding_divergence`, and
`trainer_type=anchor_distance_classifier`. The JSON files below are therefore best
treated as archived research descriptors unless the missing training backends
are reintroduced.

| Setup | Archived trainer idea | What it was testing | Current state |
| --- | --- | --- | --- |
| `setup103` | `embedding_residual_classifier` | logistic regression on `response_emb - neutral_emb` | archived config, committed evaluation artifacts exist |
| `setup104` | `embedding_classifier` | logistic regression on `[response_emb | neutral_emb | delta]` | archived config, committed evaluation artifacts exist |
| `setup105` | `cross_encoder` | jointly attend over `(response, neutral)` with DeBERTa-v3 | archived config only |
| `setup106` | `sentence_delta_classifier` | sentence-level aligned delta classifier with document aggregation | archived config only |

## Concepts Used Across The Setup Matrix

### Backbone Concepts

- `RoBERTa-base`: a strong, conventional encoder-only classifier backbone. In
  this repo it is the plain baseline against which more specialized ideas are
  compared.
- `DeBERTa-v3-base`: a stronger encoder family that often performs well on
  classification, but in practice can be less forgiving about optimizer and
  scheduler settings.
- `Longformer-base-4096`: a sparse-attention transformer built for longer
  contexts. Here it is used because `setup7` packs query, neutral reference,
  and response into one prompt.
- `ALBERT-base-v2`: a compact encoder that reuses parameters across layers.
  This lowers memory and compute cost, but the smaller effective capacity often
  benefits from a slightly higher learning rate.
- `ELECTRA-base-discriminator`: pretrained via replaced-token detection instead
  of masked-language modeling. That pretraining objective often helps on
  fine-grained classification.
- `DistilRoBERTa`: a smaller distilled RoBERTa. It usually trains and runs
  faster, with some risk of slightly reduced ceiling versus a full-size model.
- `all-mpnet-base-v2`, `BAAI/bge-large-en-v1.5`: sentence embedding encoders
  used for setups that compare responses in embedding space instead of
  fine-tuning a classifier.

### Input-Format Concepts

- `query_response`: the model only sees the user query and the target response.
  This is the simplest classifier framing and is used by `setup6`, `setup6-qwen`,
  `setup8`, `setup9`, `setup10`, `setup11`, and `setup12`.
- `query_neutral_response`: the model sees the query, a neutral rewrite, and
  the target response together. This is used by `setup7` and `setup7-qwen`.
- `query_reference_rag_response`: the model sees the query, a labeled neutral
  reference, and the target response. This is used by `setup4`.

### Reference And Provider Concepts

- `neutral_field`: the field that contains the neutral response generated by a
  provider such as Gemini or Qwen. For Gemini-backed data this is usually
  `gemini25flashlite`; for Qwen-backed data it is `qwen`.
- `reference_field`: the field injected into the prompt for reference-aware
  classifier setups.
- `reference_label`: the human-readable label rendered in the prompt, such as
  `GEMINI`, `QWEN`, or `Unbiased Reference`.
- `generated_provider`: validation shortcut that swaps the default input files
  and, for reference-aware setups, also swaps the default reference field and
  label.
- `setup6-qwen` versus `setup7-qwen`: these names do not mean the backbone is
  Qwen. They mean the input files come from the Qwen-generated neutral-response
  datasets. This matters a lot for `setup7-qwen`, because the neutral field is
  part of the prompt. It matters much less for `setup6-qwen`, because
  `query_response` ignores the neutral field entirely.

### Optimization Concepts

- `epochs`: number of passes through the training data.
- `batch_size`: per-device micro-batch size.
- `grad_accum`: number of micro-batches accumulated before each optimizer step.
  The effective batch size is `batch_size x grad_accum`.
- `learning_rate`: step size used by the optimizer.
- `optimizer_eps`: AdamW numerical-stability term. DeBERTa setups in this repo
  raise it above the default `1e-8` because that was empirically safer.
- `lr_scheduler`: how the learning rate changes over time.
  `none` keeps it flat, `linear` decays linearly after warmup, and
  `cosine_with_warmup` uses a cosine decay curve after warmup.
- `warmup_ratio`: fraction of training steps spent increasing the learning rate
  gradually from zero to its target value.
- `weight_decay`: L2-style regularization applied through AdamW.
- `max_grad_norm`: gradient clipping threshold. This is one of the key
  anti-collapse controls in the DeBERTa setups.
- `gradient_checkpointing`: trades extra compute for lower memory by
  recomputing some activations during backpropagation.
- `layerwise_lr_decay`: uses smaller learning rates in lower encoder layers and
  larger ones near the task head. This is a conservative fine-tuning trick.
- `freeze_embeddings_epochs`: temporarily keeps the embedding layer frozen at
  the start of training to avoid large destabilizing updates.
- `pad_to_max_length`: always pad to `max_length` instead of using dynamic
  batch padding. This can simplify long-context behavior at the cost of more
  wasted tokens.
- `positive_class_weight_scale`: scales the positive-class loss weight to
  compensate for class imbalance.

### Embedding-Comparison Concepts

- `distance_metric=cosine`: the score is based on angular distance in embedding
  space, not raw Euclidean distance.
- `score_granularity=response`: compare one full response embedding against one
  full neutral embedding.
- `score_granularity=sentence`: compare sentence embeddings instead of whole
  passages.
- `sentence_agg=mean`: average the sentence-level distances. Balanced, but it
  can dilute a small localized ad insertion.
- `sentence_agg=top3_mean`: only average the three most divergent sentences.
  This is a higher-recall choice when only a few sentences look promotional.
- `threshold_metric=macro_f1`: pick the decision threshold that balances both
  classes.
- `threshold_metric=positive_f1`: pick the threshold that optimizes the
  positive class. This tends to push recall up.
- `residual vector`: `response_emb - neutral_emb`. This is the representation
  used in `setup103`. The idea is that the subtraction isolates what the
  neutral rewrite removed.
- `full embedding feature stack`: `[response_emb | neutral_emb | residual]`.
  This is the representation used in `setup104`.
- `anchor-distance feature stack`: six response-level cosine distances between
  `(query, Gemini neutral, Qwen neutral, response)`. This is the active idea
  used by `setup110`.
- `cross-encoder`: a model that jointly attends over two texts instead of
  encoding them independently and comparing vectors after the fact. This is the
  idea behind archived `setup105`.
- `Hungarian matching`: optimal bipartite matching between response sentences
  and neutral sentences. Archived `setup106` uses this concept instead of the
  greedy alignment used by `setup100` to `setup102`.
- `sentence_delta_agg=mean_prob`: aggregate sentence-level predicted ad
  probabilities into one document-level score. This appears in archived
  `setup106`.

## Per-Setup Explanations

### `setup4`

- Backbone: `microsoft/deberta-v3-base`.
- Prompt concept: `query_reference_rag_response`, so the classifier sees a
  query, an explicitly labeled neutral answer, and the target response.
- Stability concepts: `optimizer_eps=1e-6`, `max_grad_norm=1.0`,
  `cosine_with_warmup`, `warmup_ratio=0.05`, and `gradient_checkpointing=true`.
- Class-balance concept: `positive_class_weight_scale=1.0`, which is lower than
  the repo-wide default and therefore less aggressive about positive weighting.
- Interpretation: this setup asks whether explicitly contrasting the response
  against a clean reference helps more than a plain query-response classifier.

### `setup6`

- Backbone: `FacebookAI/roberta-base`.
- Prompt concept: `query_response`.
- Optimization concept: intentionally simple. No scheduler, no extra
  regularization, and no reference-aware prompt.
- Role in the matrix: this is the baseline that answers, "How far can a plain
  RoBERTa classifier get without more complex prompt engineering?"
- Result status: strong committed Gemini-backed result.

### `setup6-qwen`

- Same backbone and prompt as `setup6`.
- Data-source concept: the files come from `data/generated/qwen/` instead of
  `data/generated/gemini/`.
- Important nuance: because the prompt is still `query_response`, the Qwen
  neutral text is not injected into the classifier input. This setup exists
  mainly to keep provider-specific train and evaluation files aligned.
- Result status: strongest committed Qwen-backed classifier result in the repo.

### `setup7`

- Backbone: `allenai/longformer-base-4096`.
- Prompt concept: `query_neutral_response`.
- Long-context concept: `max_length=1024`, `batch_size=4`, `grad_accum=8`,
  `pad_to_max_length=true`.
- Class-balance concept: `positive_class_weight_scale=1.5`.
- Interpretation: this setup tests whether explicitly showing the model a
  neutral rewrite improves ad detection enough to justify longer context and
  smaller batches.

### `setup7-qwen`

- Same Longformer recipe as `setup7`.
- Provider concept: `reference_field=qwen`, `reference_label=QWEN`.
- Data-source concept: both training and validation files come from
  `data/generated/qwen/`.
- Interpretation: this isolates whether the long-context neutral-reference idea
  behaves differently when the neutral source is Qwen instead of Gemini.

### `setup8`

- Backbone: `microsoft/deberta-v3-base`.
- Prompt concept: still the plain `query_response` format.
- Stability concepts: `optimizer_eps=1e-6`, `max_grad_norm=1.0`,
  `cosine_with_warmup`, `warmup_ratio=0.06`.
- Interpretation: this is the direct "DeBERTa instead of RoBERTa" comparison,
  but with the safety knobs that DeBERTa usually needs.
- Result status: the committed run collapsed to all-positive predictions, so
  the setup remains an open question rather than a proven winner.

### `setup9`

- Backbone: `microsoft/deberta-v3-base`.
- Prompt concept: `query_response`.
- Conservative optimization concepts: lower `learning_rate=8e-6`,
  `optimizer_eps=1e-7`, `weight_decay=0.01`, `warmup_ratio=0.1`,
  `layerwise_lr_decay=0.9`, `freeze_embeddings_epochs=1`.
- Interpretation: this is the "slow down and regularize DeBERTa" variant after
  the instability seen around the other DeBERTa runs.

### `setup10`

- Backbone: `albert/albert-base-v2`.
- Architecture concept: ALBERT shares parameters across layers, which reduces
  memory and compute cost.
- Optimization concepts: `linear` scheduler, `warmup_ratio=0.06`,
  `weight_decay=0.01`, slightly higher `learning_rate=3e-5`.
- Interpretation: a compact, efficient classifier baseline with conventional
  prompt structure.
- Result status: strong committed classifier result, but below the best RoBERTa
  and DistilRoBERTa runs.

### `setup11`

- Backbone: `google/electra-base-discriminator`.
- Architecture concept: ELECTRA is pretrained with replaced-token detection,
  which often transfers well to discriminative tasks.
- Optimization concepts: `linear` scheduler, `warmup_ratio=0.06`,
  `weight_decay=0.01`, `max_grad_norm=1.0`.
- Interpretation: this asks whether a discriminator-style pretraining signal is
  especially useful for detecting promotional phrasing.
- Result status: no committed evaluation artifacts yet.

### `setup12`

- Backbone: `distilroberta-base`.
- Architecture concept: a distilled RoBERTa with fewer layers and lower
  latency.
- Optimization concepts: `epochs=5`, `learning_rate=3e-5`, `linear` scheduler,
  `warmup_ratio=0.06`, `weight_decay=0.01`.
- Interpretation: the "fast but still strong" classifier candidate.
- Result status: best committed Gemini-backed classifier result in the repo.

### `setup100`

- Trainer concept: `embedding_divergence`, not a fine-tuned classifier.
- Representation concept: compare the response and neutral rewrite in sentence
  embedding space.
- Scoring concepts: `score_granularity=sentence`, `sentence_agg=mean`,
  `threshold_metric=macro_f1`.
- Interpretation: this is the balanced baseline for semantic drift scoring.
- Result status: clearly weaker than the classifier setups.

### `setup101`

- Same training backend as `setup100`.
- Recall-oriented concepts: `sentence_agg=top3_mean`,
  `threshold_metric=positive_f1`.
- Interpretation: instead of averaging away one inserted ad sentence, it keeps
  the most divergent local regions.
- Result status: much higher ad recall than `setup100`, but with severe
  overprediction.

### `setup102`

- Same scoring logic as `setup101`.
- Capacity concept: swap the encoder to `BAAI/bge-large-en-v1.5`.
- Practical concept: lower `batch_size=16` because the encoder is larger.
- Interpretation: asks whether a stronger sentence encoder can rescue the
  semantic-drift approach.
- Result status: modestly different numbers, but still far behind the
  classifier baselines.

### `setup110`

- Trainer concept: `anchor_distance_classifier`.
- Representation concept: embed `query`, `response`, Gemini neutral, and Qwen
  neutral at response level, then compute six cosine distances:
  `query-Gemini`, `query-Qwen`, `Gemini-Qwen`, `query-response`,
  `Gemini-response`, and `Qwen-response`.
- Modeling concept: a lightweight logistic regression learns how much those
  "small neutral triangle, larger response drift" signals correlate with the
  ad label.
- Interpretation: this is the runnable version of the user's multi-anchor
  delta idea. It is closer in spirit to archived `setup103` than to the
  sentence-drift baselines, but it stays compact and directly interpretable.
- Current repo state: fully wired into `touche-train` and `touche-validate`,
  but no committed evaluation artifacts exist yet.

### `setup103`

- Archived trainer concept: `embedding_residual_classifier`.
- Representation concept: use only `response_emb - neutral_emb`.
- Modeling concept: a lightweight logistic regression over the residual vector.
- Interpretation: instead of thresholding a single cosine distance, let a
  classifier learn which residual directions correspond to advertising.
- Current repo state: committed results exist and are strong, but the training
  backend is not currently wired into `touche-train`.

### `setup104`

- Archived trainer concept: `embedding_classifier`.
- Representation concept: concatenate `response_emb`, `neutral_emb`, and the
  residual vector.
- Modeling concept: logistic regression gets both absolute locations and the
  direction of change.
- Interpretation: this is the strongest archived embedding-feature design in
  the repo and it outperforms the drift-only baselines by a wide margin.
- Current repo state: committed results exist, but the training backend is not
  currently wired into `touche-train`.

### `setup105`

- Archived trainer concept: `cross_encoder`.
- Modeling concept: jointly encode `(response, neutral)` so the model can
  attend across both texts instead of comparing two independent embeddings.
- Stability concepts: uses the DeBERTa-v3 stabilization recipe with
  `optimizer_eps=1e-6`, `max_grad_norm=1.0`, `cosine_with_warmup`,
  `warmup_ratio=0.06`, and `weight_decay=0.01`.
- Interpretation: this is the most expressive archived neutral-vs-response
  design, but there are no committed results yet.

### `setup106`

- Archived trainer concept: `sentence_delta_classifier`.
- Alignment concept: sentence pairs are built with Hungarian matching instead
  of greedy matching.
- Representation concept: per-pair features are
  `[sent_emb | neutral_sent_emb | delta]`.
- Aggregation concept: `sentence_delta_agg=mean_prob`, meaning sentence-level
  predictions are averaged into one document score.
- Interpretation: this tries to preserve localized sentence evidence while
  still making one document-level decision.

## Committed Results Snapshot

The table below reflects the committed `metrics_summary.json` files in
`results/`.

| Setup | Evaluation source | Accuracy | Macro F1 | Positive F1 | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `setup6-qwen` | Qwen-enriched test file | 0.9987 | 0.9985 | 0.9979 | best committed Qwen-backed classifier |
| `setup12` | Gemini-enriched test file | 0.9981 | 0.9977 | 0.9968 | best committed Gemini-backed classifier |
| `setup6` | Gemini-enriched test file | 0.9979 | 0.9975 | 0.9966 | strongest recall among committed Gemini classifiers |
| `setup10` | Gemini-enriched test file | 0.9932 | 0.9920 | 0.9889 | strong compact baseline |
| `setup104` | Gemini-enriched test file | 0.9928 | 0.9915 | 0.9881 | strongest archived embedding-feature result |
| `setup103` | Gemini-enriched test file | 0.9926 | 0.9913 | 0.9879 | residual-only archived embedding-feature result |
| `setup100` | Gemini-enriched test file | 0.5246 | 0.4532 | 0.2557 | balanced drift baseline, not competitive |
| `setup101` | Gemini-enriched test file | 0.3759 | 0.3509 | 0.4782 | high-recall drift baseline, overpredicts positives |
| `setup102` | Gemini-enriched test file | 0.3765 | 0.3533 | 0.4758 | larger-encoder drift baseline, still weak |
| `setup8` | Gemini-enriched test file | 0.3061 | 0.2344 | 0.4687 | committed run collapsed to all-positive predictions |

No committed evaluation artifacts currently exist for `setup4`, `setup7`,
`setup7-qwen`, `setup9`, `setup11`, `setup105`, `setup106`, or `setup110`.

## Practical Takeaways

- If you want the strongest committed Gemini-backed classifier today, start
  with `setup12` and keep `setup6` as the best-recall alternative.
- If you want the strongest committed Qwen-backed run, use `setup6-qwen`.
- If you want a compact, interpretable multi-reference embedding baseline, use
  `setup110`.
- If you want the most interesting archived neutral-vs-response ideas, look at
  `setup103` and `setup104`, but treat them as historical designs until their
  training backends are restored.
- If you want simple, robust baselines, prefer the plain classifier family over
  the embedding-divergence family.
