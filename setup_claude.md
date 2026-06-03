# Setup Descriptions

## Dataset

All setups train and evaluate on the same underlying corpus split into three fixed partitions:

| Split | Samples | Positive (ad) | Negative (non-ad) |
|---|---:|---:|---:|
| Train | 32 727 | 10 311 (31.5 %) | 22 416 (68.5 %) |
| Validation | 5 780 | 1 781 (30.8 %) | 3 999 (69.2 %) |
| Test | 6 220 | 1 904 (30.6 %) | 4 316 (69.4 %) |

Each record contains: `id`, `search_engine`, `meta_topic`, `query`, `response`, `label` (0 = non-ad, 1 = ad), and provider-specific neutral rewrite fields.

**Neutral rewrite providers:**

| Field name | Provider model | Data path |
|---|---|---|
| `gemini25flashlite` | Gemini 2.5 Flash Lite | `data/generated/gemini/` |
| `qwen` | Qwen/Qwen2.5-1.5B-Instruct | `data/generated/qwen/` |
| `gemma4_e4b` | Gemma 4E4B | `data/generated/gemma4e4b/` |

---

## Family A — Fine-tuned Classifiers

A pre-trained transformer encoder is fine-tuned end-to-end. The model receives a text sequence as input, and a binary classification head on the `[CLS]` token produces a logit. Both backbone weights and the classification head are updated during training. The encoder learns task-specific representations shaped by which patterns distinguish ad responses from clean ones.

**Shared training hyperparameters (all fine-tuned setups):**

| Parameter | Value |
|---|---|
| Learning rate | 3 × 10⁻⁵ |
| Weight decay | 0.01 |
| LR scheduler | Linear with warmup |
| Warmup ratio | 0.06 |
| Gradient accumulation | 4 steps |
| Batch size (per step) | 16 |
| Optimiser | AdamW (HuggingFace default) |

**Threshold:** fixed at 0.5 (logit ≥ 0.5 → ad).

---

### setup10 — *CompactQueryClassifier* (submitted to Touché)

| Property | Value |
|---|---|
| Backbone | `albert/albert-base-v2` |
| Input format | `query_response` |
| Max sequence length | 512 tokens |
| Epochs | 5 |
| Batch size | 16 |

**Architecture.** ALBERT-base-v2 uses cross-layer parameter sharing, which reduces the effective parameter count relative to BERT-base while maintaining the same hidden dimensionality (768). It is lightweight in memory and stable with standard hyperparameters but benefits from a slightly higher learning rate because fewer effective parameters are updated per gradient step.

**Input construction.**
```
Query: {query}
Response: {response}
Answer:
```
The neutral rewrite is not part of the input; the model sees only the query and the response to classify.

**Training.** 5 epochs over 32 727 training records (effective batch = 16 × 4 = 64). Linear LR warmup for the first 6 % of steps, then linear decay to zero. No class weighting (the class imbalance of ~31/69 is mild enough that the default cross-entropy loss is stable).

---

### setup10_1_* — Longformer + neutral rewrite

| Property | Value |
|---|---|
| Backbone | `allenai/longformer-base-4096` |
| Input format | `query_neutral_response` |
| Max sequence length | 1 024 tokens |
| Epochs | 1 |
| Batch size | 16 |
| Pad to max length | Yes |

**Architecture.** Longformer-base-4096 replaces BERT's quadratic self-attention with a combination of local sliding-window attention and global attention on selected tokens. This allows longer sequences without the quadratic memory cost of standard transformers. Max length is raised from 512 to 1 024 to accommodate the concatenation of query, neutral rewrite, and response in a single input.

**Variants.**

| Setup | Neutral provider | Neutral field | Reference label | Data split |
|---|---|---|---|---|
| `setup10_1_gemini` | Gemini 2.5 Flash Lite | `gemini25flashlite` | `GEMINI` | `data/generated/gemini/` |
| `setup10_1-gemma` | Gemma 4E4B | `gemma4_e4b` | `GEMMA4-E4B` | `data/generated/gemma4e4b/` |
| `setup10_1-qwen` | Qwen/Qwen2.5-1.5B-Instruct | `qwen` | `QWEN` | `data/generated/qwen/` |

**Input construction.**
```
USER QUERY: {query}

NEUTRAL REFERENCE ({provider_label}): {neutral_field}

RESPONSE TO CLASSIFY: {response}

LABEL THIS AS AD OR NEUTRAL:
```
The neutral rewrite appears before the response being classified, giving the model an explicit reference point: it can compare the response to the neutral and attend to what differs.

**Training.** 1 epoch (sequences are substantially longer and the model is larger; 1 epoch avoids overfitting on this dataset size). Padding to max length ensures all sequences in a batch have the same length, which is required for efficient Longformer local-attention computation.

---

## Family B — Frozen Embedding + Logistic Regression (Delta Models)

The encoder is completely frozen — no weights are updated during the classification step. `sentence-transformers/all-mpnet-base-v2` (a 110M-parameter MPNet model pre-trained on 1B sentence pairs for semantic similarity) is used in inference mode to produce 768-dimensional sentence embeddings via **attention-mask-weighted mean pooling** followed by **L2 normalisation**:

```
embed(text) = L2_normalise( mean_pool(last_hidden_state, attention_mask) )
```

The resulting embeddings are 768-dimensional unit vectors in semantic space. Because the encoder is frozen, all learning happens in the logistic regression layer on top.

**Shared LR training procedure (all delta setups):**

1. **Embed** training records (response and/or neutral text fields) using the frozen encoder, batched at 32 sequences of max 512 tokens.
2. **Build feature matrix** from embeddings (see per-setup details).
3. **Fit** a `StandardScaler` (zero-mean, unit-variance per dimension) on the training features, then transform.
4. **Fit** a `LogisticRegression` with:
   - `class_weight="balanced"` (adjusts weights inversely proportional to class frequencies: ~2.19× weight on positive class to compensate for the 31/69 imbalance)
   - `C=1.0` (L2 regularisation strength)
   - `solver="lbfgs"`, `max_iter=1000`
5. **Calibrate threshold** on the validation set: sweep all possible thresholds and select the one that maximises Macro F1 on the validation set. The selected threshold is saved with the model.
6. **Evaluate** on the test set using the saved threshold.

The StandardScaler and LogisticRegression are wrapped in a scikit-learn `Pipeline` and serialised together with the threshold to `embedding_lr_classifier.pkl`.

---

### setup104-base — *BaseStack* (submitted to Touché)

| Property | Value |
|---|---|
| Trainer type | `response_embedding_classifier` |
| Feature vector | `[response_emb]` — 768 dimensions |
| Neutral field | `qwen` (data alignment only, not used in features) |
| Train data | `data/generated/qwen/responses-train-with-neutral_qwen.jsonl` |
| Validation / Test | Qwen split (same records as delta models) |

**Feature construction.** Only the response embedding is used:
```
x = response_emb   (768-dim)
```
No neutral rewrite is encoded; this setup is the response-only frozen-encoder baseline. The Qwen-backed data files are used solely to align record counts with the Qwen delta models and enable a clean ablation.

**Purpose.** Establishes the ceiling for what a frozen encoder can achieve on response text alone, with no directional delta signal. Any improvement from setup104-qwen or setup119-qwen over this baseline is attributable entirely to the neutral rewrite.

---

### setup104-* — Full Embedding Stack

| Property | Value |
|---|---|
| Trainer type | `embedding_classifier` |
| Feature vector | `[response_emb ‖ neutral_emb ‖ (response_emb − neutral_emb)]` — 2 304 dimensions |

**Variants.**

| Setup | Touché name | Neutral provider | Neutral field | Data split |
|---|---|---|---|---|
| `setup104-gemini` | — | Gemini 2.5 Flash Lite | `gemini25flashlite` | `data/generated/gemini/` |
| `setup104-gemma` | — | Gemma 4E4B | `gemma4_e4b` | `data/generated/gemma4e4b/` |
| `setup104-qwen` | `QwenResidualStack` | Qwen/Qwen2.5-1.5B-Instruct | `qwen` | `data/generated/qwen/` |

**Feature construction.**
```
r = encode(response)       # 768-dim
n = encode(neutral_field)  # 768-dim
δ = r − n                  # 768-dim
x = concat(r, n, δ)        # 2304-dim
```
The feature vector contains three blocks: the absolute position of the response in embedding space, the absolute position of the provider-specific neutral rewrite, and the direction from neutral to response. The LR can exploit all three simultaneously: the absolute positions anchor the classification in semantic space, while the delta encodes specifically what the neutral removed.

---

### setup119-* — Residual-Only

| Property | Value |
|---|---|
| Trainer type | `embedding_residual_classifier` |
| Feature vector | `[response_emb − neutral_emb]` — 768 dimensions |

**Variants.**

| Setup | Touché name | Neutral provider | Neutral field | Data split |
|---|---|---|---|---|
| `setup119-gemini` | — | Gemini 2.5 Flash Lite | `gemini25flashlite` | `data/generated/gemini/` |
| `setup119-gemma` | — | Gemma 4E4B | `gemma4_e4b` | `data/generated/gemma4e4b/` |
| `setup119-qwen` | `QwenResidualOnly` | Qwen/Qwen2.5-1.5B-Instruct | `qwen` | `data/generated/qwen/` |

**Feature construction.**
```
r = encode(response)       # 768-dim
n = encode(neutral_field)  # 768-dim
δ = r − n                  # 768-dim
x = δ                      # 768-dim  (delta only)
```
Only the delta vector is passed to the LR. The absolute positions of the response and neutral in embedding space are discarded. The hypothesis is that the delta alone encodes the advertising signal: if the response is an ad, `δ` points in the direction of advertising language in embedding space; if the response is clean, `δ ≈ 0` because the neutral barely changed anything.

---
