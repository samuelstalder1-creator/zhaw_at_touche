# Research Question

> **Does a neutral rewrite help advertisement detection, and does a frozen semantic-delta representation substitute for end-to-end fine-tuning?**

This comparison focuses on the thesis-relevant setups listed in `thesis_relevant.md`.
All reported metrics are from the committed test-set result artifacts in `results/`.
Precision, recall, and F1 refer to the positive class `1`.

---

## Modellvergleich

### Fine-tuned models

A pre-trained transformer encoder (ALBERT, Longformer) is fine-tuned end-to-end on the classification task. The model receives a text sequence as input, and a binary classification head on top of the `[CLS]` token produces a logit. Both the backbone weights and the classification head are updated during training.

The key property is that **the encoder learns task-specific representations**: after fine-tuning, the internal activations are no longer general-purpose sentence embeddings but are shaped specifically by which token patterns distinguish ad responses from clean ones.

| Input | Setup |
| --- | --- |
| query + response | `setup10` |
| query + response + neutral | `setup10_1-gemini`, `setup10_1-gemma`, `setup10_1-qwen` |

### Delta models

Instead of fine-tuning the encoder, the encoder stays **completely frozen**. The idea is that a strong general-purpose sentence encoder, here `sentence-transformers/all-mpnet-base-v2`, already represents semantic content accurately enough that the difference between the response embedding and the neutral embedding directly encodes what the neutral rewrite removed: the advertising signal.

Formally: let `r = encode(response)` and `n = encode(neutral)`. The delta is `d = r - n`. If the response is an ad, `d` points in the direction of advertising language in embedding space. If the response is clean, `d` should be smaller or less systematically aligned with advertising features.

A **logistic regression** with a standard scaler is then trained on top of frozen embedding features. Only the logistic regression weights and the scaler are learned. The encoder is never updated. This separates representation learning, done offline by the pre-trained encoder, from decision-boundary learning, done by the LR on top of fixed features.

| Input | Setup |
| --- | --- |
| response | `setup104-base` |
| response + 1 neutral + delta | `setup104-gemini`, `setup104-gemma`, `setup104-qwen` |
| response - 1 neutral | `setup119-gemini`, `setup119-gemma`, `setup119-qwen` |

---

## Results

| Setup | Family | Input | Accuracy | Recall | Precision | F1 | Errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `setup10_1-gemma` | fine-tuned | query + response + Gemma neutral | 0.998 | 0.999 | 0.996 | 0.997 | 10 |
| `setup10_1-qwen` | fine-tuned | query + response + Qwen neutral | 0.998 | 0.995 | 0.998 | 0.996 | 14 |
| `setup10_1_gemini` | fine-tuned | query + response + Gemini neutral | 0.998 | 0.995 | 0.998 | 0.996 | 14 |
| `setup10` | fine-tuned | query + response | 0.993 | 0.983 | 0.995 | 0.989 | 42 |
| `setup104-qwen` | frozen embedding LR | response + Qwen neutral + delta | 0.791 | 0.678 | 0.653 | 0.665 | 1301 |
| `setup119-qwen` | frozen embedding LR | response - Qwen neutral | 0.788 | 0.636 | 0.659 | 0.647 | 1321 |
| `setup104-gemini` | frozen embedding LR | response + Gemini neutral + delta | 0.772 | 0.662 | 0.619 | 0.640 | 1420 |
| `setup119-gemini` | frozen embedding LR | response - Gemini neutral | 0.766 | 0.601 | 0.622 | 0.612 | 1454 |
| `setup104-gemma` | frozen embedding LR | response + Gemma neutral + delta | 0.756 | 0.670 | 0.589 | 0.627 | 1519 |
| `setup104-base` | frozen embedding LR | response only | 0.737 | 0.657 | 0.560 | 0.605 | 1635 |
| `setup119-gemma` | frozen embedding LR | response - Gemma neutral | 0.727 | 0.718 | 0.540 | 0.617 | 1700 |

---

## Research Findings

### 1. Does the neutral rewrite help fine-tuned models?

Yes. The Longformer setups that include a neutral rewrite clearly outperform the compact ALBERT baseline.

| Comparison | F1 | Errors |
| --- | ---: | ---: |
| `setup10` query + response | 0.989 | 42 |
| `setup10_1-qwen` query + response + Qwen neutral | 0.996 | 14 |
| `setup10_1_gemini` query + response + Gemini neutral | 0.996 | 14 |
| `setup10_1-gemma` query + response + Gemma neutral | 0.997 | 10 |

The neutral-aware Longformer variants reduce errors by roughly two thirds compared with `setup10`. The difference is large enough to be meaningful in this experiment. The result suggests that the neutral rewrite gives the fine-tuned model an explicit contrastive context: the model can compare the original response against a cleaned version and learn which additional phrases or sections are ad-like.

Gemma is the strongest neutral provider in the fine-tuned family, but the difference between the three `setup10_1` variants is small. All three are near ceiling.

### 2. Does the frozen delta substitute for fine-tuning?

No. The frozen embedding-LR models are far below the fine-tuned models.

| Best fine-tuned | F1 | Best frozen embedding-LR | F1 |
| --- | ---: | --- | ---: |
| `setup10_1-gemma` | 0.997 | `setup104-qwen` | 0.665 |

The frozen encoder does not learn task-specific token patterns. It relies on the pretrained embedding space to already encode the ad signal. In these current thesis-relevant runs, that assumption is too weak: the LR models have many hundreds of false positives and false negatives. Fine-tuning adapts the full encoder and is therefore much better suited to this task.

### 3. Does adding the neutral and delta improve frozen embedding-LR over response-only?

Yes, but only modestly.

| Setup | Input | F1 | Accuracy |
| --- | --- | ---: | ---: |
| `setup104-base` | response only | 0.605 | 0.737 |
| `setup104-gemma` | response + Gemma neutral + delta | 0.627 | 0.756 |
| `setup104-gemini` | response + Gemini neutral + delta | 0.640 | 0.772 |
| `setup104-qwen` | response + Qwen neutral + delta | 0.665 | 0.791 |

The stacked embedding representation improves over the response-only baseline for all providers. This means the neutral rewrite does contribute information to the frozen-feature setup. However, the improvement is not enough to make the frozen approach competitive with fine-tuning.

### 4. Is full stack better than residual-only in the frozen embedding family?

Usually yes, but the gain is provider-dependent.

| Provider | Residual-only | Full stack | Difference |
| --- | ---: | ---: | ---: |
| Qwen | `setup119-qwen` F1 0.647 | `setup104-qwen` F1 0.665 | +0.018 |
| Gemini | `setup119-gemini` F1 0.612 | `setup104-gemini` F1 0.640 | +0.028 |
| Gemma | `setup119-gemma` F1 0.617 | `setup104-gemma` F1 0.627 | +0.010 |

The full stack `[response_emb | neutral_emb | response_emb - neutral_emb]` consistently beats the pure residual `response_emb - neutral_emb`. Absolute response and neutral positions help the LR decision boundary. The effect is real but small compared with the gap to fine-tuning.

### 5. Which neutral provider works best?

The answer depends on the model family.

For fine-tuned Longformer models:

| Provider | Setup | F1 |
| --- | --- | ---: |
| Gemma | `setup10_1-gemma` | 0.997 |
| Qwen | `setup10_1-qwen` | 0.996 |
| Gemini | `setup10_1_gemini` | 0.996 |

For frozen embedding-LR full-stack models:

| Provider | Setup | F1 |
| --- | --- | ---: |
| Qwen | `setup104-qwen` | 0.665 |
| Gemini | `setup104-gemini` | 0.640 |
| Gemma | `setup104-gemma` | 0.627 |

For frozen residual-only models:

| Provider | Setup | F1 |
| --- | --- | ---: |
| Qwen | `setup119-qwen` | 0.647 |
| Gemma | `setup119-gemma` | 0.617 |
| Gemini | `setup119-gemini` | 0.612 |

Provider quality is therefore not absolute. Gemma gives the best fine-tuned result, but Qwen gives the strongest frozen embedding-LR result. The fine-tuned model can learn how to use the neutral text directly, while the frozen model depends more heavily on how the neutral rewrite changes the sentence embedding geometry.

---

## Summary Answer

| Question | Answer |
| --- | --- |
| Does the neutral rewrite help fine-tuning? | Yes. `setup10_1-*` strongly improves over `setup10`. |
| Does frozen delta-LR substitute for fine-tuning? | No. Best frozen F1 is 0.665; best fine-tuned F1 is 0.997. |
| Does neutral + delta improve frozen response-only embeddings? | Yes, but modestly: 0.605 -> 0.665 at best. |
| Is full stack better than residual-only? | Yes for all three providers, but only by 0.010-0.028 F1. |
| Which provider is best? | Gemma for fine-tuned models; Qwen for frozen embedding-LR models. |

The main conclusion is that **neutral rewrites are valuable when the model can learn from them end-to-end**. Fine-tuned Longformer models use the neutral context extremely well and reach near-ceiling performance. In contrast, the frozen semantic-delta approach provides a useful but weak signal in the current thesis-relevant runs. It improves over response-only embeddings, but it does not come close to replacing task-specific fine-tuning.

