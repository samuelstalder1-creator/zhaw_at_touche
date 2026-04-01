# Setup Comparison Summary

This file compares the named training setups currently defined in the repository: `setup4`, `setup6`, `setup7`, `setup8`, `setup9`, `setup10`, `setup11`, and `setup12`.

## Quick Comparison

| Setup | Backbone | Input format | Reference text | Max length | Epochs | Batch x accum | Effective batch | Main idea |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| `setup4` | `microsoft/deberta-v3-base` | `query_reference_rag_response` | `gemini25flashlite` as `Unbiased Reference` | 512 | 6 | `16 x 4` | 64 | Most tuned DeBERTa setup with reference-aware input |
| `setup6` | `FacebookAI/roberta-base` | `query_response` | none | 512 | 3 | `16 x 4` | 64 | Simple RoBERTa baseline |
| `setup7` | `allenai/longformer-base-4096` | `query_neutral_response` | `gemini25flashlite` as `GEMINI` | 1024 | 1 | `4 x 8` | 32 | Long-context setup with Gemini neutral reference |
| `setup8` | `microsoft/deberta-v3-base` | `query_response` | none | 512 | 3 | `16 x 4` | 64 | `setup6` recipe with DeBERTa-v3 |
| `setup9` | `microsoft/deberta-v3-base` | `query_response` | none | 512 | 3 | `16 x 4` | 64 | Stabilized DeBERTa-v3 retry for `setup8` |
| `setup10` | `albert/albert-base-v2` | `query_response` | none | 512 | 5 | `16 x 4` | 64 | Lightweight ALBERT baseline with linear schedule |
| `setup11` | `google/electra-base-discriminator` | `query_response` | none | 512 | 4 | `16 x 4` | 64 | ELECTRA discriminator baseline with linear schedule |
| `setup12` | `distilroberta-base` | `query_response` | none | 512 | 5 | `16 x 4` | 64 | DistilRoBERTa baseline with linear schedule |

## Key Differences

| Setup | Notable settings |
| --- | --- |
| `setup4` | `optimizer_eps=1e-06`, `lr_scheduler=cosine_with_warmup`, `warmup_ratio=0.05`, `max_grad_norm=1.0`, `gradient_checkpointing=true`, `device=cuda`, `positive_class_weight_scale=1.0` |
| `setup6` | Uses mostly global defaults beyond the basic training hyperparameters |
| `setup7` | `pad_to_max_length=true`, `positive_class_weight_scale=1.5`, longer context window, smaller micro-batch |
| `setup8` | Same behavior as `setup6` except `model_name=microsoft/deberta-v3-base` |
| `setup9` | `learning_rate=8e-06`, `optimizer_eps=1e-07`, `weight_decay=0.01`, `lr_scheduler=cosine_with_warmup`, `warmup_ratio=0.1`, `max_grad_norm=1.0`, `layerwise_lr_decay=0.9`, `freeze_embeddings_epochs=1` |
| `setup10` | `learning_rate=3e-05`, `weight_decay=0.01`, `lr_scheduler=linear`, `warmup_ratio=0.06`, `device=cuda` |
| `setup11` | `learning_rate=2e-05`, `weight_decay=0.01`, `lr_scheduler=linear`, `warmup_ratio=0.06`, `max_grad_norm=1.0`, `device=cuda` |
| `setup12` | `learning_rate=3e-05`, `weight_decay=0.01`, `lr_scheduler=linear`, `warmup_ratio=0.06`, `device=cuda` |

## Validation And Artifact Status

| Setup | Validation preset file | Expected model dir | Expected results dir | Committed model dir | Committed results dir |
| --- | --- | --- | --- | --- | --- |
| `setup4` | `validate_model/setup4.json` | `models/setup4/` | `results/setup4/` | no | no |
| `setup6` | none | `models/setup6/` | `results/setup6/` | yes | yes |
| `setup7` | `validate_model/setup7.json` | `models/setup7/` | `results/setup7/` | no | no |
| `setup8` | none | `models/setup8/` | `results/setup8/` | no | no |
| `setup9` | `validate_model/setup9.json` | `models/setup9/` | `results/setup9/` | no | no |
| `setup10` | `validate_model/setup10.json` | `models/setup10/` | `results/setup10/` | no | no |
| `setup11` | `validate_model/setup11.json` | `models/setup11/` | `results/setup11/` | no | no |
| `setup12` | `validate_model/setup12.json` | `models/setup12/` | `results/setup12/` | no | no |

## Committed Results

Only `setup6` currently has committed evaluation artifacts in the repository, so it is the only setup with numbers that can be compared directly right now.

| Setup | Accuracy | Positive precision | Positive recall | Positive F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `setup4` | n/a | n/a | n/a | n/a | no committed results |
| `setup6` | 0.3071 | 0.3071 | 1.0000 | 0.4699 | predicted label `1` for every row |
| `setup7` | n/a | n/a | n/a | n/a | no committed results |
| `setup8` | n/a | n/a | n/a | n/a | no committed results |
| `setup9` | n/a | n/a | n/a | n/a | new stabilized DeBERTa config, no committed results yet |
| `setup10` | n/a | n/a | n/a | n/a | new ALBERT config, no committed results yet |
| `setup11` | n/a | n/a | n/a | n/a | new ELECTRA config, no committed results yet |
| `setup12` | n/a | n/a | n/a | n/a | new DistilRoBERTa config, no committed results yet |

Overall confusion counts for committed `setup6` results:

- True negatives: `0`
- False positives: `8315`
- False negatives: `0`
- True positives: `3685`

## How To Read The Input Formats

- `query_response`: model sees only the query and the response to classify
- `query_neutral_response`: model sees the query, a Gemini neutral reference, and the response to classify
- `query_reference_rag_response`: model sees the query, an explicit reference answer, and the RAG response to classify

## Fast Takeaways

- `setup6` and `setup8` are the cleanest head-to-head comparison because they use the same recipe and differ only in backbone model
- `setup9` is the practical follow-up to `setup8`: same DeBERTa backbone and prompt format, but with the lower LR and optimizer stabilizers aimed at reducing divergence
- `setup10` gives you a lighter non-DeBERTa alternative with standard `query_response` input and a linear schedule
- `setup11` gives you an ELECTRA discriminator baseline with a standard prompt format and linear scheduling
- `setup12` gives you a smaller RoBERTa-family baseline that should train faster than full RoBERTa
- `setup4` is the more engineered DeBERTa variant and is not directly comparable to `setup6` or `setup8` on architecture alone because the prompt format and optimization settings also change
- `setup7` is the outlier: longest context, smallest per-device batch, and reference-aware input
- The repository does not yet contain committed results for `setup4`, `setup7`, `setup8`, `setup9`, `setup10`, `setup11`, or `setup12`, so any real performance comparison still requires training and validation for those setups

## Experimental Setup100

`setup100` is intentionally not part of the classifier training table above because it does not fine-tune a transformer classifier.

- Train command: `uv run touche-train --setup-name setup100`
- Eval command: `uv run touche-validate --setup-name setup100` or `uv run touche-embed-divergence --setup-name setup100`
- Type: embedding-space divergence baseline with a saved threshold/state phase
- Embedding model: `sentence-transformers/all-mpnet-base-v2`
- Reference field: `gemini25flashlite`
- Score: sentence-level cosine divergence with greedy alignment and `mean` aggregation
- Training artifact: `models/setup100/embedding_state.json`
- Thresholding: `touche-train` fits the threshold on validation with `macro_f1` by default, and evaluation reuses the saved threshold and scoring config unless you override them on the CLI

The goal is to treat the neutral response as a semantic anchor and score how far
the RAG response drifts away from it, rather than training a classifier on the
combined prompt.
