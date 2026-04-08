# Results Directory

This directory stores evaluation artifacts written by `touche-validate` and
`touche-embed-divergence`.

## Committed Result Directories

The repository currently contains committed result artifacts for:

- `results/setup6/`
- `results/setup6-qwen/`
- `results/setup7-qwen/`
- `results/setup8/`
- `results/setup10/`
- `results/setup12/`
- `results/setup100/`
- `results/setup101/`
- `results/setup102/`
- `results/setup103/`
- `results/setup104/`
- `results/setup105/`
- `results/setup105_1/`
- `results/setup106/`
- `results/setup110/`
- `results/setup111/`
- `results/setup113/`
- `results/setupX/` as a placeholder/example directory

## Typical Validation Outputs

Classifier, cross-encoder, learned embedding-feature, embedding-divergence, and
anchor-distance validation runs usually write:

- `metrics_summary.json`
- `response_metrics.txt`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `standardized_results.csv`
- `misclassified_analysis.csv`
- `*-predictions.jsonl`

`metrics_summary.json` is the canonical machine-readable summary used by
`results.md`.

For `setup110` and `setup111`, the prediction JSONL rows also include the
pair-distance feature values. `setup111` additionally exposes the derived
`anchor_cohesion` and `response_drift` components of the handcrafted score.

## Notes

- A committed result directory does not necessarily mean the matching model
  bundle is committed under `models/`.
- `setup106` is the remaining descriptor-only setup with a historical committed
  artifact; the other committed learned embedding-feature results correspond to
  active backends documented in `../setup.md`.
