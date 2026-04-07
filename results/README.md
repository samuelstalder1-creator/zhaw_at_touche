# Results Directory

This directory stores evaluation artifacts written by `touche-validate`,
`touche-embed-divergence`, and `touche-pairwise-distances`.

## Committed Result Directories

The repository currently contains committed result artifacts for:

- `results/setup6/`
- `results/setup6-qwen/`
- `results/setup8/`
- `results/setup10/`
- `results/setup12/`
- `results/setup100/`
- `results/setup101/`
- `results/setup102/`
- `results/setup103/`
- `results/setup104/`
- `results/setupX/` as a placeholder/example directory

## Typical Validation Outputs

Classifier and embedding-divergence validation runs usually write:

- `metrics_summary.json`
- `response_metrics.txt`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `standardized_results.csv`
- `misclassified_analysis.csv`
- `*-predictions.jsonl`

`metrics_summary.json` is the canonical machine-readable summary used by
`results.md`.

## Pairwise Distance Outputs

`touche-pairwise-distances` writes a different result shape:

- `pairwise_distances.jsonl`
- `pairwise_distances.csv`
- `pairwise_summary.json`

## Notes

- A committed result directory does not necessarily mean the matching model
  bundle is committed under `models/`.
- Some committed results, such as `setup103` and `setup104`, correspond to
  archived experimental setup descriptors that are documented in `../setup.md`.
