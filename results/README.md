# Results

This directory stores validation and evaluation artifacts.

Suggested convention:

- `results/setupX/`
- `results/setup6/`
- `results/setup9/`
- `results/setup10/`
- `results/setup11/`
- `results/setup12/`

Typical outputs:

- prediction JSONL files
- confusion matrix CSV / PNG
- metrics JSON / TXT
- misclassification CSV exports

Only `results/setup6/` is currently committed in the repository. The newer
setup result directories are generated on demand by `touche-validate`.
