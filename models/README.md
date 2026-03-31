# Models

This directory stores trained Hugging Face model bundles.

Suggested convention:

- `models/setupX/`
- `models/setupY/`
- `models/setup6/`
- `models/setup9/`
- `models/setup10/`
- `models/setup11/`
- `models/setup12/`

Each setup directory is created automatically by `uv run touche-train --setup-name <name>`.

Only `models/setup6/` is currently committed in the repository. The newer setup
directories appear after local training runs.
