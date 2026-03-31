# Data Layout

- `task/`: official Touché dataset files plus `preprocessed/` outputs from `touche-preprocess`
- `generated/gemini/`: Gemini-generated neutral response files
- `generated/qwen/`: self-hosted Qwen-generated neutral response files
- `generated/chatgpt/`: reserved for the same file format if hosted OpenAI-generated data is added later

The original dataset description is in `task/README.md`.

The current named training setups use the Gemini-generated `responses-*-with-neutral_gemini.jsonl`
files as their default training, validation, and evaluation inputs when those
files are present. Qwen-generated files follow the same layout under
`data/generated/qwen/`.
