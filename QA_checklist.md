# QA Checklist

- Functionality: Does the implementation handle the expected paths and edge
  cases correctly?
- Completeness: Is the requested work complete for this phase?
- Consistency: Are code, configs, results, and docs aligned with one another?
- Clarity: Is there unnecessary complexity or duplication?
- Guesswork: Are there any unverified assumptions about schemas, files, or
  setup support?
- Documentation: Do the Markdown files reflect the current code and artifact
  state?
- Testing: Do the relevant unit tests cover the changed behavior?
- Security: Are there any obvious security concerns?
- Review: What is most likely to get flagged during code review?

## Current Notes

- The repo now treats `setup.md` as the canonical setup reference.
- Documentation now explicitly distinguishes between currently supported setup
  families and archived experimental descriptors.
- Current runnable training families are:
  - classifier: `setup4`, `setup6`, `setup6-qwen`, `setup7`, `setup7-qwen`,
    `setup8`, `setup9`, `setup10`, `setup11`, `setup12`
  - embedding divergence: `setup100`, `setup101`, `setup102`
- Archived setup descriptors still documented in the repo:
  - `setup103`, `setup104`, `setup105`, `setup106`
- Full integration testing still depends on a synced `uv` environment plus
  model and API access.
