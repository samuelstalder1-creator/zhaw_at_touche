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
  families and the remaining descriptor-only sentence-delta experiment.
- Current runnable training families are:
  - classifier: `setup4`, `setup6`, `setup6-qwen`, `setup7`, `setup7-qwen`,
    `setup8`, `setup9`, `setup10`, `setup11`, `setup12`, `setup115`,
    `setup116`
  - cross-encoder: `setup105`, `setup105_1`
  - learned embedding features: `setup103`, `setup104`, `setup113`,
    `setup114`, `setup117`, `setup118`, `setup119`
  - embedding divergence: `setup100`, `setup101`, `setup102`
  - scalar anchor: `setup110`, `setup111`
- Descriptor still documented in the repo:
  - `setup106`
- Full integration testing still depends on a synced `uv` environment plus
  model and API access.
