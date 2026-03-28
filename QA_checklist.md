# QA checklist

- Functionality: Code implementation is functionally correct. Does the code handle path and edge cases correctly?
- Completeness: Is the solution for this phase complete?
- Consistency: Code and documentation style is consistent with the existing code/documentation. Is there anything hacky going on?
- Clarity: Does the solution include anything that could be removed? Is it overly complex? Are there reusable components anywhere we should be using?
- Guesswork: Is the implementation free of unverified assumptions? Check for guessed data schemas, file naming, or hidden dependencies.
- Documentation: Does documentation accurately reflect the code implementation?
- Testing: Do tests have adequate coverage? Do they cover the primary use cases, edge cases, and likely failure modes?
- Security: Are there any security related concerns?
- Review: What are the most likely things in this implementation to get flagged during code review?

## Current Notes

- The migration keeps the current Gemini generation behavior and the current binary classifier behavior.
- Utility-level tests cover merge logic, text cleanup, generated-field detection, and metric computation.
- Full integration testing still depends on a synced `uv` environment plus model/API access.
