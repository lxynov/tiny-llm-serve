---
name: git-commit-style
description: This skill should be used whenever drafting a git commit message for this repo (e.g. before running `git commit`, when the user asks to "commit this", "write a commit message", or discusses commit formatting/conventions here).
version: 1.0.0
---

# Git Commit Style

## Subject Line

Format: `[Type] Subject`

- **Type**: one of `Feature`, `Fix`, `Refactor`, `Docs`, `Test`, `Chore`, `Perf`, `Style`, `Revert` — pick the one that best matches the primary intent of the change.
- **Subject**: imperative mood ("Add", "Fix", "Remove" — not "Added"/"Fixes"), capitalized first word, no trailing period.
- Keep the whole subject line under ~72 characters.

Examples:
```
[Fix] Correct off-by-one error in token counting
[Feature] Add streaming support to the completion endpoint
[Refactor] Extract request validation into its own module
[Docs] Clarify setup instructions in README
```

## Body (optional)

- Blank line after the subject.
- Wrap at ~72 characters.
- Explain *why* the change was made, not what — the diff already shows what changed.
- Use bullet points for multiple distinct reasons/changes.

## Choosing a Type

- `Feature` — new user-facing or API capability
- `Fix` — bug fix
- `Refactor` — code restructuring with no behavior change
- `Docs` — documentation only
- `Test` — tests only
- `Chore` — tooling, deps, build/config changes
- `Perf` — performance improvement
- `Style` — formatting/whitespace, no logic change
- `Revert` — reverts a previous commit

If a commit spans multiple types, pick the type matching its dominant purpose rather than combining types in one subject line.
