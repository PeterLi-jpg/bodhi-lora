# Contributor docs

This folder is for people who work on the code. For what the repo is and how to
reproduce the paper, start at the [README](../README.md).

| Page | What it covers |
|---|---|
| [getting-started/installation.md](getting-started/installation.md) | Python envs, Hugging Face access, a check that the install works |
| [getting-started/quickstart.md](getting-started/quickstart.md) | the smoke test, and how to run one stage alone |
| [contributing.md](contributing.md) | hooks, tests, repo checks, pull requests, anonymous exports |
| [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md) | every cell in the paper, its config, and what lives outside the repo |
| [../docs/CONVENTIONS.md](../docs/CONVENTIONS.md) | the lab rules for this repo and how each one is enforced |

Two words, used the same way everywhere:

- **BODHI** is the two-pass chain-of-thought prompt that makes the demonstrations
  (Balanced, Open-minded, Diagnostic, Humble, and Inquisitive; Arslan et al., BMJ
  Health & Care Informatics, 2026). The repo slug `bohdi-lora` misspells it.
- **Epistemic behaviors** are the seven graded dimensions: uncertainty
  acknowledgment, hedging quality, specificity, active inquiry, context-seeking,
  red-flag identification and scope bounding.

The TPU, MaxText and tunix pages that used to live here described a path that this
branch deleted. They are in git history; KNOWN_ISSUES.md has a short note.
