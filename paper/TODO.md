# Paper TODO — what's still placeholder

Everything marked `[TBD]` in `main.tex` that blocks a real submission.

## Blocking (must fill before April 28 AoE)

### 1. Data from a real run
All of these require one pilot + eval to exist:

- [ ] `\tbd{headline numbers once the pilot run completes}` — abstract
- [ ] Table 1 cells (mean, fail easy/hard × 4 configs)
- [ ] Figure 2 — `figures/u_fail_smooth.pdf`
- [ ] Figure 3 — `figures/theme_fail.pdf`
- [x] Figure 1 — pipeline diagram source now lives in `figures/pipeline.tex`
- [ ] Prose interpretation in §4.2
- [ ] Ablation paragraph in §4.5 (Gemma-3n-E4B)
- [ ] One-sentence conclusion in §6
- [ ] Exact H100-hour count in §4.1 setup

### 2. Citations to verify
Most `references.bib` entries are placeholder metadata. Before submission, verify:

- [ ] `natmed2026triage` — pull authors + DOI from the actual Nature Medicine page
- [ ] `arora2025healthbench` — find the real OpenAI tech report citation
- [ ] `medgemma2026` — find Google's canonical MedGemma cite
- [ ] `bodhi2025` — confirm with Sebastian if a paper exists; if not, stay with the github.io cite
- [ ] `qwen25` — confirm arXiv number

### 3. Figures
- [x] `figures/pipeline.tex` — 5-box flow now checked into the repo as source, not a missing binary asset
- [ ] Copy `u_fail_smooth.png` → `figures/u_fail_smooth.pdf` after the pilot runs
- [ ] Copy `theme_fail.png` → `figures/theme_fail.pdf` after the pilot runs

### 4. Anonymization audit
- [ ] Confirm author block is `Anonymous`
- [ ] Code URL is `anonymous.4open.science` mirror, NOT `github.com/PeterLi-jpg/...`
- [ ] No mention of MIT, specific cluster names, or unique hardware
- [ ] Acknowledgments removed
- [ ] Grant numbers removed

## Nice-to-have if time permits

- [ ] Second grader family in appendix (addresses open issue #3)
- [ ] Seed variance (3 seeds for the headline LoRA config)
- [ ] Score-normalisation ablation comparing the three options in §5
- [ ] Appendix: full hyperparameter table
- [ ] Appendix: raw U-scatter figures (the 4-panel per-example view)
- [ ] Appendix: preflight check + KNOWN_ISSUES.md as a reproducibility statement

## Post-submission (archival full paper)

- [ ] Real author list
- [ ] De-anonymize repo link
- [ ] Longer Related Work with proper engagement
- [ ] Additional models (Llama-Med, Meditron) for generalisation
- [ ] Decide whether the logprob-based confidence proxy is strong enough for the final paper, or needs a richer uncertainty protocol
- [ ] Run the cross-grader pass and report the correlation summary for issue #3
- [ ] Normalisation decision for issue #4 + ablation

## Current blockers (external, not writing)

- Felipe needs to decide when he can start the E4B pilot on the MIT cluster
- AWS quota is still pending (Case 177646415700921) — not needed if Felipe runs on MIT
- Sebastian needs to confirm SD4H is the target (vs ML4H, NeurIPS, ICLR, NEJM AI)
- Methodology calls on the 3 open issues (#1, #3, #4) from `../KNOWN_ISSUES.md`
