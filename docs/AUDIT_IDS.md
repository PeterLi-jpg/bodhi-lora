# Audit IDs

The docs, the code comments and the paper source cite findings by ID, such as PROV-1 or
EVAL-7. Most come from the audit of 2026-10-07, which reviewed the repo, the paper and the
discussion record before the resubmission. That report is kept outside the repo,
because it names people and accounts. This file says what each cited ID found, so a
reader of the repo can follow a citation. It names no person.

Families: CLAIM (a claim in the paper or a reply), EVAL (evaluation design), HYG (repo
hygiene), PROV (provenance of a number), REPRO (reproducibility). The CR and NUM IDs come
from a later review of the paper text whose report is not kept; their lines below say what
the places that cite them use them for.

"Where" names the files that track the finding: [KNOWN_ISSUES.md](KNOWN_ISSUES.md) when it
lists the finding as open or fixed, otherwise the decision, claim or withdrawal that
settled it. Every file that cites an ID can be found with `git grep <ID>`.

## CLAIM

| ID | What the audit found | Where |
|---|---|---|
| CLAIM-1 | The paper's generality appendix still made four claims that the replies withdrew, among them "essentially unchanged" aggregate quality. | CLAIMS.md, CONVENTIONS.md |
| CLAIM-2 | The abstract and contributions still claimed preserved quality, eliminated blanket disclaimers and a general framework, and stated neither the quality trade-off nor format leakage. | CLAIMS.md, DECISIONS.md |
| CLAIM-3 | The paper still said physician co-authors shaped the dimensions for clinical decision support, a claim the replies corrected. | CLAIMS.md, DECISIONS.md (D4) |
| CLAIM-4 | Several competing copies of the paper, stamped "under review" after the reject, carried withdrawn claims. | DECISIONS.md |
| CLAIM-5 | No file in the repo was the submitted source, and READMEs claimed one was. | DECISIONS.md |
| CLAIM-7 | The title and abstract claimed calibration, but no calibration metric was reported. | CLAIMS.md, DECISIONS.md (D3) |
| CLAIM-8 | "Blanket disclaimers eliminated" rested on 5 events against 0 and was not significant. | CLAIMS.md, DECISIONS.md (D10) |
| CLAIM-9 | Equivalence and non-inferiority were inferred from non-significance, with two more overclaims. | CLAIMS.md, CONVENTIONS.md |
| CLAIM-10 | "n about 1,000 per condition" counted responses, not the 667 unique prompts. | KNOWN_ISSUES.md (open), DECISIONS.md (D7) |
| CLAIM-11 | The public docs contradicted the paper's train and eval separation. | KNOWN_ISSUES.md (fixed) |
| CLAIM-13 | The BODHI name, its expansion and the number of "virtues" drifted between texts. | DECISIONS.md (D15) |
| CLAIM-15 | Clarity fixes were applied to some copies of the paper and not others. | CONVENTIONS.md |

## EVAL

| ID | What the audit found | Where |
|---|---|---|
| EVAL-1 | "Quality preserved" rested on a non-significant test with no margin set in advance. | CLAIMS.md, DECISIONS.md (D2) |
| EVAL-5 | The physician validation ran differently from its plan, missed its gate, and left no grades or output in the repo. | KNOWN_ISSUES.md (open), DECISIONS.md (D5) |
| EVAL-6 | Physician names were in file names and instructions in the public repo. | CONVENTIONS.md, DECISIONS.md (D17, D18) |
| EVAL-7 | The Med42-8B cells break the rule that the evaluator comes from another family than the base. | KNOWN_ISSUES.md (open), DECISIONS.md (D6) |
| EVAL-8 | The promised inference-time baseline from prior work was never built. | KNOWN_ISSUES.md (open) |
| EVAL-9 | Grader parse failures were dropped, differed by condition, and contradicted the paper's "<0.1%". | KNOWN_ISSUES.md, DECISIONS.md (D13) |
| EVAL-10 | The discrimination labels and the leakage split were defined after their results existed. | KNOWN_ISSUES.md (open), ANALYSIS_PLAN.md |
| EVAL-11 | Table 1 mixed interval methods across rows. | CONVENTIONS.md |
| EVAL-12 | The contamination and overlap checks had no committed outputs. | KNOWN_ISSUES.md, `bodhi_lora/data/leakage.py` |
| EVAL-13 | A Bonferroni correction was stated with no family size and no code. | DECISIONS.md |

## HYG

| ID | What the audit found | Where |
|---|---|---|
| HYG-1 | Author names and an email were in the public history of the paper source. | CONVENTIONS.md (rule 10), DECISIONS.md |
| HYG-4 | The paper's LoRA settings did not match what the submitted pipeline ran. | KNOWN_ISSUES.md (open), DECISIONS.md (D1) |
| HYG-5 | No branch could reproduce both the submitted and the added numbers. | DECISIONS.md |
| HYG-9 | Seven copies of the paper source, in five versions. | DECISIONS.md (D8), `paper/README.md` |
| HYG-11 | Model weights and a large trace file were in public history. | KNOWN_ISSUES.md (open) |
| HYG-12 | Scratch branches and AI-tool traces were on the public remote. | KNOWN_ISSUES.md, BLOCKED.md item 4 |
| HYG-13 | Other identifying strings (account and host names, local paths, bucket names) were in committed files. | CONVENTIONS.md, `scripts/export_anonymous.sh` |
| HYG-14 | Docs described a deleted pipeline, so a check that doc paths exist was added. | `scripts/check_doc_paths.py` |
| HYG-15 | Most local and remote branches were never merged. | KNOWN_ISSUES.md (open) |

## PROV

| ID | What the audit found | Where |
|---|---|---|
| PROV-1 | The added cells' numbers existed only as posted prose; no raw output was in the repo. | KNOWN_ISSUES.md, `rebuttal/README.md` |
| PROV-2 | The paired t-tests and Holm correction behind the posted drops had no code. | KNOWN_ISSUES.md (fixed) |
| PROV-3 | "Every hyperparameter fixed (tau = 0.4)" did not match the launcher, which derived tau for some cells. | KNOWN_ISSUES.md (fixed), `rebuttal/withdrawals.md` |
| PROV-4 | "n around 985 to 998 per condition" was wrong for both CoT conditions. | `rebuttal/withdrawals.md` (W10) |
| PROV-5 | The kappa of 0.35 could not be recomputed from the repo. | KNOWN_ISSUES.md (open) |
| PROV-6 | The training-loss figure came from an early run, not the reported one. | DECISIONS.md (D14) |
| PROV-7 | The posted second-person question rates had no script or output. | CLAIMS.md, `rebuttal/withdrawals.md` (W23) |
| PROV-8 | "Consistent with non-inferiority" was posted without a test, and a later paired test contradicted it. | CLAIMS.md, DECISIONS.md |
| PROV-9 | No script produced the paper's p-values and interval tables. | CONVENTIONS.md (rule 2) |
| PROV-10 | The posted odds ratios were computed by hand, and their outputs were gitignored. | `rebuttal/README.md` |
| PROV-11 | The trace counts per seed could be traced for seed 7 only. | `paper/NUMBERS.md`, EXPERIMENTS.md |

## REPRO

| ID | What the audit found | Where |
|---|---|---|
| REPRO-3 | Generation and eval settings differed between the submitted and the added cells. | KNOWN_ISSUES.md (open), REPRODUCIBILITY.md |
| REPRO-6 | There was no one-command path from results to the paper's tables. | DECISIONS.md, `.github/workflows/ci.yml` |
| REPRO-7 | Results had no manifest of the code, settings and versions that made them. | CONVENTIONS.md, `bodhi_lora/manifest.py` |
| REPRO-8 | Seeds never varied the teacher traces, and varied different things in different cells. | KNOWN_ISSUES.md (open), EXPERIMENTS.md (E23) |
| REPRO-10 | The dependency and model pins disagreed, and no environment was frozen. | KNOWN_ISSUES.md, BLOCKED.md item 7 |
| REPRO-11 | The ChatDoctor and MedQuAD data were not pinned and used a first-rows holdout. | KNOWN_ISSUES.md (fixed) |
| REPRO-12 | The Mistral-Small-24B adapters were trained on a 4-bit base and evaluated on a bf16 base. | KNOWN_ISSUES.md (open), the cell records |

## CR and NUM (the later review)

| ID | What the citing places use it for | Where |
|---|---|---|
| CR-4 | Prior work on asking and abstaining was missing from the references. | `paper/references.bib` |
| CR-8 | The datasets, models and methods the paper uses were not all cited. | `paper/references.bib` |
| CR-19 | The share of replies that ask is reported with a stated better direction. | `bodhi_lora/report/cell.py` |
| NUM-2 | The physician kappa is computed from an input that is kept outside the repo. | `paper/NUMBERS.md` |
| NUM-4 | The format-leakage split counted with the seven Pass-1 headers alone. | `rebuttal/analyses/interference_mechanism.py` |
| NUM-7 | Whether a base model completes benchmark prompts verbatim more often than control text. | EXPERIMENTS.md (E11), RUN_PLAN.md (R11) |
