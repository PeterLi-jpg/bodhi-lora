# Experiments

The experiment matrix for the resubmission, written on 2026-10-08. It answers every
reason the NeurIPS 2026 submission 32818 was rejected, and every improvement the
reviewers suggested or the posted replies promised. Each item names the experiment or
analysis that answers it, whether it needs a GPU, the cells it uses, and its status.
[ARCHITECTURE.md](ARCHITECTURE.md) gives the design the items run on.

No item here runs without the first author's go-ahead. No GPU work starts until he says
so (2026-10-08). The two remaining physician raters are deferred.

## 1. How to read this

**Status.**

| Word | Meaning |
|---|---|
| exists | The code and an output exist in the repo. Uniform analysis may still have to recompute it. |
| to build | Code must be written. CPU only, unless the GPU column says yes. |
| to run | The code exists or the build lanes write it; the run needs a GPU or people. |
| deferred | Waits on people, by the first author's decision. |
| text | Wording only. Done items cite the commitment (`rebuttal/commitments.md`). |

**Cells** use the ids of ARCHITECTURE.md section 2: `<benchmark>__<model>__<protocol>`,
with `__2026-05` or `__2026-07` for the existing runs. `<HB-new>` means the new
HealthBench cells of E13.

**Costs** are rough estimates, not measurements. The base figure is the posted one:
about two H100-hours per 24B seed end to end (`rebuttal/responses/posted/FzpA_1_rebuttal.md`),
so about 10 H100-hours for a five-seed 24-27B cell and about 4 for a 7-8B cell. E20
measures the real values.

## 2. What the reviewers asked

Short labels for each concern. The full text is in the OpenReview thread and in
`rebuttal/responses/posted/`.

| ID | Concern |
|---|---|
| MR-W1 | MedGemma does not fit HealthBench's use case. |
| MR-W2 | One pipeline is tested, but the contribution is stated as a general framework. |
| MR-W3 | Automated grading is limited, so the results may not be robust. |
| MR-W4 | It is unclear whether SFT teaches calibration or mimicry of favourable replies. |
| MR-W5 | Discussion and context of the results lack clarity. |
| MR-W6 | No comparison with prior work. |
| MR-F | Final: limited baselines, narrow protocols and base models; the discussion changed the scope; the paper needs a full new review. |
| FzpA-W1 | "Framework" overstates one instantiation; say which tasks suit it and how the decomposition changes across contexts. |
| FzpA-W2 | MedGemma does not fit HealthBench; use a widely used model or a small variant. |
| FzpA-W3 | Clarity: specific clinical implications, a Table 2 arrow, the mixed scales of Figure 2, vague "not" statements. |
| FzpA-S | Strength with a request: compare training and inference cost per setup explicitly. |
| FzpA-Q1 | Where the seven dimensions came from; clinician input. |
| FzpA-Q2 | How each dimension looks in text. |
| FzpA-Q3 | One benchmark is not enough; test clinician-posed questions (for example from MIMIC or eICU). |
| FzpA-F1 | Follow-up: clinical decision support criteria applied to patient-facing tools; formal grounding and validation of the criteria. |
| FzpA-F2 | Follow-up: specific clinical implications; how the recipe is used. |
| FzpA-F3 | Follow-up: a concrete example of a "qualified statement". |
| FzpA-final | The use case is not scoped; the rebuttal volume needs a full new review. |
| LxMF-1 | Why sample sizes differ (200, 191, 192). |
| LxMF-2 | Calibration or surface template: compare prompts that lack information with prompts that do not. |
| LxMF-3 | Why a Llama judge; consider medical judges (Meditron-3, Aloe, Med42-v2). |
| LxMF-4 | The CoT and LoRA "competition" is speculative; truncation or long outputs may explain the red-flag drop. |
| LxMF-C | The dual-axis figure misleads. |
| HomM-W1 | One model, one benchmark, one CoT protocol. |
| HomM-W2 | Expected extensions: more models, benchmarks, human evaluation. |
| HomM-W3 | Subjective dimensions graded by non-medical LLMs. |
| HomM-W4 | No comparison with prior methods. |
| HomM-Q1, Q2 | Reasons for the narrow evaluation; one or two more base models or one more clinical set. |
| HomM-Q3 | Compare with other calibration or uncertainty protocols. |
| HomM-Q4 | Why the graders are not clinically tuned. |
| HomM-F1 | Follow-up: what "scope condition" means for BioMistral. |
| HomM-F2 | Follow-up: the convergence claim is not supported. |
| HomM-F3 | Follow-up: the tables behind the prose values. |
| HomM-F4 | Follow-up: discuss clinically tuned against general judges as a limitation. |
| HomM-F5 | Follow-up: an empirical comparison with one or two of the closest published methods. |
| HomM-final | Open: human validation, prior-work comparison, a second protocol, consistent aggregate drops and when they are worth it, claims corrected during the discussion. |
| C1 to C33 | Every promise in the posted replies (`rebuttal/commitments.md`). |

## 3. Experiments and analyses

### Process

**P1. Dated analysis plan.** Answers MR-F, HomM-final (claims built during the
discussion), convention 3. Fix the metrics, unit, tests, correction families, contrasts,
label definitions and any margin before a new run, and before a new statistic is computed
on existing outputs. BLOCKED.md item 5 has the proposed defaults. GPU: no. Status: to
write (the first author's methodology call). Everything below that is "to run", and E02,
depends on it.

**P2. Citation check before submission.** The program chairs' checker flagged 0 of 22
references. Run the `.bib` check again on the resubmission bibliography. GPU: no.
Status: to run (CPU).

### Analyses of the existing outputs (no GPU)

These run on the eight existing cells with outputs: the 2026-05 MedGemma cell and the
seven 2026-07 cells (the MedQA preamble cell set aside, Med42-8B cells under a family
waiver).

**E01. Uniform per-cell analysis.** Answers MR-W3, MR-W5, LxMF-1, HomM-F3, D2, D7, D10,
D13. Every statistic of ARCHITECTURE.md section 7 for every cell: seven dimensions and
rates with seed-level paired t(4) intervals and Holm over the seven, pooled d with
prompt-clustered bootstrap intervals, counts and parse failures for both graders, worst
case, blanket disclaimers with Fisher, length, askers gap, rubric axes, dimension
correlations, and one drawn example per score level per dimension (FzpA-Q2). The
spreads of base and adapted active inquiry are reported without any convergence claim
(HomM-F2, W4). Today only the 2026-05 cell has the full set (`scripts/paper_stats.py`);
the 2026-07 cells have a subset (`rebuttal/analyses/cell_values.py`). Status: to build
(analysis lane); data exists.

**E02. Physician-label discrimination index.** Answers MR-W4, LxMF-2, HomM-W3. HealthBench
physicians labelled prompts by whether context is missing
(`physician_agreed_category:not-enough-context`, `not-enough-info-to-complete-task`,
`context-matters-but-unclear` against `enough-context`, `enough-info-to-complete-task`,
`context-matters-is-clear`, `context-does-not-matter`). HealthBench-Hard holds 204 such
positives and 113 negatives. The graders never see these labels, the labels predate
this study, and physicians made them, which answers the post-hoc objection to the two
labellings of the discussion (D9). Statistic: P(ask given positive) minus P(ask given
negative) per condition, odds ratio, prompt-clustered and seed-level intervals, and LoRA
minus Base. Cells: the four HealthBench cells of 2026-05 and 2026-07. The two earlier
labellings are reported next to it, marked post hoc. GPU: no. Status: to build; must be
in P1 before it is computed.

**E03. Grader-free question rate.** Answers MR-W3, HomM-W3, PROV-7. Share of responses
with a second-person question, for every cell and condition. Exists for the 2026-05
cell (`rebuttal/analyses/second_person_questions.py`, which gives 17.9% against 36.7%,
not the posted 18.4% against 39.6%). GPU: no. Status: to build (uniform).

**E04. Format leakage for every cell.** Answers LxMF-4, D11. For each leak rule of the
protocol used at inference: leak share per condition, the seven dimensions by leak
status, red flag by length quartile in non-leaked responses. With `finish_length`
recorded in new runs, truncation is measured directly instead of inferred. Exists for
the 2026-05 cell (`rebuttal/analyses/interference_mechanism.py`). GPU: no. Status: to
build (uniform).

**E05. What the aggregate drop is made of.** Answers HomM-final (consistent aggregate
drops), withdrawals W7 and W8. Per cell: LoRA minus Base on each HealthBench rubric axis,
the askers gap, and the cross-cell Holm family of the aggregate drops. Exists in part
(`rebuttal/analyses/quality_drop.py`, the askers gap in `cell_values.py`). GPU: no.
Status: to build (uniform).

**E06. Harm proxy.** Answers C33 (in part), HomM-final, FzpA-F2. Share of responses that
meet at least one negative-point rubric criterion, and the points lost to them, per
condition, with seed-level contrasts. HealthBench rubrics carry physician-written
negative criteria (4,133 in Hard); the synthesized rubric has two (dangerous advice,
false reassurance). This is an LLM-graded proxy, not the physician harm review (E28).
GPU: no. Status: to build.

**E07. Deployment-risk criterion.** Answers C10 (the text is done), HomM-final ("when the
gains justify the loss"), FzpA-F2. A rule per cell, fixed in P1: prefer the adapter over
the base for a setting when (a) the harm proxy (E06) does not rise beyond a margin,
(b) red-flag identification does not fall beyond a margin, (c) asking rises on prompts
that lack information (E02, E14), and (d) over-flagging (E18) does not rise beyond a
margin. The analysis reports each part with its interval. The margins are the first
author's decision (open decision O5); without them it reports intervals only. GPU: no.
Status: to build.

**E08. Cost from the outputs.** Answers FzpA-S, C25 (in part), LxMF-1. Calls,
generated tokens and characters per response and condition, and completions lost to
the context window. The 2026-05 files hold `response_token_count`; new runs add calls
and tokens over all passes. GPU: no. Status: to build. E20 adds measured time.

### Inference only (GPU, no training)

**E09. Cross-family re-grade of the Med42-8B cells.** Answers D6, EVAL-7, LxMF-3,
HomM-W3. Re-grade the stored outputs of `healthbench__med42-8b__bodhi__2026-07` and
`chatdoctor__med42-8b__bodhi__2026-07` with a judge outside Llama, Qwen and the base's
family (Gemma-2-27B by default; MedGemma-27B is the clinical choice). Report per-criterion
kappa against the evaluator and the LoRA minus Base contrasts under the judge. The dagger
stays. GPU: yes, about 2 to 5 H100-hours per cell, under 1 with the dimensions only
(BLOCKED.md item 7). Code: `scripts/regrade_cross_family.py`, dry-run tested. Status: to
run.

**E10. Clinical-judge robustness panel.** Answers C26, LxMF-3, HomM-Q4, HomM-W3,
HomM-F4, MR-W3. Re-grade every cell's stored outputs with clinical judges that pass the
family rule for that cell (R3: not the base's family, not the filter's). Candidates:
Med42-8B and Meditron3-8B (Llama-3 family, clinical; they compare clinical and general
grading at a fixed family), Aloe-Beta-8B (Llama-3.1 based; the Qwen-based Aloe models
fail R3), and MedGemma-27B (Gemma, clinical). Per judge: per-criterion kappa against the
evaluator, weighted kappa on the 0 to 2 dimensions, Pearson r of the rubric score, and
every contrast recomputed. It does not replace the evaluator. Cells: every existing cell
and every new one. GPU: yes, about 1 to 5 H100-hours per judge and cell. Status: to build
(judge lists in the grade stage), then to run. Meditron3-8B has no pinned revision yet
(BLOCKED.md item 12), and Aloe-Beta-8B has no model record yet.

**E11. Contamination probe for every base model and benchmark.** Answers EVAL-12,
NUM-7. Does the base complete benchmark prompts verbatim more often than a control text?
`scripts/contamination_probe.py` exists for HealthBench and the 2026-05 base; its output
was never kept. Generalize the control text per benchmark, run it for Mistral-Small-24B,
Med42-8B, BioMistral-7B, Mistral-7B-Instruct and MedGemma-27B on each benchmark they are
evaluated on, and keep the JSON. GPU: yes, about 1 H100-hour per model and benchmark.
Status: to build, then to run.

**E12. Prior-work baselines.** Answers MR-W6, HomM-W4, HomM-Q3, HomM-F5, C27, C14, and
the open item the LxMF rebuttal named. Three inference-time baselines on the unchanged
base, on each cell's eval prompts, graded like every other condition:
- `base+clarify`: a clarifying-question instruction. Exists (`scripts/baselines/prompts.py`).
- `base+verbalized-confidence`: state confidence in words and as a number, and what
  would change the answer (Lin et al., 2022; Tian et al., 2023). Exists.
- `base+mediq-abstain`: a MediQ-style abstention step (Li et al., NeurIPS 2024). Pass 1
  rates on a five-level scale, with a short rationale, whether the information is enough
  to answer safely; if it is below the threshold, pass 2 asks the most important question,
  otherwise it answers. This is the paper's Scale plus rationale variant without
  self-consistency, so it costs two calls like BODHI. To build.
Contrasts: LoRA minus each baseline on the seven dimensions and the rubric score;
each baseline minus Base on active inquiry; and the discrimination index (E02, E14)
for each baseline, because a prompt can raise asking without targeting it. Cells: every
new cell (E13, E16, E25); optional on the 2026-07 cells, which pair with their stored
Base rows only if the grader repo and the serving settings match the cell record. GPU:
yes, about 2 to 3 H100-hours per 24-27B cell for the two single-pass baselines
(BLOCKED.md item 6), more with the two-pass one. Status: two exist (dry-run tested), one
to build; all to run.

**E14. MediQ paired probe: calibration against mimicry.** Answers MR-W4, LxMF-2 directly.
MediQ (Li et al., NeurIPS 2024; repository licence CC BY 4.0) turns MedQA vignettes into
interactive cases. For iMedQA the development set has 1,272 cases, each split into atomic
facts, with "initial information" of age, sex and chief complaint. Each case gives a pair
of open-ended prompts with the options removed: complete (all facts) and incomplete
(initial information only). The label comes from the construction, not from us or a
grader. Calibration predicts that the adapter asks more on the incomplete prompt than on
its complete twin, by more than the base does; mimicry predicts that asking rises on both.
Statistic: within-pair difference in asking (evaluator and grader-free count), seed-level
paired t, McNemar per condition, and the LoRA minus Base difference in differences.
Draw 200 pairs per seed. Cells: eval-only cells with `adapters_from` the E13 and E16
cells, for example `mediq-paired__mistral-small-24b__bodhi` (benchmark record
`mediq-paired`, loader `mediq_paired`). MediQ cases come from MedQA, so check overlap
with any MedQA-trained cell (the development split against our train split). GPU: yes,
about 2 to 4 H100-hours per adapter cell. Status: to build (loader, benchmark record),
then to run.

### New training runs (GPU)

**E13. The uniform HealthBench grid.** Answers MR-W1, MR-W2, MR-F, HomM-W1, HomM-Q1,
HomM-Q2, FzpA-W2, C9, D1, REPRO-3, REPRO-8, REPRO-12, PROV-11. One set of settings for
every new cell (proposed, to fix in P1):

| Setting | Proposed value | Differs from |
|---|---|---|
| Split | HealthBench `default`: evaluate on Hard; the emergency-labelled prompts (both groups, 397) and the clinician-user prompts (282) also leave the 4,000-row train pool, which keeps 3,321 | 2026-05 and 2026-07: train on all 4,000 |
| Eval ids | a separate random 200 of the eval pool per seed | 2026-07: one fixed set |
| Recipe | LoRA r 16, alpha 32, all seven projections, 3 epochs, effective batch 16, cosine with 5% warmup, sequence 4,096, best validation loss; QLoRA 4-bit for 24-27B bases | 2026-05: r 8, alpha 16, q/k/v, 300 steps, batch 64, TPU |
| Serving | the base served as trained (4-bit for QLoRA); context 8,192, to lose fewer CoT completions than the about 5% lost at 4,096 | 2026-07 Mistral: 4-bit train, bf16 eval; 2026-05: context 4,096 |
| Graders | Qwen2.5-14B filter, meta-llama Llama-3.1-8B evaluator, pinned | 2026-07: the NousResearch re-upload |
| Train split | per seed | 2026-07: once |
| Conditions | `base`, `base+bodhi`, `lora`, `lora+bodhi`, the E12 baselines, `base+red-flags-first`, `lora+red-flags-first` | 2026-05 and 2026-07: the first four only |

Cells: `healthbench__mistral-small-24b__bodhi` (general, widely used: FzpA-W2) and
`healthbench__medgemma-27b-text__bodhi` (the clinical base again; the 2026-05 adapters are
treated as not retained, so every new experiment on MedGemma needs this rerun, open
decision O7). Each gets the eval-only cell `<HB-new>__labelled` for E18 and E19. GPU:
yes, about 15 to 20 H100-hours per cell with all conditions (rough). Status: to run
after the build.

The proposed record of the first cell. Together with the records it references, it
passes `load_registry()`.

```yaml
# configs/cells/healthbench__mistral-small-24b__bodhi.yaml
id: healthbench__mistral-small-24b__bodhi
benchmark: healthbench
split: default
model: mistral-small-24b
protocol: bodhi
seeds: [42, 7, 13, 99, 101]
grading:
  filter: qwen2.5-14b
  evaluator: llama-3.1-8b
  judges: [medgemma-27b-text, meditron3-8b]
traces: {max_prompts: 4200, max_model_len: 8192, max_num_seqs: 64, max_new_tokens_per_pass: 2048}
filter:
  tau: {rule: fixed, value: 0.4}
  split_policy: per_seed
train:
  engine: peft
  quantization: 4bit
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  epochs: 3
  per_device_batch: 2
  grad_accum: 8
  effective_batch: 16
  lr: 1.0e-4
  scheduler: cosine
  warmup_ratio: 0.05
  max_seq_length: 4096
  gradient_checkpointing: true
  checkpoint: best_val_loss
  hardware: 1x H100 80GB
eval:
  conditions: [base, base+bodhi, lora, lora+bodhi, base+clarify, base+verbalized-confidence,
               base+mediq-abstain, base+red-flags-first, lora+red-flags-first]
  ids: {policy: per_seed_draw, n: 200}
  base_precision: train
  max_new_tokens: 1024
  max_new_tokens_per_pass: 2048
  serving: {max_model_len: 8192, max_num_seqs: 32, concurrency: 32, request_timeout_s: 900}
report:
  label: HealthBench-Hard, Mistral-Small-24B
  token: HBMistral
  status: planned
```

**E15. MedQA without the preamble.** Answers W12, C24 (follow-through), FzpA-Q3 (the
clinician framing), C1 (tasks with no missing information). The only MedQA run used a
preamble that told the model to ask; 035d85f removed it. Cell:
`medqa-open__mistral-small-24b__bodhi`, plus the rubric's keyed-answer check as an
accuracy measure. GPU: yes, about 10 H100-hours. Status: to run (benchmark record to
build).

**E16. Second teaching protocol.** Answers HomM-W1, HomM-final, MR-F (narrow protocols),
C29, C12. Ask-first (`scripts/baselines/ask_first.py`): one call, a three-question plan,
then a reply that asks first. Cell: `healthbench__mistral-small-24b__ask-first` with
`base`, `base+ask-first`, `lora`, `lora+ask-first`, and `base+bodhi` to compare the two
teachers on the same prompts; the E14 probe on its adapters. If budget allows, the same
on MedGemma-27B. GPU: yes, about 10 to 13 H100-hours per cell. Status: exists (dry-run
tested), to run.

**E17. Red flags first.** Answers LxMF-4, C31. The posted diagnosis says LoRA plus BODHI
loses red flags because the analysis format leaks into the reply. LxMF suggested
surfacing red flags early in the protocol. Protocol `red-flags-first`: BODHI with the red
flag section moved to the start of pass 1 and an instruction that the reply states red
flags first. Test: `lora+red-flags-first` minus `lora+bodhi` on red flag identification
and leak share, on the adapters trained with BODHI; `base+red-flags-first` shows the
effect of the order alone. Cells: E13. Optional: a cell that also teaches with it.
GPU: yes, inside E13. Status: to build (protocol), then to run.

**E18. Over-flagging set.** Answers C30, C18, FzpA-W3 (red-flag false positives).
HealthBench physicians labelled cases emergent, conditionally emergent or not emergent
(139, 180 and 134 in the full set). That is the labelled set of prompts with no genuine
red flag that the replies promised, made by physicians, so the deferred raters are not
needed for it. The new split keeps all 453 out of training. Statistics: red-flag score
and rate on non-emergent prompts (lower is better), the same on emergent prompts, and
their difference as a red-flag discrimination index, per condition. Cells: the
`<HB-new>__labelled` eval-only cells. On the existing cells only the Hard subset is
clean (8 non-emergent, 48 emergent or conditional); report it as descriptive. GPU: yes,
inside E13. Status: to build (labels and split), then to run.

The eval-only cell evaluates the E13 adapters on all 775 labelled prompts (453
emergency-labelled, 322 clinician-user):

```yaml
# configs/cells/healthbench__mistral-small-24b__bodhi__labelled.yaml
id: healthbench__mistral-small-24b__bodhi__labelled
benchmark: healthbench
split: default
model: mistral-small-24b
protocol: bodhi
seeds: [42, 7, 13, 99, 101]
adapters_from: healthbench__mistral-small-24b__bodhi
grading: {filter: qwen2.5-14b, evaluator: llama-3.1-8b, judges: [medgemma-27b-text]}
eval:
  conditions: [base, base+bodhi, lora, lora+bodhi, lora+red-flags-first]
  ids:
    policy: fixed_pool
    n: 775
    only:
      - {label: emergent-physician}
      - {label: clinician-user, groups: [positive]}
  base_precision: train
  max_new_tokens: 1024
  max_new_tokens_per_pass: 2048
  serving: {max_model_len: 8192, max_num_seqs: 32, concurrency: 32, request_timeout_s: 900}
report:
  label: HealthBench labelled probes, Mistral-Small-24B
  token: HBMistralLabelled
  status: planned
```

**E19. Clinician-posed evaluation.** Answers FzpA-Q3, C32, D4. MIMIC-IV and eICU need
credentialed PhysioNet access, and so does DiSCQ (physician questions on MIMIC notes).
Two openly available sets instead:
- Primary: HealthBench prompts that physicians labelled as coming from a health
  professional (`physician_agreed_category:health-professional`; 322 in the full set,
  40 of them in Hard). They are real conversations with expert rubrics, so the rubric
  score and the harm proxy work as on the main eval. The new split keeps all 322 out of
  training, so the adapters meet them unseen. Its limit: the same source as the training
  data.
- Secondary: the NLM Clinical Questions Collection (Ely et al., 1997, 1999; D'Alessandro
  et al., 2004): 4,654 questions that clinicians asked at the point of care, in four XML
  sets, public on the NLM data server. It is truly external and clinician-posed, and the
  questions are short and often lack patient context, so they test context-seeking.
  Its limits: no reference answers, so only the seven dimensions and the grader-free
  counts apply; the questions date from 1991 to 2003. Check the terms page before use.
Not chosen: MedQA (exam vignettes with complete information, E15 covers it), and
Asclepius or PMC-Patients questions (extraction and summarization over notes, which the
scope excludes, C1). Cells: `<HB-new>__labelled` for the primary;
`cqc__<model>__bodhi` eval-only cells for the secondary. GPU: yes, about 2 to 3
H100-hours per adapter cell for the secondary. Status: to build (CQC loader and record),
then to run.

**E20. Measured cost per configuration.** Answers C25, FzpA-S. The run manifests record
wall-clock time, GPU model and tokens per stage. The `bench` stage times a fixed batch
of 50 prompts per condition on an otherwise idle GPU, so shared-node queueing does not
enter the figures. Report training time per seed, and time and tokens per response per
condition. GPU: yes, inside E13 and E16. Status: to build (timers, `bench` stage from
`scripts/latency_benchmark.py`), then to run.

**E21. BioMistral null: teacher or student?** Answers HomM-F1, withdrawals W5. One null
cannot separate teacher strength from model size, instruction tuning or recipe.
(a) `healthbench__mistral-7b-instruct__bodhi`: the same size and architecture as
BioMistral-7B, general instruction tuning. (b)
`healthbench__biomistral-7b__bodhi__teacher-mistral-small-24b`: BioMistral trained on the
filtered traces of the Mistral-Small-24B cell (`traces_from`). If (b) transfers, the null
was the teacher; if (a) transfers and (b) does not, it is the student. GPU: yes, about 4
H100-hours each (no trace generation for b). Status: to build (`traces_from`), then to
run.

**E22. Filter ablation.** Answers a posted claim the discussion never corrected ("the
filter removes the teacher's failures before they reach the weights",
`rebuttal/withdrawals.md`), MR-W4, LxMF-2. Two cells beside
`healthbench__mistral-small-24b__bodhi`: `__unfiltered` (tau rule `none`, every trace)
and `__random-same-count` (a random sample as large as the filtered set). Contrasts on the
dimensions where teacher and student differ (the selectivity table). GPU: yes, about 8
H100-hours each (traces shared). Status: to build (tau rules), then to run.

**E23. Teacher-trace seed variance.** Answers REPRO-8, HomM-final (consolidation). Today
seeds vary training and the eval draw, never the teacher traces (greedy decoding, one
run per cell). Stage A, inference only: three sampled trace sets (temperature 0.7, trace
seeds 1 to 3) for the Mistral-Small-24B HealthBench cell; compare teacher behaviour and
filter clearance across them. Stage B: one adapter per trace set at a fixed training
seed, against the five-seed training spread. Cells:
`healthbench__mistral-small-24b__bodhi__traces-s1` to `-s3`. GPU: yes, about 3 H100-hours
for A, about 6 more for B. Status: to build (sampled decoding), then to run.

**E24. The general twin of MedGemma.** Answers FzpA-W2, MR-W1. Gemma-3-27B-it against
MedGemma-27B separates clinical tuning from family and size. Cell:
`healthbench__gemma-3-27b__bodhi`. The family rule removes Gemma judges for it; the
Llama clinical judges remain. GPU: yes, about 15 H100-hours. Status: to run (model record
and pin to build). Optional.

**E25. Uniform reruns on ChatDoctor and MedQuAD.** Answers HomM-Q2, MR-W2, FzpA-Q3 (more
than one benchmark). `chatdoctor__mistral-small-24b__bodhi` and
`medquad__mistral-small-24b__bodhi` under E13's settings, with a seeded pool and a seeded
holdout of 1,000 in place of the first rows in source order (REPRO-11). GPU: yes, about
10 to 15 H100-hours each. Status: to run. Optional if E13 holds.

**E26. BODHI component ablations.** The submitted paper named them as beyond its budget.
Protocols `bodhi-no-calibration`, `bodhi-no-questions`, `bodhi-no-abstention` as teachers
(`scripts/_bodhi_ablation.py` builds them, with a prompt-drift check). GPU: yes, about 10
H100-hours per protocol. Status: exists (code), to run. Optional; no reviewer asked.

### People

**E27. Physician validation.** Answers C28, MR-W3, HomM-W2, HomM-W3, FzpA-Q1. Consensus
kappa of three physicians against the 0.6 target; today one rater, kappa 0.35, n = 509
(D5). Deferred: the first author deferred the two remaining raters. The kappa interval
(prompt-clustered bootstrap) is computed now from the returned file (`make kappa`).

**E28. Physician harm review.** Answers C33. The Appendix C review by clinicians.
Deferred. E06 runs as the automated proxy in the meantime.

## 4. Planned cells and cost

Tier A answers the rejection reasons. Tier B adds strength. Tier C is optional. All
H100-hour figures are rough estimates (section 1).

| Tier | Cell | Kind | Items | H100-h |
|---|---|---|---|---|
| A | `healthbench__mistral-small-24b__bodhi` | train | E13, E12, E17, E20 | 15-20 |
| A | `healthbench__medgemma-27b-text__bodhi` | train (QLoRA) | E13, E12, E17, E20 | 15-20 |
| A | `healthbench__mistral-small-24b__bodhi__labelled`, `healthbench__medgemma-27b-text__bodhi__labelled` | eval only | E18, E19 | 3-5 each |
| A | `mediq-paired__mistral-small-24b__bodhi`, `mediq-paired__medgemma-27b-text__bodhi` | eval only | E14 | 2-4 each |
| A | `healthbench__mistral-small-24b__ask-first` (+ its `mediq-paired` cell) | train | E16, E14 | 12-17 |
| A | judges on the eight existing cells | grade only | E09, E10 | 15-40 |
| B | `medqa-open__mistral-small-24b__bodhi` | train | E15 | 10 |
| B | `healthbench__mistral-7b-instruct__bodhi`, `healthbench__biomistral-7b__bodhi__teacher-mistral-small-24b` | train | E21 | 4 each |
| B | `cqc__mistral-small-24b__bodhi`, `cqc__medgemma-27b-text__bodhi` | eval only | E19 | 2-3 each |
| B | probes for five base models | probe | E11 | 1 per model and benchmark |
| B | `healthbench__mistral-small-24b__bodhi__unfiltered`, `__random-same-count` | train | E22 | 8 each |
| C | `healthbench__mistral-small-24b__bodhi__traces-s1` to `-s3` | traces, then train | E23 | 3-9 |
| C | `healthbench__gemma-3-27b__bodhi` | train | E24 | 15 |
| C | `chatdoctor__mistral-small-24b__bodhi`, `medquad__mistral-small-24b__bodhi` | train | E25 | 10-15 each |
| C | `healthbench__mistral-small-24b__bodhi-no-<component>` (three) | train | E26 | 10 each |

Summed from the rows, Tier A comes to roughly 65 to 115 H100-hours, and Tiers A and B
to roughly 115 to 170. On the shared host this has no meter; at an on-demand rate of 2 to
4 dollars per H100-hour, Tier A is roughly 130 to 460 dollars. Nothing runs before the
first author's go-ahead (open decision O6).

## 5. Coverage

Every concern and every commitment, and what answers it. "Done" items are text or table
changes already in `paper/main.tex`, with the commitment that records them.

| Concern | Answered by |
|---|---|
| MR-W1 | E13 (a general base), E24, E19; text: scope is patient-facing Q&A (D4, C6, done) |
| MR-W2 | E16, E17, E13, E25; text: recipe plus decomposition, not a framework (C1, C2, C12, done) |
| MR-W3 | E09, E10, E02, E03, E14, E06; E27 deferred; text: grader limits (C13, done) |
| MR-W4 | E02, E14, E22, E03; the 2026-05 discrimination analysis exists (`rebuttal/results/mimicry_split.json`) |
| MR-W5 | text done (C5, C10, C20); E07 makes the criterion operational; E01 tables |
| MR-W6 | E12; text: prior-work subsection and table (C15, done) |
| MR-F | P1, E01, E13 (one set of settings), E12, E16; every corrected claim in the paper (C11, done) |
| FzpA-W1 | text done (C1, C2, C4); E15 shows a task with no missing information |
| FzpA-W2 | E13 (Mistral-Small-24B), E24, E21 (Mistral-7B-Instruct) |
| FzpA-W3 | text and figures done (C5, C16, C17, C18, C20) |
| FzpA-S | E08, E20 (C25) |
| FzpA-Q1 | text done (C21); E01 dimension correlations; E27 deferred |
| FzpA-Q2 | text done (C22, C23); E01 drawn examples per score level |
| FzpA-Q3 | E19, E15, E25; the 2026-07 ChatDoctor and MedQuAD cells exist |
| FzpA-F1 | text done (C6, C7; withdrawals W2, W3); E19; E27 deferred |
| FzpA-F2 | text done (C10, C20); E07, E06 |
| FzpA-F3 | text done (C23) |
| FzpA-final | D4 and C6 done; E19; P1 |
| LxMF-1 | E01 counts and parse failures, E08 lost completions; caption done (C19) |
| LxMF-2 | E02, E14, E22, E03; exists for the 2026-05 cell |
| LxMF-3 | E10 (Med42-8B, Meditron3-8B, Aloe-Beta-8B, MedGemma-27B), E09; text done (C13) |
| LxMF-4 | E04, E17; truncation measured through `finish_length` in E13 |
| LxMF-C | done (C16) |
| HomM-W1, Q1, Q2 | E13, E16, E25, E21, E24; the 2026-07 cells exist |
| HomM-W2 | E13, E19, E25; E27 deferred |
| HomM-W3, Q4, F4 | E10, E09, E02, E03, E14; text done (C13) |
| HomM-W4, Q3, F5 | E12; text done (C14, C15) |
| HomM-F1 | E21; text done (withdrawals W5) |
| HomM-F2 | withdrawn (W4); E01 reports the spreads with no claim |
| HomM-F3 | E01 per-cell and cross-cell tables |
| HomM-final | E27 deferred, E12, E16, E05, E06, E07, P1 |

| Commitment | Status now | Answered by |
|---|---|---|
| C1 to C24 | done (text, figures and tables; see `rebuttal/commitments.md`) | E01 recomputes every number they cite from the response tables |
| C25 cost per configuration | to-do | E08, E20 |
| C26 clinical judge panel | to-do | E10 (E09 for the Med42-8B cells) |
| C27 inference-time baseline | to-do | E12 |
| C28 three-physician validation | to-do | E27, deferred |
| C29 second CoT protocol | to-do | E16 |
| C30 over-flagging set | to-do | E18 |
| C31 red flags early | to-do | E17 |
| C32 clinician-posed benchmark | to-do | E19 (open sets in place of MIMIC and eICU) |
| C33 harm grading | to-do | E06 now (proxy), E28 deferred |

Audit items that need a run: REPRO-8 (E23), EVAL-12 and NUM-7 (E11), REPRO-12 (E13
serves the base as trained), REPRO-3 (E13), PROV-11 (E13 manifests), REPRO-11 (E25).

## 6. Decisions for the first author

These are methodology choices, so they are his (they are also in ARCHITECTURE.md
section 12):

- The analysis plan (P1), including whether E02 is a primary or a secondary test.
- The E13 settings, most of all the HealthBench split (679 fewer training prompts) and
  the context length of 8,192 (it should cut the CoT attrition, but it differs from the
  2026-05 cell).
- The margins of E07.
- The tiers and order of the GPU work, and when it starts.
- Whether the new cells replace the 2026-05 and 2026-07 cells in the main tables, or
  report next to them as the record.
