# Run plan

**Needs the first author's go-ahead.** No GPU job runs until he says so (DECISIONS.md,
2026-10-08). Nothing in this file has run. Dated 2026-10-08.

This is the ordered list of GPU work from [EXPERIMENTS.md](EXPERIMENTS.md), with the
command, the cells, the seeds, an H100-hour estimate and its basis for each run, and the
reviewer points each run answers. [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) fixes how each
result is read. The commands are the command line of [ARCHITECTURE.md](ARCHITECTURE.md)
section 5.

## 1. Before any run

1. The first author adopts [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md), or changes it in its
   section 12 (convention 3). No run starts before that.
2. The cell record exists in `configs/` with `report.status: planned`, and
   `python -m bodhi_lora list cells` loads it. The experiments lane writes the records.
3. Every protocol and loader that the record names exists and passes its stub test. The
   runner freezes a record at its first run (`cell.yaml` in the cell folder must match
   it), so a condition left out at the start needs a new cell later. Today
   `mediq-abstain`, `red-flags-first`, the `mediq_paired` and `cqc_xml` loaders, and the
   records of Gemma-3-27B and Aloe-Beta-8B do not exist.
4. `python -m bodhi_lora run <cell id> --dry-run` passes on the CPU. It prints every GPU
   command and runs the rest with the stub engine.
5. The GPU machine has `.venv` and `.venv-infer` (`bash setup.sh`), and `HF_TOKEN` has
   access to the gated models the cell uses: MedGemma, Llama-3.1, and Gemma 2 (licence
   accepted) for R1. Meditron3-8B needs a pinned revision first (BLOCKED.md).
6. The checkout on the GPU machine is clean (`git status` shows nothing), so every
   manifest records a clean commit.
7. The first author chooses where it runs: the shared H100 host or a paid cloud. Not
   Modal (DECISIONS.md, 2026-10-08).

## 2. How a cell runs

```bash
python -m bodhi_lora run <cell id> --dry-run
# Technical check: the shared stages and one seed. Read the manifests only (ANALYSIS_PLAN.md 2.5).
python -m bodhi_lora run <cell id> --stages probe,traces,filter,train,eval,grade --seeds 42 --gpus 0
# The other seeds, one per GPU. Stages with a complete manifest are skipped.
python -m bodhi_lora run <cell id> --stages probe,traces,filter,train,eval,grade --gpus 0,1,2,3,4
# Timing on an otherwise idle GPU, one seed only (E20).
python -m bodhi_lora run <cell id> --stages bench --seeds 42 --gpus 0
# Judges on a cell that has already run (a recorded cell): adds judge files only.
python -m bodhi_lora run <recorded cell id> --stages grade --gpus 0
```

The runner runs seeds in parallel, one per GPU, so with five GPUs a cell takes about its
shared stages plus one seed in wall-clock time. An eval-only cell (`adapters_from`) needs
the adapters of its source cell on the same machine.

The `bench` stage runs for every seed when it is in the stage list. One seed is enough
for timing, so the main run leaves it out and the last command runs it for seed 42.

## 3. Cost basis

All figures are H100-hours: one GPU for one hour. They are estimates, not measurements of
the planned runs. The run manifests record the real times (E20), so the first batch
replaces them.

### 3.1 Measured stage times of the 2026-07 cells

Wall-clock minutes on the shared H100 host, July 2026, one GPU per stage. Training times
are `train_runtime` in each seed's `trainer_state.json` (tracked). The other times are
the last progress-bar time of each stage plus the vLLM start times, read from the host
logs in each cell's `logs/` folder (ignored by git; copied on 2026-10-08). Eval covers
the four conditions with their rubric and dimension grading.

| Cell | Traces | Filter | Train, per seed | Eval, per seed | Cell total (h) |
|---|---|---|---|---|---|
| `healthbench__mistral-small-24b__bodhi__2026-07` | 50 | 74 | 60 | 50 | 11.2 |
| `chatdoctor__mistral-small-24b__bodhi__2026-07` | 49 | 50 (a) | 92 | 33 | 12.1 |
| `medquad__mistral-small-24b__bodhi__2026-07` | 34 | 50 (a) | 55 | 30 | 8.5 |
| `medqa-open-preamble__mistral-small-24b__bodhi__2026-07` | 64 | 53 | 122 | 42 | 15.6 |
| `healthbench__med42-8b__bodhi__2026-07` | 15 | 66 | 9 (b) | 29 | 4.5 |
| `chatdoctor__med42-8b__bodhi__2026-07` | 18 | 49 | 16 | 20 | 4.1 |
| `healthbench__biomistral-7b__bodhi__2026-07` | 6 | 65 | 4 | 47 | 5.4 |

(a) The filter log of this cell shows a resumed run with no grading time. The value is
the Med42-8B ChatDoctor filter time, the same benchmark size. (b) Three seeds; the other
two trainer states come from resumed runs that took under a minute.

Per condition, for 200 HealthBench prompts with Mistral-Small-24B: inference 1.7 minutes
for `base`, 4.6 for `base+bodhi`, 6.7 for `lora` and 17.1 for `lora+bodhi`; rubric
grading 2.2 and dimension grading 0.9 with the 8B evaluator; about 1.9 for the vLLM starts.
The measured cell is 11.2 H100-hours, or 2.2 per seed. The posted reply gave about two
per 24B seed (`rebuttal/responses/posted/FzpA_1_rebuttal.md`).

The host was shared, so these times include other jobs on the machine. C25 promised
times measured in isolation; those come from the `bench` stage, not from this table.

### 3.2 Assumptions for the planned cells

- **A1.** Stage time scales with the number of prompts or training rows.
- **A2.** A one-call condition on the base costs what `base` cost. A two-call condition
  on the base (`base+mediq-abstain`, `base+red-flags-first`) costs what `base+bodhi`
  cost. A two-call condition on the adapter costs what `lora+bodhi` cost. Each condition
  adds its grading and its vLLM starts as measured.
- **A3.** The new cells serve the base as it was trained (4-bit for QLoRA). vLLM serves a
  4-bit model more slowly than bf16 by an amount not measured here. The low figure
  assumes no slowdown; the high figure doubles the inference time.
- **A4.** A 27B judge grades at one third of the speed of the 8B evaluator (parameter
  ratio about 3.4). An 8B judge grades at the evaluator's speed. Not measured.
- **A5.** On the base model's stages, MedGemma-27B and Gemma-3-27B take 1.1 to 1.5 times
  as long as Mistral-Small-24B: parameter ratio 1.13, and Gemma 3's vocabulary of 262k
  tokens makes the output layer heavier. Not measured on an H100.
- **A6.** The HealthBench `default` split leaves 3,321 train-pool prompts instead of
  4,000, so traces, filter and training take 0.83 of the 2026-07 times.

## 4. Ordered run list

Tier A answers the reasons for rejection. Tier B adds strength. Tier C is optional. The
seeds are 42, 7, 13, 99 and 101 in every cell (three trace sets at one seed in R13).

| Run | Items | Cells | Kind | Needs first | H100-h | Basis | Reviewer points |
|---|---|---|---|---|---|---|---|
| R1 (A) | E09 | `healthbench__med42-8b__bodhi__2026-07`, `chatdoctor__med42-8b__bodhi__2026-07`, judge `gemma-2-27b` | judges on recorded cells | the two cell folders with their eval files on the GPU machine | 5.1 | 4 conditions x 5 seeds, A4 | D6, EVAL-7, LxMF-3, HomM-W3, HomM-Q4, C26 (part) |
| R2 (A) | E13, E12, E17, E20 | `healthbench__mistral-small-24b__bodhi`, nine conditions | train | `mediq-abstain` and `red-flags-first` protocols | 14.6 to 19.6; bench 0.4 to 0.6; judges `medgemma-27b-text` 7.1 and `meditron3-8b` 2.5 | the Mistral HealthBench cell of 3.1, A2, A3, A4, A6 | MR-W1, MR-W2, MR-W6, MR-F, FzpA-W2, FzpA-S, HomM-W1, HomM-W4, HomM-Q1 to Q3, HomM-F5, LxMF-4; C9, C25, C27, C31; REPRO-3, REPRO-8, REPRO-12, PROV-11 |
| R3 (A) | E14 | `mediq-paired__mistral-small-24b__bodhi` on the R2 adapters | eval only | R2; the `mediq_paired` loader and benchmark record | 2.5 to 3.9 (`base`, `lora`); 4.8 to 7.5 with `base+bodhi` and `base+clarify` | 400 prompts per seed, MedQA-preamble cell times, A3 | MR-W4, LxMF-2 |
| R4 (A) | E18, E19 (primary set) | `healthbench__mistral-small-24b__bodhi__labelled` on the R2 adapters, 775 prompts | eval only | R2 | 7.7 to 11.9 with `base`, `base+bodhi`, `lora`; 21.0 to 36.3 with the five conditions of EXPERIMENTS.md; a `medgemma-27b-text` judge adds 15.2 | A1 to A3 | C30, C18, FzpA-W3, C32, FzpA-Q3 |
| R5 (A) | E16, E14 | `healthbench__mistral-small-24b__ask-first` and `mediq-paired__mistral-small-24b__ask-first` | train, then eval only | the cell records | 9.4 to 13.7, and 2.5 to 3.9 | one-call traces at 0.5 to 1 times the BODHI trace time, A2, A3 | HomM-W1, HomM-final, MR-F, C12, C29 |
| R6 (A) | E10 | judge `medgemma-27b-text` on the Mistral HealthBench, BioMistral, Mistral ChatDoctor and Mistral MedQuAD cells of 2026-07; judge `med42-8b` on the 2026-05 cell | judges on recorded cells | judges added to those five records | 11.6 | 4 conditions x 5 seeds per cell, A4 | C26, LxMF-3, HomM-Q4, HomM-W3, HomM-F4, MR-W3 |
| R7 (A) | E13, E12, E17, E20, E14, E18, E19 | `healthbench__medgemma-27b-text__bodhi` (QLoRA), its `__labelled` cell and `mediq-paired__medgemma-27b-text__bodhi` | train, then eval only | open decision O7 | 15.6 to 27.0; bench 0.5 to 0.9; an 8B judge 2.5; labelled 8.1 to 16.1; paired 2.6 to 5.4 | R2 to R4 with A5 | MR-W1 (the clinical base again), continuity with the 2026-05 cell |
| R8 (B) | E15 | `medqa-open__mistral-small-24b__bodhi` | train | the `medqa-open` record | 15.6 to 18.0 | the MedQA-preamble cell of 3.1, A3 | W12, C1, C24 (follow-through), FzpA-Q3 |
| R9 (B) | E21 | `healthbench__mistral-7b-instruct__bodhi`, `healthbench__biomistral-7b__bodhi__teacher-mistral-small-24b` | train | R2 (traces) | 4.2 to 6.1, and 4.7 to 6.3 | the Med42-8B and BioMistral-7B HealthBench cells of 3.1, A6 | HomM-F1, withdrawals W5 |
| R10 (B) | E19 (secondary set) | `cqc__mistral-small-24b__bodhi`, `cqc__medgemma-27b-text__bodhi` | eval only | R2, R7; the `cqc_xml` loader | 1.8 to 2.9, and 1.9 to 3.9 | 200 questions per seed, three conditions, no rubric grading | FzpA-Q3, C32 |
| R11 (B) | E11 | probes: Mistral-Small-24B on four benchmarks, Med42-8B on two, BioMistral-7B, Mistral-7B-Instruct and MedGemma-27B on HealthBench | probe | a `probe` block in a cell record per model | 2.2 to 4.5 | 0.25 to 0.5 each: 400 short greedy continuations and one vLLM start | EVAL-12, NUM-7 |
| R12 (B) | E22 | `healthbench__mistral-small-24b__bodhi__unfiltered`, `__random-same-count` | train | R2 (traces) | 8.9 to 10.7, and 5.7 to 7.4 | training on 3,321 or the kept rows; `base` and `lora` only | MR-W4, LxMF-2, the filter claim |
| R13 (C) | E23 | `healthbench__mistral-small-24b__bodhi__traces-s1` to `-s3` | traces, then train at one seed | sampled decoding | 10.1 to 11.6 | three trace and filter passes, three adapters | REPRO-8, HomM-final |
| R14 (C) | E24 | `healthbench__gemma-3-27b__bodhi` | train | the model record and pin | 10.8 to 17.5 | four conditions, A5 | FzpA-W2, MR-W1 |
| R15 (C) | E25 | `chatdoctor__mistral-small-24b__bodhi`, `medquad__mistral-small-24b__bodhi` | train | seeded pools in the records | 12.1 to 13.7, and 8.5 to 10.0 | the 2026-07 cells of 3.1, A3 | HomM-Q2, MR-W2, FzpA-Q3 |
| R16 (C) | E26 | `healthbench__mistral-small-24b__bodhi-no-calibration`, `-no-questions`, `-no-abstention` | train | the cell records | 10.0 to 12.5 each | four conditions | none asked |

The labelled cells (R4, R7) cost much more than EXPERIMENTS.md section 4 says, because
they evaluate 775 prompts per seed. H7 and H11 need only `base` and `lora`, with
`base+bodhi` as the teacher, so this plan proposes those three conditions and no judge.
The runner also evaluates `base` again for every seed, although with fixed eval ids and
greedy decoding its output is the same for each seed. A runner option to evaluate such a
condition once would save nearly half of the time of a labelled cell with three conditions.

## 5. Minimal first batch

If compute is tight, run these first, about **23 to 29 H100-hours**:

1. R1, the Med42-8B judge re-grade (5.1). It needs no training and closes the D6
   promise.
2. R2, the uniform Mistral-Small-24B cell with the baselines and red flags first, judges
   left for later (15.0 to 20.2 with the bench run).
3. R3, the MediQ paired probe on the R2 adapters, `base` and `lora` only (2.5 to 3.9).

Together with E02 on the existing outputs (no GPU), this answers the reasons for
rejection that a run can answer: a widely used general base under one set of settings
(MR-W1, MR-W2, FzpA-W2), a prior-work comparison (MR-W6, HomM-W4, C27), calibration
against mimicry (MR-W4, LxMF-2), red flags first (C31), the Med42-8B grader overlap (D6),
and measured cost (C25). It leaves the second protocol (R5), the clinical judge panel
(R6), the labelled probes (R4) and the MedGemma rerun (R7) for the next batch.

To add the judges to R2 later: when R2 is complete, its record gets the judge list and
`report.status: complete`, and `python -m bodhi_lora run healthbench__mistral-small-24b__bodhi --stages grade --gpus 0`
adds them (7.1 for `medgemma-27b-text`, 2.5 for `meditron3-8b`).

## 6. Totals

| Set | H100-hours | At 2 to 4 dollars per H100-hour |
|---|---|---|
| Minimal first batch | 22.6 to 29.2 | 45 to 117 dollars |
| Tier A (labelled cells with three conditions, R2 judges included) | 92.7 to 131.8 | 185 to 527 dollars |
| Tiers A and B | 137.7 to 191.6 | 275 to 766 dollars |
| All tiers | 209.2 to 281.9 | 418 to 1,128 dollars |

The dollar rate is the one EXPERIMENTS.md assumes; it was not checked against a current
price list. The shared host has no meter, but each run holds a card for its hours.

## 7. After each run

1. If the cell ran on another machine, copy its folder back to the same path here. Leave
   the adapter weights on the GPU machine unless an eval-only cell needs them here:

   ```bash
   rsync -av --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' \
       <gpu_machine>:<repo>/results/<cell_id>/ results/<cell_id>/
   ```

2. Set `report.status: complete` (or `partial`) in the cell record. From then on the
   runner refuses to run the cell again, except for judge re-grades.
3. Run the analysis: `python -m bodhi_lora analyze <cell id> --extract`, then
   `make analyze` and `make report` (ARCHITECTURE.md section 5).
4. Commit the tracked files: the manifests, `cell.yaml`, the id lists, the response
   tables and the cell's analysis folder. Git ignores the raw outputs by kind.
5. Read the results only by the rules of ANALYSIS_PLAN.md, and record any change of plan
   in its section 12 before the analysis runs.
