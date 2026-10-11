# Run plan

**Needs the first author's go-ahead.** No GPU job runs until he says so (DECISIONS.md,
2026-10-08). Nothing in this file has run. Dated 2026-10-08. On 2026-10-09 sections 3
to 6 were updated to cost the cell records as they stand. On 2026-10-10 R17 was added,
the judge classifier that replaces the withdrawn keyword rules, R11 lost the MedQuAD
probe, which cannot run, and the MedQuAD split of R15 began to leave out repeated
questions, so that its dry run passes the leakage gate. On 2026-10-11 the first author
adopted the analysis plan, R17 got its stage (`classify`), dry runs stopped downloading
anything, and section 4.1 records the dry run of every run.

This is the ordered list of GPU work from [EXPERIMENTS.md](EXPERIMENTS.md), with the
command, the cells, the seeds, an H100-hour estimate and its basis for each run, and the
reviewer points each run answers. [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) fixes how each
result is read. The commands are the command line of [ARCHITECTURE.md](ARCHITECTURE.md)
section 5.

## 1. Before any run

1. The first author adopts [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md), or changes it in its
   section 12 (convention 3). Done on 2026-10-11 (section 12, the adoption entry).
2. The cell record exists in `configs/` with `report.status: planned`, and
   `python -m bodhi_lora list cells` loads it. The experiments step writes the records.
3. Every protocol and loader that the record names exists and passes its stub test. The
   runner freezes a record at its first run (`cell.yaml` in the cell folder must match
   it), so a condition left out at the start needs a new cell later. All of them exist
   since 2026-10-09: `mediq-abstain`, `red-flags-first`, the `paired_context_jsonl` and
   `xml_records` loaders, and the Gemma-3-27B and Phi-4 records. The MediQ and NLM files
   are not pinned yet: `python -m bodhi_lora fetch <benchmark>` downloads them and prints
   the sha256 to put in the benchmark record. The downloads need the first author's
   go-ahead (BLOCKED.md item 2).
4. `python -m bodhi_lora run <cell id> --dry-run` passes on the CPU. It prints every GPU
   command and runs the rest with the stub engine. It never downloads: a benchmark that is
   not in the data root stops it, after the GPU commands, with the fetch command.
5. The GPU machine has `.venv` and `.venv-infer` (`bash setup.sh`), and `HF_TOKEN` has
   access to the gated models the cell uses: MedGemma (base and judge), Llama-3.1, Gemma 2
   (licence accepted) for R1, and Gemma 3 for R14.
6. The checkout on the GPU machine is clean (`git status` shows nothing), so every
   manifest records a clean commit.
7. The first author chooses where it runs: the shared H100 host or a paid cloud. Not
   Modal (DECISIONS.md, 2026-10-08). On the shared host, claim the cards with the group's
   lease file first; the runner does not. The repo's copy of the lease tool went on
   2026-10-11, and `854761f:scripts/gpu_lease.sh` has it.

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
# One judge of the record only, for example R1 on a Med42-8B cell:
python -m bodhi_lora run healthbench__med42-8b__bodhi__2026-07 --stages grade --judges gemma-2-27b --gpus 0
# The judge classifier of R17 (only when --stages names it): adds classify files only.
python -m bodhi_lora run <cell id> --stages classify --judges <judge> --gpus 0
```

`--stages grade` on a recorded cell runs every judge its record lists. The two Med42-8B
records list `gemma-2-27b` (R1) and `medgemma-27b-text` (R6), and the set-aside MedQA
record lists `medgemma-27b-text`, which no run below costs; `--judges` picks one.

The runner runs seeds in parallel, one per GPU, so with five GPUs a cell takes about its
shared stages plus one seed in wall-clock time. An eval-only cell (`adapters_from`) needs
the adapters of its source cell on the same machine.

The `bench` stage runs only for the seeds in the record's `bench.seeds` (seed 42 in
every planned cell), at the eval's token limits. A run with every stage therefore times
seed 42 once its eval is done; the separate `--stages bench` command above only matters
when the timing must wait for an otherwise idle GPU.

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

Per condition and seed, for 200 prompts, from the same logs, in minutes. Inference makes
the replies of one condition. Rubric and dimensions are the grading by the 8B evaluator.
Each condition also needs about 1.9 minutes of vLLM starts.

| Cell | Inference: `base`, `base+bodhi`, `lora`, `lora+bodhi` | Rubric | Dimensions |
|---|---|---|---|
| `healthbench__mistral-small-24b__bodhi__2026-07` | 1.7, 4.6, 6.7, 17.1 | 2.2 | 0.9 |
| `chatdoctor__mistral-small-24b__bodhi__2026-07` | 1.6, 4.4, 4.2, 8.9 | 1.3 | 0.6 |
| `medquad__mistral-small-24b__bodhi__2026-07` | 1.3, 3.0, 3.9, 10.0 | 1.2 | 0.4 |
| `medqa-open-preamble__mistral-small-24b__bodhi__2026-07` | 2.2, 5.5, 6.5, 14.3 | 1.4 | 0.7 |
| `healthbench__med42-8b__bodhi__2026-07` | 0.6, 1.3, 2.5, 7.2 | 1.9 | 0.7 |
| `chatdoctor__med42-8b__bodhi__2026-07` | 0.8, 1.6, 2.2, 5.2 | 1.3 | 0.9 (c) |
| `healthbench__biomistral-7b__bodhi__2026-07` | 0.5, 0.7, 5.2, 21.6 | 2.3 | 0.8 |

(c) From the re-grade log of the cell; its eval logs hold no dimension grading.

The measured Mistral-Small-24B HealthBench cell is 11.2 H100-hours, or 2.2 per seed. The
posted reply gave about two per 24B seed (`rebuttal/responses/posted/FzpA_1_rebuttal.md`).

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
  ratio about 3.4). Phi-4 (14.7B) grades at 1/1.8 of it. On a recorded cell, the
  evaluator's speed is that cell's own (the table in 3.1); on a planned cell, that of the
  2026-07 cell on the same benchmark. Each judge needs one vLLM start per seed. Not
  measured.
- **A5.** On the base model's stages, MedGemma-27B and Gemma-3-27B take 1.1 to 1.5 times
  as long as Mistral-Small-24B: parameter ratio 1.13, and Gemma 3's vocabulary of 262k
  tokens makes the output layer heavier. Not measured on an H100.
- **A6.** The HealthBench `default` split leaves 3,321 train-pool prompts instead of
  4,000, so traces, filter and training take 0.83 of the 2026-07 times.
- **A7.** A judge makes three calls per reply (`bodhi_lora/pipeline/grade.py`): the
  rubric criteria that the evaluator graded, the seven dimensions, and the harm
  questions. The harm call costs what a dimension call costs. On a benchmark with no
  rubric (the NLM questions), a judge makes only the last two calls. Not measured.
- **A8.** A contamination probe costs 0.25 to 0.5: 400 short greedy continuations and one
  vLLM start. A timing bench with fewer conditions than R2 costs R2's bench times the
  ratio of the condition counts.

## 4. Ordered run list

Tier A answers the reasons for rejection. Tier B adds strength. Tier C is optional. The
seeds are 42, 7, 13, 99 and 101 in every cell (three trace sets at one seed in R13).

Each figure costs the cell record as it stands on 2026-10-09: its conditions, prompts,
seeds and probe. The judges that a record lists have their own column, because
`--judges none` leaves them for later without a change to the record.

| Run | Items | Cells | Kind | Needs first | H100-h | Judges, H100-h | Basis | Reviewer points |
|---|---|---|---|---|---|---|---|---|
| R1 (A) | E09 | `healthbench__med42-8b__bodhi__2026-07`, `chatdoctor__med42-8b__bodhi__2026-07` | judge `gemma-2-27b` on recorded cells | the two cell folders with their eval files on the GPU machine | none | 6.7 | 4 conditions x 5 seeds at each cell's own grading times, A4, A7 | D6, EVAL-7, LxMF-3, HomM-W3, HomM-Q4, C26 (part) |
| R2 (A) | E13, E12, E17, E20 | `healthbench__mistral-small-24b__bodhi`, nine conditions | train | none left | 14.6 to 19.6; bench 0.4 to 0.6 | 9.2 (`medgemma-27b-text`) | the Mistral HealthBench cell of 3.1, A2 to A4, A6, A7 | MR-W1, MR-W2, MR-W6, MR-F, FzpA-W2, FzpA-S, HomM-W1, HomM-W4, HomM-Q1 to Q3, HomM-F5, LxMF-4; C9, C25, C27, C31; REPRO-3, REPRO-8, REPRO-12, PROV-11 |
| R3 (A) | E14 | `mediq-paired__mistral-small-24b__bodhi` on the R2 adapters, seven conditions | eval only | R2; the pinned MediQ file | 9.9 to 16.3 | none | 400 prompts per seed at the MedQA-preamble cell's times, A2, A3 | MR-W4, LxMF-2 |
| R4 (A) | E18, E19 (primary set) | `healthbench__mistral-small-24b__bodhi__labelled` on the R2 adapters, 775 prompts, five conditions | eval only | R2 | 21.0 to 36.3 | none | A1 to A3 | C30, C18, FzpA-W3, C32, FzpA-Q3 |
| R5 (A) | E16, E14 | `healthbench__mistral-small-24b__ask-first` (five conditions) and `mediq-paired__mistral-small-24b__ask-first` (four) | train, then eval only | the pinned MediQ file | 9.4 to 13.7; bench 0.2 to 0.3; paired 4.9 to 11.5 | 5.2 (`medgemma-27b-text`) | low: `ask-first` costs one call and its traces half of BODHI's; high: two calls and the full trace time; A2, A3, A8 | HomM-W1, HomM-final, MR-F, C12, C29 |
| R6 (A) | E10 | judge `medgemma-27b-text` on six cells of 2026-07 (Mistral-Small-24B on HealthBench, ChatDoctor and MedQuAD, BioMistral-7B, both Med42-8B cells); judge `phi-4` on the 2026-05 cell | judges on recorded cells | none left (the records list the judges) | none | 22.4 | 4 conditions x 5 seeds per cell at each cell's own grading times, A4, A7 | C26, LxMF-3, HomM-Q4, HomM-W3, HomM-F4, MR-W3 |
| R7 (A) | E13, E12, E17, E20, E14, E18, E19 | `healthbench__medgemma-27b-text__bodhi` (QLoRA, nine conditions), its `__labelled` cell (five) and `mediq-paired__medgemma-27b-text__bodhi` (seven) | train, then eval only | open decision O7; the pinned MediQ file | 15.6 to 27.0; bench 0.5 to 0.9; labelled 22.6 to 51.5; paired 10.6 to 22.7 | 5.6 (`phi-4`) | R2 to R4 with A5 | MR-W1 (the clinical base again), continuity with the 2026-05 cell |
| R8 (B) | E15 | `medqa-open__mistral-small-24b__bodhi`, seven conditions | train | none left | 17.4 to 20.6 | 5.0 (`medgemma-27b-text`) | the MedQA-preamble cell of 3.1 and three baseline conditions, A2, A3 | W12, C1, C24 (follow-through), FzpA-Q3 |
| R9 (B) | E21 | `healthbench__mistral-7b-instruct__bodhi` with its probe, `healthbench__biomistral-7b__bodhi__teacher-mistral-small-24b` | train | R2 (traces) | 4.5 to 6.6, and 4.7 to 6.3 | 3.5 and 4.0 (`medgemma-27b-text`) | the Med42-8B and BioMistral-7B HealthBench cells of 3.1, A6, A8 | HomM-F1, withdrawals W5 |
| R10 (B) | E19 (secondary set) | `nlm-clinical-questions__mistral-small-24b__bodhi`, `nlm-clinical-questions__medgemma-27b-text__bodhi`, seven conditions each | eval only | R2, R7; the pinned NLM files | 4.8 to 8.0, and 5.1 to 11.1 | 3.3 (`medgemma-27b-text`) and 2.1 (`phi-4`) | 200 questions per seed at the Mistral HealthBench times, no rubric grading, A2, A3, A5 | FzpA-Q3, C32 |
| R11 (B) | E11 | probes on seven recorded cells: Mistral-Small-24B on HealthBench, ChatDoctor and MedQA, Med42-8B on two benchmarks, BioMistral-7B and MedGemma-27B on HealthBench. Not MedQuAD: its questions are shorter than the 40 tokens the probe needs (at most 21 tokens in the eval pool with the Mistral tokenizer; 1 of 14,979 rows reaches 40), so its record has no probe block | probe | none left (`probe` blocks in the recorded cells; R9 and R14 probe in their own runs) | 1.8 to 3.5 | none | A8 | EVAL-12, NUM-7 |
| R12 (B) | E22 | `healthbench__mistral-small-24b__bodhi__unfiltered`, `__random-same-count` | train | R2 (traces) | 8.9 to 10.7, and 5.7 to 7.4 | none | training on 3,321 or the kept rows; `base` and `lora` only | MR-W4, LxMF-2, the filter claim |
| R13 (C) | E23 | `healthbench__mistral-small-24b__bodhi__traces-s1`, `-s2`, `-s3` | traces, then train at one seed | none left | 10.1 to 11.6 | none | three trace and filter passes, three adapters | REPRO-8, HomM-final |
| R14 (C) | E24 | `healthbench__gemma-3-27b__bodhi` with its probe | train | none left | 11.1 to 18.0 | 2.6 (`phi-4`) | four conditions, A5, A8 | FzpA-W2, MR-W1 |
| R15 (C) | E25 | `chatdoctor__mistral-small-24b__bodhi`, `medquad__mistral-small-24b__bodhi`, seven conditions each. MedQuAD repeats questions under several ids; since 2026-10-10 the split leaves a train row out when its text equals a held-out row's (20 rows), and the dry run passes the leakage gate. The trace count stays at `max_prompts`, so the cost does not change | train | none left | 13.7 to 15.9, and 9.9 to 11.8 | 4.6 and 3.8 (`medgemma-27b-text`) | the 2026-07 cells of 3.1 and three baseline conditions, A2, A3 | HomM-Q2, MR-W2, FzpA-Q3 |
| R16 (C) | E26 | `healthbench__mistral-small-24b__bodhi-no-calibration`, `healthbench__mistral-small-24b__bodhi-no-questions`, `healthbench__mistral-small-24b__bodhi-no-abstention` | train | none left | 10.0 to 12.5 each | none | four conditions | none asked |
| R17 (B) | E03, H1 and H7 secondary (ANALYSIS_PLAN.md section 12, 2026-10-10) | the judge classifier on the eight cells of `cross_cells` in `configs/report.yaml`; on a planned cell, after its grade stage | classify, on recorded cells | the 300 hand labels (two authors, no GPU) | none | 5 to 9 | one call per reply at each cell's judge speed: the dimension times of 3.1, A4 | MR-W3, HomM-W3, PROV-7, C30; withdrawals W23 |

**R17, the judge classifier.** It replaces the keyword rules that the 2026-10-10 entry of
ANALYSIS_PLAN.md withdrew: a pinned judge answers, for each reply, whether it asks the
person for information and whether it sends them to urgent care, with a semantic rubric
and no keyword list. Its rates count only if it passes the hand-label gate of that entry
(Cohen's kappa at least 0.80 against two authors' settled labels, lower bound at least
0.70). The hand labels need no GPU and come first. The stage exists since 2026-10-11
(`bodhi_lora/pipeline/classify.py`, the template and the gate in
`bodhi_lora/grading/classify.py`; `tests/test_grading_classify.py` and the stub run in
`tests/test_end_to_end.py`). Each of the eight cells already lists a judge, so the command
per cell is `python -m bodhi_lora run <cell id> --stages classify --judges <its judge>
--gpus 0`; the Med42-8B cells list two judges, and `--judges` picks one. The runner runs
the stage only when `--stages` names it, so no other run here includes it.
Each recorded cell has about 4,000 replies. At the 8B evaluator's dimension times of 3.1
and a judge three times slower (A4), a cell costs 0.6 to 1.1 H100-hours with its vLLM
starts; the eight cells cost about 5 to 9. An API judge is the other route; its price
was not checked.

**Smaller records.** Some records cost more than the primary tests of
[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) need. A record can change only before its cell first
runs, because the runner freezes it then. Each change below is the first author's
(BLOCKED.md item 1):

- The labelled cells evaluate five conditions on 775 prompts per seed. H7 and H11 need
  only `base` and `lora`, with `base+bodhi` as the teacher. With those three conditions,
  R4 costs 7.7 to 11.9 and the labelled cell of R7 costs 8.1 to 16.1.
- The paired cells evaluate seven conditions. The primary test of H3 needs only `base`
  and `lora`; the teacher and the prompt baselines feed its secondary analyses. With two
  conditions, R3 costs 2.5 to 3.9 and the paired cell of R7 costs 2.6 to 5.4.
- Judges need no change to a record: run with `--judges none`, and add them later with
  `--stages grade`. The judges are about 78 of the H100-hours of all tiers.

The runner also evaluates `base` again for every seed, although with fixed eval ids and
greedy decoding its output is the same for each seed. A runner option to evaluate such a
condition once would save nearly half of the time of a labelled cell with three conditions.

### 4.1 Dry runs on 2026-10-11

Each run's dry run, at seed 42, on this machine, into a scratch results root
(`python -m bodhi_lora run <cell id> --dry-run --seeds 42 --results-root <scratch>`; on a
recorded cell with `--stages grade`, `classify` or `probe` and its `--judges`). A dry run
prints every GPU command, then runs the CPU steps with the stub engine and does not train.

| Run | Dry run |
|---|---|
| R1 | passes (both Med42-8B cells, `--stages grade --judges gemma-2-27b`; the stub judge grades the stored replies) |
| R2 | passes |
| R3 | stops after the GPU commands: the MediQ file is not local (fetch needs the go-ahead, BLOCKED.md item 2) |
| R4 | passes |
| R5 | the training cell passes; its paired cell stops at the MediQ fetch, as R3 |
| R6 | passes (checked on the 2026-05 cell with `phi-4` and the MedQuAD cell with `medgemma-27b-text`) |
| R7 | the training cell and its labelled cell pass; its paired cell stops at the MediQ fetch, as R3 |
| R8 | stops after the GPU commands: the MedQA rows are built from Hugging Face, which is a download (`python -m bodhi_lora fetch medqa-open`, go-ahead needed) |
| R9 | both cells pass |
| R10 | both cells stop after the GPU commands: the NLM files are not local (fetch needs the go-ahead) |
| R11 | passes (checked on the Mistral-Small-24B HealthBench cell, `--stages probe`) |
| R12 | both cells pass |
| R13 | all three cells pass |
| R14 | passes |
| R15 | both cells pass |
| R16 | all three cells pass |
| R17 | passes (checked on two cells, `--stages classify --judges medgemma-27b-text`) |

Until 2026-10-11 a dry run downloaded a missing benchmark, so the MedQA dry run passed by
downloading it; now a dry run never downloads. After the fetches, rerun the five dry runs
that stop.

## 5. Minimal first batch

If compute is tight, run these first: about **32 to 43 H100-hours** with the records as
they stand, or **24 to 31** with the R3 record cut to `base` and `lora`:

1. R1, the Med42-8B judge re-grade (6.7). It needs no training and closes the D6
   promise.
2. R2, the uniform Mistral-Small-24B cell with the baselines and red flags first, with
   `--judges none` (15.0 to 20.2 with the bench run).
3. R3, the MediQ paired probe on the R2 adapters (9.9 to 16.3; 2.5 to 3.9 with `base`
   and `lora` only, which gives the primary test of H3 and none of its secondary ones).

Together with E02 on the existing outputs (no GPU), this answers the reasons for
rejection that a run can answer: a widely used general base under one set of settings
(MR-W1, MR-W2, FzpA-W2), a prior-work comparison (MR-W6, HomM-W4, C27), calibration
against mimicry (MR-W4, LxMF-2), red flags first (C31), the Med42-8B grader overlap (D6),
and measured cost (C25). It leaves the second protocol (R5), the clinical judge panel
(R6), the labelled probes (R4) and the MedGemma rerun (R7) for the next batch.

To add the judge of R2 later: when R2 is complete, its record gets
`report.status: complete`, and
`python -m bodhi_lora run healthbench__mistral-small-24b__bodhi --stages grade --gpus 0`
adds the judge (9.2 for `medgemma-27b-text`).

## 6. Totals

| Set | H100-hours | At 2 to 4 dollars per H100-hour |
|---|---|---|
| Minimal first batch, as the records stand | 31.6 to 43.2 | 63 to 173 dollars |
| Minimal first batch, R3 with `base` and `lora` only | 24.1 to 30.8 | 48 to 123 dollars |
| Tier A | 158.6 to 249.4 | 317 to 998 dollars |
| Tier A, labelled cells with three conditions | 130.7 to 189.5 | 261 to 758 dollars |
| Tiers A and B | 234.2 to 350.4 | 468 to 1,402 dollars |
| All tiers | 319.8 to 456.2 | 640 to 1,825 dollars |

The rows for Tiers A and B and for all tiers include R17 (Tier B, 5 to 9 H100-hours),
added on 2026-10-10. On the same day R11 lost the MedQuAD probe (0.25 to 0.5). The minimal batch holds one judge, R1. Every other row includes every judge that its
records list: 77.7 H100-hours over all tiers, 49.0 of them in Tier A. Most of the rise from the first
version of this plan (2026-10-08) comes from what the records hold: the judges' harm
call, seven conditions in the paired and NLM cells, five in the labelled cells, and the
prompt baselines in the ChatDoctor, MedQuAD and MedQA cells.

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
   `make analyze` (ARCHITECTURE.md section 5). Then `make numbers` writes the cell's
   table (`tab_cell_<token>`) and macros. Add the cell to `cross_cells` in
   `configs/report.yaml`, and to a group that each cross-cell table prints in its
   `blocks` (or to that table's `leave_out`, to keep it out on purpose): the tables list
   groups, and the report stops while a cross cell is in none of a table's blocks. The
   rule-based groups take the cell in by themselves. No code changes.
4. Commit the tracked files: the manifests, `cell.yaml`, the id lists, the response
   tables and the cell's analysis folder. Git ignores the raw outputs by kind.
5. Read the results only by the rules of ANALYSIS_PLAN.md, and record any change of plan
   in its section 12 before the analysis runs.
