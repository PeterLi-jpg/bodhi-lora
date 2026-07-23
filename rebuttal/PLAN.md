# NeurIPS 2026 rebuttal — generality experiments (branch `rebuttal/generality-experiments`)

**This branch is for rebuttal experiments and does NOT get merged into `main`.**
It exists to answer the reviewers' generality critique with real additional runs and to
hold the supporting analyses / rebuttal text. Keep everything under `rebuttal/`.

Submission 32818, "Ask Before You Answer." Ratings: FzpA 3 (borderline reject),
LxMF 4 (borderline accept), HomM 2 (reject). Meta-review AC 9irF consolidates 6 weaknesses.

## Meta-review weaknesses -> responses

| # | Meta-review weakness | Response | Compute |
|---|---|---|---|
| W1 | MedGemma not aligned with HealthBench use case | add a general-domain base model | H100 |
| W2 | one pipeline vs. "general framework" claim | +2 base models x +2 benchmarks; scope the claim in text | H100 |
| W3 | automated grading limited / robustness unclear | clinical-judge re-grade + surface physician IRR (kappa) | cheap re-grade |
| W4 | distills calibration vs. mimics favorable responses | answerable-vs-missing-info split (LxMF's exact test) | **done, no H100** |
| W5 | discussion/contextualization lack clarity | prose rewrite + figure/caption fixes | text |
| W6 | no prior-work comparison | inference-time calibration baseline (verbalized-confidence / clarifying-Q prompt) | inference-only |

Per-reviewer extras: FzpA Q1 (7-vs-5 dims provenance), FzpA W3 (Fig 2 dual-scale, Table 2
red-flag arrow), LxMF Q1 (Table 1 n=200/191/192 = BODHI 2-pass exceeding the 4096-tok cap),
LxMF W4 (interference is truncation/attention-dilution, not capacity competition — test on
existing data), HomM/LxMF (why non-clinical judges).

## Locked design

- **Hardware:** Brev instance `mit-critical-data-1` (org "BODHI: GPU Accelerated Epistemic AI"),
  hostname `brev-pzwlliwrl`: ONE node with **8x H100 80GB**, 18TB `/data`, 5.9TB `/ephemeral`.
  NOT the paper's 8x TPU v6e. Training goes through the single-GPU/multi-GPU PEFT/TRL path
  (`gpu/`, `scripts/train_lora.py`), not the TPU tunix/qwix path (Gemma-only).
  **Caveat (2026-07-23): the box is SHARED and busy** — only GPU 6 was idle; GPUs 0-5,7 were
  running other jobs (MODALENS/probmed, a vLLM server on GPU0). Base python has no torch; jobs
  run in project `.venv`s. The repo is NOT checked out on the box yet, and NONE of our text
  benchmarks are installed (only imaging: mimic-iv-echo videos, VQA-RAD, cholect50, PSI-AVA,
  ProbMed). Throughput depends on how many GPUs we can actually claim -> coordinate before launch.
- **Grader families are fixed:** filter = Qwen2.5-14B-Instruct, evaluator = Llama-3.1-8B-Instruct.
  New base models must avoid BOTH families or the paper's asymmetric-grading claim breaks.
- **Seeds:** 5 (42, 7, 13, 99, 101), matching the paper.

### Base models — LOCKED (MedGemma-27B stays as the existing result; add 2)
User chose bigger general + small clinical (2026-07-23):
1. **Mistral-Small-24B-Instruct-2501** — general-domain, ~24B, clean family (not Qwen/Llama).
   Rebuts FzpA W1/W2 (general model matching HealthBench's patient-facing use). 8x H100 handles
   24B LoRA + vLLM comfortably.
2. **BioMistral-7B** — clinical, Mistral family, clean vs both graders. Controlled general-vs-clinical
   pair within Mistral, plus cross-family clinical (Gemma MedGemma vs Mistral BioMistral).
   (If a more current clinical model is wanted: Meditron3-8B / Med42-v2, but both are Llama-3 based
   -> family-adjacent to the evaluator; would need the clinical-judge re-grade + evaluator swap.)

### Benchmarks — LOCKED after vetting (HealthBench-Hard stays; add 2 training + 1 probe)
Two new TRAINING benchmarks, chosen source-diverse (different origins from HealthBench and each other):
1. **MedQA-USMLE, open-ended reframe** (`bigbio/med_qa`, ~12.7k English board-exam Qs, options stripped)
   — clinician-facing (answers FzpA Q3). Complete-info vignettes stress-test appropriate NON-over-asking
   (anti-mimicry) + quality preservation. Filter signal: consistency with the known-correct option.
2. **MedQuAD** (NIH consumer-health QA; 47k pairs, ~16k with answers after the MedlinePlus-copyright
   removals) — patient-facing, different source than HealthBench. Filter signal: NIH reference alignment.

One new EVAL-ONLY calibration probe (no training cell):
3. **MediQ** (Li et al., NeurIPS 2024, CC BY 4.0, github stellalisy/MediQ) — gives PAIRED complete vs.
   incomplete versions of the same clinical cases. Run each trained adapter + base on it and grade
   active-inquiry on the complete-vs-incomplete split: the cleanest exhibit that the adapter asks when
   info is missing and restrains when it isn't (meta-review W4). NOTE MediQ is derived FROM MedQA, so it
   is NOT source-independent — use it as a paired discrimination probe, not a generality data point.

Dropped: **MIMIC/eICU-derived clinician questions** — the cluster's MIMIC is `mimic-iv-echo` (videos),
not text notes; would need credentialed PhysioNet access to MIMIC-IV notes / eICU. Cite as future work.

### Judges (grading robustness — W3, HomM Q4, LxMF Q3)
Keep asymmetric Qwen-14B / Llama-8B unchanged for comparability. ADD one clinical judge
re-grade of the epistemic dimensions (Med42-v2 or a medical-Qwen) as a robustness panel.
Surface the in-progress physician IRR (`scripts/compute_irr_kappa.py`, `results_modal/irr/`).

## Generality run matrix (8x H100 shared node — claim GPUs before launch)

Per (model x benchmark) cell: generate BODHI traces -> filter (Qwen-14B) -> LoRA train (5 seeds)
-> eval 2x2 (base/base+cot/lora/lora+cot) x 5 seeds -> epistemic grade (Llama-8B) + clinical judge.

Cells = {Mistral-7B, BioMistral-7B} x {MedQA-open, MedQuAD} = 4 new cells.
Suggested order (front-load highest rebuttal value):
1. Mistral-7B x HealthBench-Hard  (isolates "general model works" — FzpA W2/W1)
2. Mistral-7B x MedQA-open        (general model + clinician-facing benchmark)
3. BioMistral-7B x HealthBench    (second clinical model — HomM/LxMF)
4. Mistral-7B x MedQuAD, BioMistral x MedQA/MedQuAD as time allows

**Compute constraint (user, 2026-07-23):** ONE GPU at a time, QUEUED (shared box). It's a big
H100 80GB, so pack it hard (QLoRA-24B leaves headroom; high-throughput vLLM) and parallelize
WITHIN the card, but the cells run serially.

## Wall-clock estimate (one queued H100)

Per model x benchmark cell (rough, grounded estimates):

| Stage | 24B/27B cell | 7B cell |
|---|---|---|
| Generate BODHI traces (once/cell, ~4k prompts, 2-pass) | ~1-1.5h | ~0.5h |
| Filter/grade w/ Qwen-14B (once/cell) | ~0.5h | ~0.5h |
| LoRA train x 5 seeds | ~4-8h | ~2-3h |
| Eval 2x2 + epistemic grade | ~2-3h | ~2h |
| **Cell total** | **~8-13h** | **~5-6h** |

Matrix (pure compute, before shared-queue wait):
- Both axes isolated (4 cells: Mi x HB, B x HB, M x MedQA, M x MedQuAD) ~= **1.5-2 days**
- Full 3x3 grid (8 new cells) ~= **3-4 days**
- MediQ eval-only probe: +a few hours.

Calendar time is longer (shared queue). Speed levers: drop the TPU-only pad-to-4096 on the GPU
path (2-3x faster training — a train_lora.py tweak); generate 2k not 4k prompts; fewer seeds on
corner cells. Compute already minimized: generate + filter ONCE per cell; Base / Base+CoT eval
ONCE per cell (they don't depend on the LoRA seed).

## Cluster setup prerequisites (before any run)

- [ ] Clone the repo to the box (`/data/PROJECTS/bohdi` or `~`); it is NOT there. `git` from the
      private repo or the anon 4open.science mirror.
- [ ] Create a project venv (`python3 -m venv .venv`; base python has no torch) and `bash setup.sh`;
      add a post-install import smoke check (vllm, peft, trl, transformers).
- [ ] Gated HF access on the box: MedGemma-27B-text-it (paper used `-text-it`; cache has `-it`),
      Mistral-Small-24B-Instruct, BioMistral-7B, Qwen2.5-14B-Instruct (filter), Llama-3.1-8B (eval).
- [ ] Download the 2 new benchmarks (MedQA, MedQuAD) — small public HF downloads; none installed.

## Engineering task list (code on this branch)

- [x] `scripts/train_lora.py`: two surgical fixes so non-Gemma models train on GPU —
      (1) gate the `Gemma3DecoderLayer` FSDP wrap-class check behind `_ON_TPU` (it hard-crashed
      at model load on GPU for Mistral); (2) support a `data.response_template` config override
      for completion-only masking (Mistral's `[INST]...[/INST]` defeats the auto-detector).
      Both backward-compatible (TPU/Gemma runs unchanged).
- [x] `rebuttal/configs/lora_mistral_small_24b_qlora.yaml` + `lora_biomistral7b.yaml` — written,
      paper hyperparameters (eff batch 16, 3 epochs, lr 1e-4 cosine), `response_template: "[/INST]"`.
      TODO verify the `[/INST]` masking + BioMistral chat_template on first run (loud-fail nets exist).
- [ ] `scripts/generate_traces.py`: add benchmark loaders (MedQA-open, MedQuAD). Convert each to
      the pipeline's `{prompt_id, prompt:[messages], ...}` shape. BODHI wrapper is model-agnostic
      (`chat_fn`), so `--model` swap is free.
- [ ] **Filtering new benchmarks (key design):** the filter step needs a quality signal, and it
      must stay a DIFFERENT family from the Llama evaluator (preserve the asymmetric-grading claim).
      HealthBench uses its rubrics + Qwen-14B. New benchmarks have natural per-item signals graded
      by Qwen-14B: MedQA-open = consistency with the known-correct option; MedQuAD = alignment with
      the NIH reference answer. Implement a `filter_traces.py` benchmark mode that grades against
      these instead of HealthBench rubrics.
- [ ] Benchmark-agnostic EVAL: for MedQA/MedQuAD there is no HealthBench rubric, so the 2x2 eval =
      generate 4-config responses + `scripts/eval_epistemic.py` (the 7 dims, already model/benchmark
      agnostic) + a quality proxy (MedQA answer-accuracy; MedQuAD reference-alignment). `eval_healthbench.py`
      stays only for the HealthBench cell.
- [ ] `rebuttal/launch/run_cell.sh`: one-GPU queued driver, args MODEL + BENCHMARK + SEEDS. Structure
      it to generate + filter ONCE per (model,benchmark), reuse traces across seeds, and run Base /
      Base+CoT eval ONCE per (model,benchmark) (they don't depend on the LoRA seed) — big compute save.
- [ ] clinical-judge re-grade path (second `--grader-model`, e.g. Med42-v2) for W3 robustness.

## No-H100 analyses (run locally against `results_modal/`)

- [x] `rebuttal/analyses/mimicry_split.py` — B1/W4. Result: Base does NOT reliably discriminate
      (CI includes 0); Wrapper AND LoRA both DO (theme-only: LoRA +13.2pp, CI [+4.1,+21.7]).
      Adapter reproduces the wrapper's targeting; inconsistent with template mimicry.
- [x] `rebuttal/analyses/interference_mechanism.py` — C1/LxMF W4. Result: red-flag drop is 54.6%
      Pass-1-format leakage; non-leaked responses hold at 1.73 (>LoRA 1.67); within non-leaked,
      red-flag RISES with length -> refutes the truncation/attention-dilution alternative.
- [ ] `rebuttal/analyses/prior_work_baseline.py` — E2/W6. Inference-time verbalized-confidence /
      clarifying-question prompt on the existing eval set; compare to BODHI-LoRA.

**The definitive concern-by-concern checklist is `rebuttal/COVERAGE.md`** — every reviewer point
mapped to a response + status. Newly surfaced workstreams beyond the generality runs:
clinical-judge re-grade (D1), physician IRR write-up (D2), compute/latency table (G1 via
`latency_benchmark.py`), prior-work baselines (E2), and the clarity/figure/text fixes (F1-F8).

## Rebuttal text deliverables

- [ ] `rebuttal/responses/` — per-reviewer response (FzpA, LxMF, HomM) + a common section.
- [ ] Paper edits (on this branch, for the revised PDF): scope the "framework" claim; Fig 2 split
      0-2 vs rate panels; Table 2 red-flag caption (higher = better, no down-arrow); explain 7-vs-5
      dims + clinician input; explain Table 1 n differences; rewrite vague "not-statements."

## Open inputs needed from the user
- Rebuttal **deadline** (sizes the run matrix).
- Confirm model-size vs. number-of-cells tradeoff (7-8B recommended for full 5-seed coverage).
- PhysioNet credentials? (gates the MIMIC/eICU option.)
