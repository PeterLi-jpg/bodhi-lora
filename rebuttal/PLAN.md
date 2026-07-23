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

### Benchmarks (HealthBench-Hard stays; add 2)
1. **MedQA-USMLE, open-ended reframe** — clinician-facing board vignettes with options stripped.
   Directly answers FzpA Q3 ("how does calibration work when clinicians pose the questions").
   Complete-info vignettes also stress-test appropriate NON-over-asking (rebuts mimicry).
2. **MedQuAD** (NIH consumer-health QA) — patient-facing, different source than HealthBench ->
   source-generality of the patient-facing result.

Optional / stretch: **MediQ** (missing-info clarifying-question benchmark, tailor-made for the
mimicry rebuttal); **MIMIC/eICU-derived clinician questions** (what FzpA literally suggested) —
NOTE: the MIMIC on the cluster is `mimic-iv-echo` (echocardiogram VIDEOS), not text notes, so this
needs credentialed PhysioNet access to MIMIC-IV notes / eICU tables. Not available from what's
installed. Build the loader, gate on the user's PhysioNet data.

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

**Feasibility note:** 8x H100 makes 20 full 24B/7B runs very doable IF we can claim the GPUs, but
the node is shared and was mostly busy on 2026-07-23 (only GPU 6 idle). Coordinate GPU access
before launch; with N free GPUs we can run N seeds/cells in parallel. If access stays limited,
keep 5 seeds on cell 1 (Mistral-Small-24B x HealthBench) and 3 on expansion cells; name what's cut.

## Cluster setup prerequisites (before any run)

- [ ] Clone the repo to the box (`/data/PROJECTS/bohdi` or `~`); it is NOT there. `git` from the
      private repo or the anon 4open.science mirror.
- [ ] Create a project venv (`python3 -m venv .venv`; base python has no torch) and `bash setup.sh`;
      add a post-install import smoke check (vllm, peft, trl, transformers).
- [ ] Gated HF access on the box: MedGemma-27B-text-it (paper used `-text-it`; cache has `-it`),
      Mistral-Small-24B-Instruct, BioMistral-7B, Qwen2.5-14B-Instruct (filter), Llama-3.1-8B (eval).
- [ ] Download the 2 new benchmarks (MedQA, MedQuAD) — small public HF downloads; none installed.

## Engineering task list (code on this branch)

- [ ] `scripts/generate_traces.py`: add benchmark loaders (MedQA-open, MedQuAD) + DATASET_URLS.
      The BODHI wrapper is already model-agnostic (`chat_fn`), so `--model` swap is free.
- [ ] Benchmarks without HealthBench rubrics: rely on the benchmark-agnostic epistemic grader
      (`scripts/eval_epistemic.py`) for the 7 dimensions; use an ideal-answer / reference rubric
      or LLM-rubric for an aggregate-quality proxy where no native rubric exists.
- [ ] `configs/`: `lora_mistral7b_gpu.yaml`, `lora_biomistral7b_gpu.yaml` (PEFT target_modules for
      Mistral: q/k/v/o/gate/up/down proj — same 7 as MedGemma, different names).
- [ ] `rebuttal/launch/run_cell.sh`: single-H100 driver (generate->filter->train 5 seeds->eval->grade).
- [ ] clinical-judge re-grade path (second `--grader-model`).

## No-H100 analyses (run locally against `results_modal/`)

- [x] `rebuttal/analyses/mimicry_split.py` — W4. Result: Base does NOT reliably discriminate
      (CI includes 0); Wrapper AND LoRA both DO (theme-only: LoRA +13.2pp, CI [+4.1,+21.7]).
      Adapter reproduces the wrapper's targeting; inconsistent with template mimicry.
- [ ] `rebuttal/analyses/interference_mechanism.py` — LxMF W4. Show the LoRA+BODHI red-flag drop
      is Pass-1-analysis-format leakage, not context truncation (correlate drop with output
      length / format-header regex / parse_failure; show non-leaked responses still regress).
- [ ] `rebuttal/analyses/prior_work_baseline.py` — W6. Inference-time verbalized-confidence /
      clarifying-question prompt on the existing eval set; compare to BODHI-LoRA.

## Rebuttal text deliverables

- [ ] `rebuttal/responses/` — per-reviewer response (FzpA, LxMF, HomM) + a common section.
- [ ] Paper edits (on this branch, for the revised PDF): scope the "framework" claim; Fig 2 split
      0-2 vs rate panels; Table 2 red-flag caption (higher = better, no down-arrow); explain 7-vs-5
      dims + clinician input; explain Table 1 n differences; rewrite vague "not-statements."

## Open inputs needed from the user
- Rebuttal **deadline** (sizes the run matrix).
- Confirm model-size vs. number-of-cells tradeoff (7-8B recommended for full 5-seed coverage).
- PhysioNet credentials? (gates the MIMIC/eICU option.)
