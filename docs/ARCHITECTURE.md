# Architecture

Design of the resubmission code, written on 2026-10-08 and brought up to date when the
build lanes merged on 2026-10-09. Sections 3 to 6 and 9 describe code that is in the
repo. Section 7 is built up to `analyze`; the report (`python -m bodhi_lora report`) is
not built yet, so `make numbers` still writes the paper's numbers. Section 11 records how
the lanes ran. [EXPERIMENTS.md](EXPERIMENTS.md) lists the runs and analyses this design
must carry.

## 1. Why the code changes

The code grew in two review phases, and each phase left its own launcher, its own
results folder and its own analysis. The result is not general:

- **Two launch paths and six result locations.** The submitted MedGemma-27B cell ran
  through `legacy/modal_eval.py` into `02ac9a3:results_modal/`. The cells added in the
  discussion ran through `02ac9a3:rebuttal/launch/run_cell.sh` into `02ac9a3:results_rebuttal/`. A pilot
  run sat in the ignored results/seed_42/, analysis outputs in `02ac9a3:rebuttal/results/`,
  and copied scores in `02ac9a3:data/derived/` and `02ac9a3:data/rebuttal_cells/`.
  Section 10 records the move into one results root.
- **Analysis special-cases one cell.** `scripts/paper_stats.py` computes the full
  statistics only for the submitted cell (`MAIN_CELL_ID`, a fixed `CONDITIONS` tuple).
  The added cells get a subset through `rebuttal/analyses/cell_values.py` and
  `rebuttal/analyses/quality_drop.py`. The discrimination, leakage and question-count
  analyses read `02ac9a3:results_modal/` only.
- **Conditions are code, not data.** `scripts/aggregate_seeds.py` knows four BODHI
  condition names, so a cell run with `PROTOCOL=ask_first` cannot be aggregated.
- **Names and paths are literals in code.** The cell table in
  `02ac9a3:rebuttal/launch/run_grid.sh`, the benchmark case statement in
  `02ac9a3:rebuttal/launch/_common.sh`, the model tokens in `scripts/paper_stats.py`, the
  dataset-keyed row builders in `02ac9a3:scripts/build_benchmark_jsonl.py`, and
  `DATA_DIR = Path("data/raw")` in three scripts.
- **Settings live in four places.** `configs/*.yaml`, `02ac9a3:rebuttal/configs/`, the
  `PROFILE` blocks of `02ac9a3:rebuttal/launch/_common.sh`, and script flag defaults.
- **A name collision.** The installed `bodhi-llm` package imports as `bodhi`
  (`02ac9a3:scripts/_bodhi_ablation.py` does `from bodhi import BODHIConfig`). A repo-root
  package named bodhi would hide it. The package is therefore `bodhi_lora/`, after
  the repository.

## 2. Terms

One term for each thing. Code, configs, docs and the paper use these words.

| Term | Meaning |
|---|---|
| cell | One benchmark x one base model x one teaching protocol, run over a list of seeds. The unit of every run and every analysis. |
| cell id | `<benchmark>__<model>__<protocol>`, plus `__<tag>` when another cell has the same three parts. The tag is the run month for a run made under earlier settings (`2026-05`), or a design tag for a planned variant (`unfiltered`). |
| protocol | A way to produce a reply at inference: a wrapper (one or more model calls, such as BODHI or ask-first) or a prompt (one instruction block, such as the clarifying-question baseline). |
| teacher | The model and protocol whose traces the student trains on. By default the cell's own base model under the cell's protocol. |
| condition | The weights (`base` or `lora`) plus an optional protocol at inference: `base`, `base+bodhi`, `lora`, `lora+bodhi`, `base+clarify`. |
| filter grader | The model that grades teacher traces against the benchmark rubric. |
| evaluator | The model that grades eval responses: the benchmark rubric and the seven epistemic dimensions. |
| judge | An extra grader over stored eval responses. It never changes the evaluator's scores. |
| eval pool | The rows a cell evaluates on by default. No row of it is ever trained on. |
| probe pool | Rows held out of training by label, so a labelled probe set stays unseen. A cell evaluates on them only when its eval ids name label groups. |
| train pool | The rows a cell may generate teacher traces for. |
| label | A binary fact about a benchmark row that the graders never see, such as a physician's judgement that the prompt lacks context. |
| stage | One step of a cell run: `probe`, `traces`, `filter`, `train`, `eval`, `grade`, `bench`. |

## 3. Registry

All settings are data in `configs/`. Four kinds of record, one file each, named after
the record's id:

| Directory | Record | Holds |
|---|---|---|
| `configs/models/` | `ModelConfig` | HF repo, pinned revision, family, domain, size, default precision, chat-template facts, licence. |
| `configs/benchmarks/` | `BenchmarkConfig` | Loader, pinned source, audience, task type, rubric kind, prompt template, split rules, label rules. |
| `configs/protocols/` | `ProtocolConfig` | Kind (wrapper or prompt), implementation key, pinned package, passes, instructions, leak rules, reference. |
| `configs/cells/` | `CellConfig` | Benchmark, split, model, protocol, seeds, graders, and the trace, filter, train, eval and report settings. |

Two more files hold project settings: `configs/project.yaml` (the results root, the data
root, and the scratch root where stub and dry runs write) and `configs/report.yaml` (the
grader-free text rules, the labels of the discrimination index, the cells and groups of
the cross-cell statistics, and the posted values to check). Nothing in the package names
a model, a benchmark or a results path (section 9).

### 3.1 Validation

`bodhi_lora/registry/schema.py` defines the records with pydantic 2. Every record
forbids unknown keys, so a misspelled key fails at load time instead of being ignored.
Each record checks its own fields (examples: a fixed tau needs a value; a QLoRA adapter
served on a full-precision base needs a stated reason, audit REPRO-12; a cell id must
start with its benchmark, model and protocol). `load_registry()` reads every file,
checks that each file name is its id, and then `validate_registry()` checks the
references between files and the grader-family rule. All problems are reported in one
error.

The grader-family rule (`family_problems()`, convention 5 in
[CONVENTIONS.md](CONVENTIONS.md), decision D6):

- **R1.** The filter grader and the evaluator come from different families.
- **R2.** Neither the base model nor the teacher shares a family with the filter grader
  or the evaluator, unless `grading.same_family_waiver` gives a reason. A waived cell is
  marked in every table and stays out of cross-model statements.
- **R2b.** A waived cell lists at least one judge from a family outside all four.
- **R3.** No judge shares a family with the base model, the teacher or the filter grader,
  and no judge is the evaluator. A clinical judge from the evaluator's family is allowed:
  it compares clinical and general grading at a fixed family.

### 3.2 Examples

These show the shape. The foundation lane copies the values from `02ac9a3:configs/families.yaml`,
`02ac9a3:rebuttal/configs/`, `02ac9a3:rebuttal/launch/_common.sh` and `legacy/README.md`, and checks
each against its source. Together with the records they reference (the graders, the
`bodhi` protocol), the examples pass `load_registry()` as written.

```yaml
# configs/models/mistral-small-24b.yaml
id: mistral-small-24b
hf_id: mistralai/Mistral-Small-24B-Instruct-2501
revision: 9527884be6e5616bdd54de542f9ae13384489724
last_modified: "2025-07-28"
family: mistral
domain: general
params_b: 24
chat_template:
  response_template: "[/INST]"
  tokenizer_kwargs: {fix_mistral_regex: true}
license: apache-2.0
```

```yaml
# configs/benchmarks/healthbench.yaml
id: healthbench
title: HealthBench (Arora et al., 2025)
loader: rubric_jsonl
source:
  kind: url
  # sha256 of the copies setup.sh downloaded to the data root; the data lane checks
  # them against a fresh download.
  files:
    full:
      url: "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
      sha256: e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f
    hard:
      url: "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
      sha256: b0320430e5cd974e746585594c1dd10b5a3fc2aff9c72b26106c2c4a069d74e9
license: MIT
audience: mixed
task: open_ended
rubric: native
splits:
  # The split of the 2026-05 and 2026-07 cells: evaluate on Hard, train on the rest.
  hard: {method: named_files, eval_file: hard, train_file: full}
  # The split for new cells. The emergency-labelled prompts (both groups) and the
  # clinician-user prompts leave the train pool (397 + 282 of its 4,000 rows) and form
  # the probe pool. Cells still draw eval ids from Hard unless eval.ids.only says so.
  default:
    method: named_files
    eval_file: hard
    train_file: full
    hold_out:
      - {label: emergent-physician}
      - {label: clinician-user, groups: [positive]}
labels:
  needs-info-physician:
    description: HealthBench physicians agreed that the prompt lacks the context or information a safe reply needs
    source: HealthBench physician-agreed categories
    timing: pre_specified
    positive_tags: ["physician_agreed_category:not-enough-context", "physician_agreed_category:not-enough-info-to-complete-task", "physician_agreed_category:context-matters-but-unclear"]
    negative_tags: ["physician_agreed_category:enough-context", "physician_agreed_category:enough-info-to-complete-task", "physician_agreed_category:context-matters-is-clear", "physician_agreed_category:context-does-not-matter"]
  emergent-physician:
    description: HealthBench physicians agreed that the case is emergent or conditionally emergent (positive) or not emergent (negative)
    source: HealthBench physician-agreed categories
    timing: pre_specified
    positive_tags: ["physician_agreed_category:emergent", "physician_agreed_category:conditionally-emergent"]
    negative_tags: ["physician_agreed_category:non-emergent"]
  clinician-user:
    description: HealthBench physicians agreed that the user is a health professional
    source: HealthBench physician-agreed categories
    timing: pre_specified
    positive_tags: ["physician_agreed_category:health-professional"]
    negative_tags: ["physician_agreed_category:not-health-professional"]
  needs-info-theme:
    description: the context_seeking theme tag only (the theme-only labelling of the 2026-07 discussion)
    source: HealthBench theme tags
    timing: post_hoc
    positive_tags: ["theme:context_seeking"]
```

```yaml
# configs/protocols/clarify.yaml
id: clarify
title: Clarifying-question prompt
kind: prompt
impl: instruction
passes: 1
instructions: >-
  You are answering a health question. If facts that would change your advice are
  missing, ask the person for them with short, direct questions. Ask before you give
  advice that depends on those facts. If you can give safe general advice now, give it,
  and say how the answers to your questions would change it. If no key facts are
  missing, answer directly and do not ask questions.
reference: Clarifying-question prompting baseline (EXPERIMENTS.md E12)
```

```yaml
# configs/cells/healthbench__med42-8b__bodhi__2026-07.yaml
id: healthbench__med42-8b__bodhi__2026-07
benchmark: healthbench
split: hard
model: med42-8b
protocol: bodhi
seeds: [42, 7, 13, 99, 101]
grading:
  filter: qwen2.5-14b
  evaluator: llama-3.1-8b-nous
  judges: [gemma-2-27b, medgemma-27b-text]
  same_family_waiver: Run before the family rule existed; the base shares the evaluator's Llama-3 family (D6).
traces:
  max_prompts: 4200
  max_new_tokens_per_pass: 2048
  serving: {max_model_len: 8192, max_num_seqs: 64, concurrency: 32, request_timeout_s: 180}
filter:
  tau: {rule: fixed, value: 0.4}
  split_policy: once
  split_seed: 42
  serving: {max_model_len: 8192, max_num_seqs: 64, concurrency: 16, request_timeout_s: 180}
train:
  engine: peft
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  epochs: 3
  per_device_batch: 4
  grad_accum: 4
  effective_batch: 16
  lr: 1.0e-4
  scheduler: cosine
  warmup_ratio: 0.05
  max_seq_length: 4096
  gradient_checkpointing: true
  checkpoint: best_val_loss
  hardware: 1x H100 80GB
eval:
  conditions: [base, base+bodhi, lora, lora+bodhi]
  ids: {policy: fixed_file, n: 200, file: eval_ids.json}   # in the cell's shared/inputs/
  base_precision: model
  max_new_tokens: 1024
  max_new_tokens_per_pass: 2048
  serving: {max_model_len: 8192, max_num_seqs: 64, concurrency: 16, request_timeout_s: 180, drain_s: 30}
report:
  label: HealthBench-Hard, Med42-8B
  token: HBMedFortyTwoJul
  status: complete
```

A cell whose adapters were trained outside this package records the recipe with
`train.engine: external` and `train.record` (where the recipe is documented, for the
2026-05 cell `legacy/README.md` section 3). `run` refuses such a cell; `analyze` and
`report` treat it like any other. An eval-only cell sets `adapters_from` to the cell
whose adapters it evaluates, and has no `traces`, `filter` or `train` block. A teacher
swap sets `traces_from` to the cell whose teacher traces it trains on.

Fields the experiments lane added (each is checked by the schema):

- `grading.judge_serving`: the judges' engine settings; null means `eval.serving`. A
  judge prompt holds the conversation, the reply and the template, so it can need a
  longer context than the eval.
- `bench.seeds`: the seeds whose adapters the `bench` stage times, at the eval's token
  limits. They must be seeds of the cell.
- `eval.ids.unit: pair`: `n` counts pairs of a paired benchmark, and both rows of each
  drawn pair are evaluated.
- `rubric: none` on a benchmark: its rows have no reference criteria, the eval records
  no rubric score, and `validate_registry()` refuses a cell that would train on it.
- `source.files[].sha256: null` with `unpinned_reason`: a file whose hash nobody has read
  yet. `python -m bodhi_lora fetch` prints the hash to pin, and each run's inputs
  manifest records the hash it read.
- `probe` on a recorded cell: the contamination probe adds files only, so the runner
  allows it there, like a judge re-grade.

## 4. Package

One importable package at the repository root, run as `python -m bodhi_lora`. No
install step: the repo root is on `sys.path` for the CLI and for the tests.

```
bodhi_lora/
  __init__.py
  __main__.py      python -m bodhi_lora
  cli.py           subcommands list, fetch, run, analyze, report (report not built)
  paths.py         results and data roots from configs/project.yaml; CellPaths
  manifest.py      run manifests                       <- 02ac9a3:scripts/run_manifest.py
  registry/
    schema.py      records, validate_registry, load_registry
  engine/
    vllm.py        vLLM OpenAI server as a subprocess  <- 02ac9a3:scripts/_vllm_engine.py
    stub.py        the CPU stub engine for dry runs     <- 02ac9a3:scripts/_vllm_engine.py
  data/
    loaders.py     loader registry by file format: rubric_jsonl, hf_qa,
                   paired_context_jsonl, xml_records
    rubric.py      the synthesized multi-criterion rubric  <- build_benchmark_jsonl.py
    splits.py      pools, holdouts, eval-id draws           <- make_bootstrap_eval_ids.py
    labels.py      label rules on rows
    leakage.py     train and eval overlap gate              <- check_dataset_overlap.py
  protocols/
    base.py        Protocol interface and Output record
    bodhi_llm.py   BODHI via bodhi-llm, section ablations, red-flags-first order
                                                        <- 02ac9a3:scripts/_bodhi_ablation.py
    ask_first.py                                        <- 02ac9a3:scripts/baselines/ask_first.py
    instruction.py prompt protocols                     <- 02ac9a3:scripts/baselines/prompts.py
    abstain.py     impl abstain: confidence check, then ask or answer (mediq-abstain)
  grading/
    rubric.py      rubric grader prompt, parse, retries <- 02ac9a3:scripts/filter_traces.py
    epistemic.py   seven-dimension grader               <- 02ac9a3:scripts/eval_epistemic.py
    harm.py        judge harm axes: extent, likelihood, delayed care (C33 proxy)
  pipeline/
    runner.py      stage order, seeds, GPU assignment, resume, manifests
    probe.py       contamination probe                  <- 02ac9a3:scripts/contamination_probe.py
    traces.py      teacher traces                       <- 02ac9a3:scripts/generate_traces.py
    filter.py      grade, tau, train and val split      <- 02ac9a3:scripts/filter_traces.py
    train.py       LoRA and QLoRA with PEFT and TRL      <- 02ac9a3:scripts/train_lora.py
    eval.py        responses and rubric grades per condition
                                                        <- 02ac9a3:scripts/eval_healthbench.py
    grade.py       evaluator dimensions and judges over stored responses
                                                        <- eval_epistemic.py, regrade_cross_family.py
    bench.py       measured time and tokens per condition  <- 02ac9a3:scripts/latency_benchmark.py
  analysis/
    config.py      configs/report.yaml as typed records
    extract.py     raw stage outputs to the per-seed response table and prompts.csv
    stats.py       t interval, paired t, Holm, Fisher, clustered bootstrap, Cohen's d
                                                        <- scripts/paper_stats.py
    cell.py        every statistic of one cell, judge agreement included
    cross.py       cross-cell groups, Holm over cells, the posted-value check
  report/          not built: numbers.tex, numbers.json, tab_*.tex and figures from the
                   analysis outputs (scripts/paper_stats.py and
                   scripts/build_paper_figures.py write them until then)
```

Physician agreement stays in `scripts/compute_irr_kappa.py` (`make kappa`); the analysis
reads its output, `results/<cell_id>/validation/kappa_results.json`, into the statistics.

The arrows name the existing file whose logic each module reuses. Each lane moves the
file with `git mv` and then edits it, so `git log --follow` keeps the history. The TPU
docker modes of `02ac9a3:scripts/_vllm_engine.py` are not carried over; the TPU path stays in
`legacy/` as the record of the 2026-05 cell.

### 4.1 Interfaces between modules

The foundation step built these (2026-10-09), and the lanes changed them as noted. A
change to one lands at the integrate step. `report` is a stub that raises
`NotImplementedError`.

```python
# bodhi_lora.paths
def project(configs_dir: Path = CONFIGS_DIR) -> Project    # results_root, data_root, scratch_root, interpreters
class Project:
    def cell(self, cell_id: str) -> CellPaths
    def benchmark_dir(self, benchmark_id: str) -> Path      # <data_root>/<benchmark_id>
class CellPaths:                                            # one per cell
    root: Path                                              # <results_root>/<cell_id>
    def shared(self, stage: str) -> Path                    # root/shared/<stage>
    def inputs(self) -> Path                                # root/shared/inputs
    def seed(self, seed: int) -> Path                       # root/seed_<seed>
    def stage(self, seed: int, stage: str) -> Path          # root/seed_<seed>/<stage>
    def eval_file(self, seed: int, condition: str) -> Path  # stage eval, <condition>.json
    def grade_file(self, seed: int, evaluator_id: str) -> Path   # epistemic__<evaluator>.json
    def judge_file(self, seed: int, judge_id: str) -> Path       # judge__<judge>.json
    def adapter_dir(self, seed: int) -> Path                # the seed's train folder
    def table(self, seed: int) -> Path                      # table.csv in the seed folder
    def analysis(self) -> Path; def validation(self) -> Path; def logs(self) -> Path

# bodhi_lora.registry
def load_registry(configs_dir: Path) -> Registry           # every record, validated, family rule
def parse_condition(name: str) -> tuple[str, str | None]   # "lora+bodhi" -> ("lora", "bodhi")
def condition_from_legacy(name: str) -> str                # "lora_no_wrapper" -> "lora" (4.1 rule below)

# bodhi_lora.data
@dataclass(frozen=True)
class Row:
    prompt_id: str
    messages: list                 # chat turns up to the turn to answer
    rubrics: list                  # {"criterion", "points", "tags"}
    tags: list
    labels: dict                   # label name -> True, False or None
    pair_id: str | None            # rows of one pair share it (paired probes)
    positions: dict                # source file key -> row index, so a split keeps file order
LOADERS: dict[str, Callable[..., list[Row]]]                # by loader key
def load_rows(bench: BenchmarkConfig, data_root: Path, fetch: bool = True) -> list[Row]
def split_rows(bench: BenchmarkConfig, split_name: str, rows: list[Row],
               fixed_eval_ids: list[str] | None = None) -> Split   # train, eval and probe pools
def eval_ids(cell: CellConfig, split: Split, rows_by_id: dict[str, Row], seed: int,
             inputs_dir: Path) -> list[str]
def check_leakage(trained: dict[str, str], split: Split, held_texts: dict[str, str],
                  eval_ids_by_seed: dict[int, Iterable[str]], population: Iterable[str]) -> dict
                                                            # raises LeakageError on overlap

# bodhi_lora.protocols
@dataclass(frozen=True)
class Decoding:
    max_new_tokens: int            # the limit of each call
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
@dataclass(frozen=True)
class Output:
    response: str                  # the reply the user sees; the training target of a trace
    analysis: str | None           # a wrapper's internal text, kept for audit
    calls: int
    prompt_tokens: int             # summed over calls
    completion_tokens: int         # summed over calls
    finish_length: bool            # some call stopped at its token limit
    metadata: dict
    token_logprobs: list | None    # when the reply is one call's output
class Protocol:
    def respond(self, engine, messages: list, decoding: Decoding) -> Output
IMPLS: dict[str, type]             # by impl key
def build(protocol: ProtocolConfig | None, model: ModelConfig) -> Protocol
                                   # None: the identity protocol, one call with the messages as given
def decoding_for(proto: Protocol, *, max_new_tokens: int, max_new_tokens_per_pass: int, ...) -> Decoding

# bodhi_lora.engine
class Engine:                      # context manager; the stub has the same surface
    def chat(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
             top_p: float = 1.0) -> str
    def complete(self, messages: list, max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, seed: int | None = None, logprobs: bool = False) -> Completion
    def settings(self) -> dict     # for the manifest
    # The contamination probe continues raw text, with no chat template: token ids with
    # the served tokenizer on /tokenize and /detokenize, then /v1/completions.
    def tokenize(self, text: str) -> list[int]
    def detokenize(self, ids: list[int]) -> str
    def continue_ids(self, prompt_ids: list[int], max_new_tokens: int,
                     temperature: float = 0.0) -> Completion
class ContextLengthError(ValueError)   # the server refused a request longer than its context;
                                   # traces, filter and grade list such items and skip them
def make_engine(model: ModelConfig, *, serving: Serving, adapter: Path | None = None,
                quantize: bool = False, stub: bool = False, ...) -> Engine

# bodhi_lora.pipeline
STAGES = ("probe", "traces", "filter", "train", "eval", "grade", "bench")
def run_cell(reg: Registry, proj: Project, cell_id: str, *, stages: Sequence[str] = STAGES,
             seeds: Sequence[int] | None = None, gpus: Sequence[int] = (),
             mode: str = "gpu",              # "gpu", "stub" or "dry_run"
             results_root: Path | None = None, force: bool = False,
             judges: Sequence[str] | None = None) -> Path    # None: every judge of the cell
# A stub or dry run writes under <scratch_root>/<mode>/, never under the results root, and
# refuses a results_root inside it. A recorded cell takes only the additive stages
# (probe, judge re-grades).
class StageContext:                # what a stage reads: registry, project, cell, paths
    def split(self) -> Split       # the benchmark split of the cell
    def rows_by_id(self) -> dict[str, Row]
# Each stage is also a module: python -m bodhi_lora.pipeline.<stage> --cell <id> [--seed n]

# bodhi_lora.analysis (project and report config default to configs/)
def extract_cell(reg, cell_id: str, proj=None, cfg=None) -> dict   # raw outputs -> each seed's table.csv
def analyze_cell(reg, cell_id: str, proj=None, cfg=None) -> dict   # tables -> the cell's stats.json
def analyze_cross(reg, report_cfg=None, proj=None) -> dict    # every cell's stats.json -> the cross-cell one

# bodhi_lora.report (scripts/paper_stats.py calls it; make numbers writes its output)
def report(reg, cfg, proj) -> Report   # every stats.json -> macros, tables, record for paper/generated/
```

Stage output formats stay compatible with the files that exist: `results/<cell_id>/seed_<n>/eval/<condition>.json`
keeps the fields `02ac9a3:scripts/eval_healthbench.py` writes (`results[]` with `prompt_id`,
`response`, `score`, `criteria_results`, `parse_failures`, `response_token_count`), and
`results/<cell_id>/seed_<n>/grade/epistemic__<grader>.json` keeps the fields of
`02ac9a3:scripts/eval_epistemic.py` (`configs[].examples[]`). New runs add `condition`, `protocol`, and per response
`calls`, `completion_tokens_total` and `finish_length`. Fields are added, never renamed,
so one reader serves the migrated files and the new ones. Inside migrated files the old
condition names (`base_no_wrapper`, `lora_bodhi`) map to the new ones by one rule:
`<weights>_no_wrapper` is `<weights>`, and `<weights>_<protocol>` is
`<weights>+<protocol>`.

## 5. Command line and Makefile

```
python -m bodhi_lora list [models|benchmarks|protocols|cells] [--settings]
python -m bodhi_lora fetch <benchmark> ...
python -m bodhi_lora run <cell_id> [--stages probe,traces,filter,train,eval,grade,bench] [--seeds 42 7]
                                   [--gpus 0,1] [--judges a,b | --judges none]
                                   [--stub-engine | --dry-run] [--results-root DIR] [--force]
python -m bodhi_lora analyze (<cell_id> ... | --all) [--extract]
python -m bodhi_lora report        # says where the report is written (make numbers)
```

- `list --settings` prints the settings that differ between cells, so a reader sees at
  once where the 2026-05 and 2026-07 cells depart from the current ones.
- `run` checks the registry, refuses an `external` cell, writes a manifest per stage,
  and resumes a stage whose output is complete. `--stub-engine` replaces vLLM with the
  CPU stub for generation and grading and runs everything else for real. `--dry-run`
  first prints the GPU command of every step, then runs like `--stub-engine` without
  training. Neither needs a token or a GPU, and both write under the scratch root of
  `configs/project.yaml`, never under the results root. The smoke cell in
  `configs/cells/` has a tiny public base model, so `make smoke` trains it for real on
  the CPU.
- `run --judges a,b` runs only those judges of the cell in the grade stage;
  `--judges none` runs the evaluator only. On a recorded cell, `--stages grade` adds
  judge files and `--stages probe` adds the contamination probe; nothing else runs.
- `analyze --extract` reads the raw stage outputs (local, ignored by git) and writes the
  per-seed response tables. `analyze` without `--extract` reads only the tracked tables,
  so it runs in CI. `--all` takes the cells whose status `configs/report.yaml` lists,
  says which have no tables, and runs the cross-cell statistics once every cell they
  need has its `stats.json`.
- `report` prints where the report is written and exits 2: `make numbers` runs
  `scripts/paper_stats.py`, which calls `bodhi_lora.report.report` and writes its macros
  and tables with its own into `paper/generated/`.

Makefile targets call the CLI, the repository tools in `scripts/` and the LaTeX build.
No target runs pipeline code any other way. The paper targets still call the scripts in
`scripts/` and `rebuttal/analyses/` until `report` exists. The target set when it does:

| Target | Command |
|---|---|
| `make extract` | `python -m bodhi_lora analyze --all --extract` (needs the raw outputs) |
| `make analyze` | `python -m bodhi_lora analyze --all` |
| `make report` | `python -m bodhi_lora report` |
| `make paper` | `make report`, the macro check, then the LaTeX build |
| `make test` | `pytest` |
| `make smoke` | `python -m bodhi_lora run <smoke cell> --stub-engine` |
| `make kappa` | physician agreement from rater files kept outside the repo |
| `make check` | doc paths, secrets scan, registry load |

Today `make extract`, `make analyze`, `make test`, `make smoke`, `make kappa` and
`make check` are as above. `make numbers`, `make figures`, `make cells`, `make rebuttal`
and `make derived` still run the old analysis scripts on the results root; they go away
when `make analyze` and `make report` reproduce their outputs (10.3).

## 6. Results root

One root, `results/`, set in `configs/project.yaml`. One folder per cell, one folder per
seed, one folder per stage:

```
results/
  PROVENANCE.md             how the migration mapped each old file
  provenance.json           one entry per migrated file (below)
  cross_cell/
    stats.json              cross-cell statistics
  <cell_id>/
    cell.yaml               the resolved cell record the run used
    manifest.json           cell manifest: settings, family check, stage index
    shared/
      inputs/               pool and holdout ids, eval ids per seed, source sha256
      traces/               teacher traces when every seed shares them
      filter/               filter grades when every seed shares them
      probe/                contamination probe
    seed_<n>/
      traces/  manifest.json   (points to shared/traces, or holds the seed's traces)
      filter/  manifest.json   train.jsonl val.jsonl (ignored)  scores.csv
      train/   manifest.json   adapter_config.json trainer_state.json   weights (ignored)
      eval/    manifest.json   <condition>.json (ignored)
      grade/   manifest.json   epistemic__<grader>.json judge__<judge>.json (ignored)
      table.csv                the per-response table (tracked)
    analysis/
      stats.json            every statistic of the cell
    validation/             physician grading sheets and kappa, where a cell has them
    logs/                   stage logs (ignored)
```

Until the analysis lane lands, `results/cross_cell/` and the 2026-05 cell's analysis folder
hold the moved outputs of the old scripts (10.2), and that cell's `seed_7/filter/` holds
`filter_scores.json` in place of `scores.csv`. The cells recorded before the package have
a manifest per stage folder that lists the files the move kept, with status `recorded` or
`not_kept`; they are not resume markers, because the runner never reruns those cells.

Every seed has all five stage folders, each with a `manifest.json`. A stage that does
not depend on the seed (teacher traces from greedy decoding, filter grades) writes its
large output once under `results/<cell_id>/shared/`, and each seed's manifest records the
shared path and its sha256. A cell that samples traces per seed writes them in the seed's
own folder. In an eval-only cell the traces, filter and train manifests point to the
cell whose adapters it uses.

**Tracked and ignored.** The rule depends on the kind of file, never on the cell. Tracked:
manifests, `cell.yaml`, ids, `scores.csv`, `adapter_config.json`, `trainer_state.json`,
`table.csv`, the analysis and validation folders, `PROVENANCE.md`, `provenance.json`. Ignored:
traces, filter JSONL, adapter weights, optimizer state, raw eval and grade JSON,
logs. Today the submitted cell's raw eval JSON (80 MB) is tracked and the added cells'
(370 MB) is not. Under one rule both are ignored and every cell has a tracked response
table; history keeps the submitted files at the migration base commit (open decision
O2).

**The response table.** `results/<cell_id>/seed_<n>/table.csv` has one row per
(condition, prompt id), and the analysis reads nothing else. `analyze --extract` writes it from the raw files, with
`table.meta.json` beside it (sha256 of each source file, and a hash of each text-feature
definition). Columns:

| Group | Columns |
|---|---|
| key | `cell`, `seed`, `condition`, `prompt_id`, `pair_id` |
| response | `completed`, `chars`, `tokens`, `calls`, `tokens_total`, `finish_length` |
| rubric | `rubric_score`, `rubric_items`, `rubric_parse_failures`, `axis_<axis>` for each HealthBench axis, `neg_items_met`, `neg_points_lost` |
| evaluator dimensions | `epi_scored`, `epi_parse_failure`, `uncertainty_ack`, `active_inquiry`, `context_seeking`, `red_flag`, `scope_bounding`, `hedging_quality`, `specificity`, `questions`, `red_flag_rate`, `scope_bounded_rate`, `specificity_rate`, `blanket_disclaimer`, `appropriate_hedging` |
| text features | `second_person_q`, `leak_<protocol>_<rule>` (marker count) for each protocol the cell uses |
| labels | `label_<name>`: 1, 0 or empty |
| judges | `j_<judge>_<metric>` for the rubric score and the seven dimensions |

A value a legacy file does not have (for example `finish_length`) is empty, and the
analysis reports it as not measured for that cell.

**Manifests.** `bodhi_lora/manifest.py` keeps what `02ac9a3:scripts/run_manifest.py` records
today (resolved config and sha256, git commit and dirty diff hash, package versions,
model ids with family and revision, adapter and id-list hashes, stage counts, hashed
host name) and adds wall-clock time, GPU model and count, and token totals per stage.
The cost analysis reads these.

**Provenance.** `results/provenance.json` holds one entry per old file:
`{"old": <path>, "new": <path or null>, "kind": "moved" | "extracted" | "superseded" |
"archived", "sha256_old", "sha256_new", "tracked_before", "tracked_after", "note"}`.
`moved` files are byte-identical. `extracted` means a value was read out of the file
into a manifest (for example tau and counts from a host filter log). `superseded` means
an analysis output that `make analyze` now regenerates; the note names the statistic
that replaces it. `archived` means the file went to the archive folder next to the repo.
`results/PROVENANCE.md` explains the mapping in prose and lists the host runs that left
no folder. A test checks that every path tracked at the migration base commit under the
old roots has an entry, and that each `moved` file's sha256 matches.

## 7. Uniform analysis

`analyze` runs the same code on every cell. A statistic is absent only when its input is
absent (no labels, no judge files, a field a legacy file lacks), and then the output says
"not measured" with the reason. Nothing names a cell.

**Statistical unit (D7).** The seed. For each seed and condition, a dimension's value is
the mean over scored responses, and the rubric score is the mean over completions. Every
test pairs conditions within a seed: mean difference, t interval with n_seeds - 1 degrees
of freedom (t(4) with five seeds), two-sided paired t. A seed with no value (for example
a seed whose grading found none of its prompts) leaves the statistic, the degrees of
freedom fall, and the output names the seed (ANALYSIS_PLAN.md section 2.3). Pooled response-level numbers are
descriptive, with prompt-clustered bootstrap intervals (5,000 resamples, fixed seed),
because a prompt can appear in several seeds' draws.

**Contrasts.** Built from the cell's conditions by rule: every condition minus `base`;
`lora` minus `base+<protocol>`; `lora+<protocol>` minus `lora`; `lora` minus each
`base+<baseline>`. Holm runs over the seven dimensions within each contrast. The rubric
score and the rates are secondary and outside the Holm family. Cross-cell families (for
example LoRA minus Base on the rubric score over the cells named in `configs/report.yaml`)
get their own Holm correction.

**Per cell, every time:**

| Statistic | From | Decision or audit item |
|---|---|---|
| Seven dimensions and rates per condition, seed means, SD, t interval | evaluator | D7 |
| Paired contrasts, t(n-1) interval, p, Holm over the seven | evaluator | D2, D7 |
| Pooled Cohen's d with prompt-clustered bootstrap interval | evaluator | D7 |
| Counts: prompts, unique prompts, completions, missing completions, parse failures of both graders | eval, grade | D13, LxMF 1 |
| Worst case: failed grades scored at the undesirable end | evaluator | D13 |
| Blanket disclaimers: counts and two-sided Fisher exact test | evaluator | D10 |
| Discrimination index for every response and label of `configs/report.yaml` with both groups present: the rate in each group, P(response given positive) minus P(response given negative), odds ratio, bootstrap intervals, seed-level intervals, and per contrast the difference of the indices and of each group's rate | labels, evaluator, text | D9, MR-W4 |
| Over-flagging (post hoc): the escalation mention rate on the negative group of `red-flag-at-stake`, prompts whose theme is not emergency referrals and whose rubric says nothing about urgency | labels, text | C30 |
| Format leakage for every leak rule of every protocol used: leak share, dimensions by leak status, red flag by length quartile in non-leaked responses | text, evaluator | D11, LxMF 4 |
| Length and token cost: characters, tokens, calls, total tokens per condition; stage time from manifests | eval, manifests | C25 |
| Askers gap: rubric score of asking minus non-asking responses within a condition | evaluator | withdrawals W7 |
| Rubric axes per contrast | rubric | aggregate-cost decomposition |
| Harm proxy: share of responses that meet at least one negative rubric criterion, and points lost to them | rubric | C33 (proxy only) |
| Grader-free question rate: share of responses with a second-person question | text | PROV-7 |
| Teacher: `base+<protocol>` behaviour and filter clearance, tau, kept and raw counts | eval, filter manifest | HomM follow-up Table 2 |
| Judges: per-criterion kappa against the evaluator, weighted kappa for the 0 to 2 dimensions, Pearson r of the rubric score, and the contrasts recomputed with each judge | judge files | D6, C26 |
| Dimension correlations | evaluator | FzpA Q1 |

**Cross-cell.** One row per cell for each contrast and statistic, the cross-cell Holm
families, the teacher strength against the effect, and each cell's marks (family
waiver, set aside, status). Labels with `timing: post_hoc` are printed as exploratory.

**Report.** `report` turns `results/<cell_id>/analysis/stats.json` and
`results/cross_cell/stats.json` into macros and tables, and `make numbers` writes them
into `paper/generated/` (its README lists them). Macro names follow one rule,
`\stat<CellToken><Metric><Condition><Field>`, with the cell token from `report.token`.
Each analyzed cell gets the same table, `tab_cell_<token>`; the cross-cell table
`tab_cross_cell_checks` has the columns `configs/report.yaml` lists under `cross_table`,
the same for every cell. The macros of `scripts/paper_stats.py` keep their names, so the
paper builds unchanged; moving `paper/main.tex` to the report's macros is the paper's
change.

**Pre-registration.** Convention 3 still holds. A statistic that has not been computed
on a cell's outputs before (for example the physician-label discrimination index) goes
into the dated analysis plan before anyone runs it on real outputs. The build lanes test
the analysis code on fixtures and synthetic tables only.

**State on 2026-10-09.** `analyze` computes, for every cell with tables: the dimensions,
rates, rubric score, length and harm proxy per condition and their contrasts by the rule
above, counts and parse failures, the worst case, blanket disclaimers, length quartiles,
cost, shift correlations, the context-seeking anchor check, rubric axes, themes, the
discrimination index for the labels `configs/report.yaml` lists, format leakage, the
askers gap, filter and recipe records, judge agreement (criterion kappa, Pearson r of the
rubric score, exact agreement per dimension, rubric-score contrasts under each judge),
and the physician kappa file. The cross-cell step adds the groups, Holm over each group's
cells, and the posted-value check. Only the 2026-05 cell has tracked tables; the other
recorded cells wait for `make extract` after the analysis plan is adopted (BLOCKED.md
item 1). Not built yet, each needed by the analysis plan before its cells exist: the
weighted kappa and dimension contrasts under a judge (H8), the judges' harm grades
(C33), the probe and bench outputs (E11, E20), stage times from the manifests (E08),
contrasts between cells (H9, H10, E24), the paired-probe statistics (H3 through E14), and
`lora+red-flags-first` minus `lora+bodhi` (H6), which the contrast rule does not make.

## 8. What stays outside the package

| Path | Keeps | Leaves |
|---|---|---|
| `rebuttal/` | The review record: `rebuttal/responses/posted/`, `rebuttal/responses/drafts/`, `rebuttal/withdrawals.md`, `rebuttal/commitments.md`, `rebuttal/posted_values.json` (moved from `02ac9a3:data/rebuttal_cells/`), `rebuttal/README.md`, `rebuttal/PLAN.md`, `rebuttal/COVERAGE.md` | `02ac9a3:rebuttal/launch/`, `02ac9a3:rebuttal/configs/`, `rebuttal/analyses/` and `02ac9a3:rebuttal/results/` go (code into the package, settings into cell records, outputs superseded). The unposted internal note on the aggregate drop moved to `rebuttal/responses/drafts/FINDINGS_quality_drop.md`, and the candidate references to `paper/prior_work_candidates.bib`. |
| `legacy/` | The launcher, trainer, exporter, config and eval script of the 2026-05 cell, the old lock file, and `README.md`. `legacy/scripts/prefetch_gemma_safetensors.py` joined them: it is TPU-era tooling (blob 3d21545 at 8262d49), though the frozen launcher does not call it; the trainer downloads the weights itself. | Nothing else. The 2026-07 launcher is cited by commit (`61dada9:rebuttal/launch/run_cell.sh`), not copied. |
| `scripts/` | Repository tools that are not pipeline steps: `check_doc_paths.py`, `check_no_secrets.sh`, `check_paper_macros.py`, `export_anonymous.sh`, `scripts/maintenance/`, `gpu_lease.sh` (host tool, the package never calls it), the physician sheet tools `make_irr_excel.py` and `make_tiebreaker_excel.py`, and the one-off `migrate_results.py`. | Every pipeline and analysis script (section 10). No shims: the Makefile and the CLI are the only entry points, and docs cite module paths. |
| `paper/` | The paper source and `paper/generated/`. | Nothing. |

`02ac9a3:slurm/`, `02ac9a3:gpu/`, `02ac9a3:run_all.sh`, `02ac9a3:smoke.sh` and
`02ac9a3:eval/` went on 2026-10-08: no reported number came from them, and `make smoke`
replaces `smoke.sh`.

## 9. The no-hardcode check

`tests/test_no_hardcode.py` fails when package code names a model, a benchmark or a
results path. It reads every `bodhi_lora/**/*.py` with `ast` and collects all string
constants, docstrings and f-string parts. It fails on any of these:

1. A model `id` or `hf_id` from `configs/models/`, matched case-insensitively as a whole
   token.
2. A benchmark `id`, an HF dataset id or a source URL from `configs/benchmarks/`.
3. A results or data path: `results_<word>` (the config key `results_root` aside), or
   the `results_root` or `data_root` value of `configs/project.yaml` followed by `/` (so
   `results/` and `data/raw` fail, while the word "data" in a docstring does not).
4. A file or folder name under `bodhi_lora/` that contains `modal`, `tpu`, `gcs`,
   `rebuttal`, `submitted`, `posted`, `neurips` or `reviewer`.
5. A list or tuple literal whose elements are the seed list `42, 7, 13, 99, 101` in any
   order.

Comments are not read, so a comment may still explain why a line exists. There is no
allow-list; a needed exception is a design change for the integrate lane. Protocol and
loader implementation keys (`bodhi_llm`, `rubric_jsonl`) are code names and are allowed.

## 10. Migration map

The move ran on 2026-10-08. `scripts/migrate_results.py --apply` moved tracked files
with `git mv` and ignored files on disk, hashed every file before and after, and wrote
`results/provenance.json` (one entry per old file) and `results/PROVENANCE.md`. Nothing
was deleted; a file with no new home went to the archive folder next to the repo
(`../_archive/foundation_2026-10-08/`). `--manifests` then wrote each recorded cell's
`cell.yaml` and a manifest per stage folder, with status `recorded` or `not_kept`.
Old paths below are as of `02ac9a3` when git tracked them; the others were ignored.

### 10.1 Cell ids for existing outputs

| Old location | Cell id | Status |
|---|---|---|
| `02ac9a3:results_modal/` | `healthbench__medgemma-27b-text__bodhi__2026-05` | complete; train `external` (TPU recipe, `legacy/README.md` section 3); adapters not retained |
| `02ac9a3:results_rebuttal/healthbench__mistralai_Mistral-Small-24B-Instruct-2501/` | `healthbench__mistral-small-24b__bodhi__2026-07` | complete |
| `02ac9a3:results_rebuttal/healthbench__m42-health_Llama3-Med42-8B/` | `healthbench__med42-8b__bodhi__2026-07` | complete; family waiver (D6) |
| `02ac9a3:results_rebuttal/healthbench__BioMistral_BioMistral-7B/` | `healthbench__biomistral-7b__bodhi__2026-07` | complete |
| `02ac9a3:results_rebuttal/chatdoctor__mistralai_Mistral-Small-24B-Instruct-2501/` | `chatdoctor__mistral-small-24b__bodhi__2026-07` | complete |
| `02ac9a3:results_rebuttal/chatdoctor__m42-health_Llama3-Med42-8B/` | `chatdoctor__med42-8b__bodhi__2026-07` | complete; family waiver (D6) |
| `02ac9a3:results_rebuttal/medquad__mistralai_Mistral-Small-24B-Instruct-2501/` | `medquad__mistral-small-24b__bodhi__2026-07` | complete |
| `02ac9a3:results_rebuttal/medqa__mistralai_Mistral-Small-24B-Instruct-2501/` | `medqa-open-preamble__mistral-small-24b__bodhi__2026-07` | complete; set aside (W12) |
| `02ac9a3:results_rebuttal/medqa__BioMistral_BioMistral-7B/` | `medqa-open-preamble__biomistral-7b__bodhi__2026-07` | partial (seed 42) |
| `02ac9a3:results_rebuttal/medqa__google_medgemma-27b-text-it/` | `medqa-open-preamble__medgemma-27b-text__bodhi__2026-07` | not run (eval ids only) |
| `02ac9a3:results_rebuttal/medquad__google_medgemma-27b-text-it/` | `medquad__medgemma-27b-text__bodhi__2026-07` | not run (eval ids only) |
| no folder; host logs only | `healthbench__gemma-2-9b__bodhi__2026-07` | not run (stopped in trace generation) |
| results/seed_42/ and results/_rescue/ (ignored) | `healthbench__medgemma-27b-text__bodhi__2026-04` | pilot: 2,999 traces, 838 train rows, seed 42, Qwen grader |

The run month comes from `run_metadata.timestamp_utc` in each cell's eval files
(2026-05-04 for the submitted cell; 2026-07-23 to 2026-07-28 for the added cells;
2026-04-29 for the pilot). `medqa-open-preamble` is the MedQA benchmark record with the
prompt template from before 035d85f; `medqa-open` is the template after it.

### 10.2 Files

`<C>` is the new cell folder from 10.1, `<S>` is `seed_<n>`, `<M>` the 2026-05 cell and
`<pilot>` the 2026-04 cell. A moved file kept its bytes; "superseded" means `make analyze`
replaces it once it reproduces its values (10.3).

| Old path | New path | Kind |
|---|---|---|
| `02ac9a3:results_modal/seed_7/base_no_wrapper.json` (each seed; and `base_bodhi`, `lora_no_wrapper`, `lora_bodhi`) | `results/<M>/<S>/eval/base.json` (and `base+bodhi`, `lora`, `lora+bodhi`) | moved; still tracked (O2) |
| `02ac9a3:results_modal/seed_7/epistemic_scores.json` (each seed) | `results/<M>/<S>/grade/epistemic__llama-3.1-8b.json` | moved; still tracked (O2) |
| `02ac9a3:results_modal/cross_seed_aggregate.json`, `.md` | `results/<M>/analysis/` | moved; superseded |
| `02ac9a3:results_modal/irr/` (sheets, answer key, README) | `results/<M>/validation/` | moved |
| `02ac9a3:results_modal/irr/kappa_results.json` | `results/<M>/validation/kappa_results.json` | moved |
| `02ac9a3:data/irr_instructions/` | `results/<M>/validation/instructions/` | moved |
| `02ac9a3:data/derived/seed_7_filter_scores.json` | `results/<M>/seed_7/filter/filter_scores.json` | moved |
| data/sft_qwen_graded/seed_7_train.jsonl (ignored) | `results/<M>/seed_7/filter/train.jsonl` | moved |
| `02ac9a3:data/derived/seed_42_qwen_grader_scores.json` | `results/<pilot>/analysis/qwen_grader_scores.json` | moved |
| results/seed_42/eval/*.json (ignored) | `results/<pilot>/seed_42/eval/` | moved |
| results/_rescue/raw_traces.jsonl and results/_rescue/sft/ (ignored) | `results/<pilot>/shared/traces/traces.jsonl`, `results/<pilot>/seed_42/filter/` | moved |
| checkpoints/seed_42/ (ignored) | `results/<pilot>/seed_42/train/` | moved |
| results/seed_42/figures/ and three pilot analysis outputs (ignored) | archive | archived |
| `02ac9a3:results_rebuttal/medqa__google_medgemma-27b-text-it/eval_ids.json` (each folder) | `results/<C>/shared/inputs/eval_ids.json` | moved |
| `02ac9a3:data/raw/hard_200_sample_ids.json` | `results/<C>/shared/inputs/eval_ids.json` of the three 2026-07 HealthBench cells, the Gemma-2-9B run and the pilot | moved (one to five) |
| `02ac9a3:results_rebuttal/healthbench__BioMistral_BioMistral-7B/seed_42/ckpt/best/` (each seed: `adapter_config.json`, `trainer_state.json`) | `results/<C>/<S>/train/` | moved |
| `02ac9a3:results_rebuttal/healthbench__BioMistral_BioMistral-7B/seed_42/ckpt/` (the trainer state of each epoch checkpoint) | `results/<C>/<S>/train/checkpoint-<k>/trainer_state.json` | moved |
| the host copy's seed_N/eval/ condition files and epistemic_scores.json (ignored) | `results/<C>/<S>/eval/<condition>.json`, `results/<C>/<S>/grade/epistemic__llama-3.1-8b-nous.json` | moved |
| the host copy's multi_seed_summary.json (ignored, wrong seed glob) | archive | archived |
| the host logs, logs/host/ (ignored) | `results/<C>/logs/`; tau and counts into `results/<C>/shared/filter/manifest.json`; four logs that span cells to the archive | moved and extracted |
| `02ac9a3:rebuttal/results/mimicry_split.json`, `interference_mechanism.json`, `second_person_questions.json` | `results/<M>/analysis/` | moved; superseded |
| `02ac9a3:rebuttal/results/quality_drop.json`, `cell_recipe.json` | `results/cross_cell/` | moved; superseded |
| `02ac9a3:data/rebuttal_cells/generated_values.json` | `results/cross_cell/generated_values.json` | moved; superseded |
| `02ac9a3:data/rebuttal_cells/posted_values.json` | `rebuttal/posted_values.json` | moved |
| `02ac9a3:data/rebuttal_cells/README.md` | `rebuttal/README.md` | merged |
| smoke outputs in checkpoints/, data/sft/ and eval/smoke/ (ignored) | archive | archived |
| `02ac9a3:configs/families.yaml` | `configs/models/*.yaml` | superseded |
| the old configs/lora_*.yaml files, `02ac9a3:rebuttal/configs/` | `train` blocks of `configs/cells/*.yaml` | superseded |

The ignored copy in `../_remote_sync/results_rebuttal` stays where it is; each of its
files has the sha256 of a file that moved (results/PROVENANCE.md).

The scripts that wrote the superseded files read the new layout and still key their
outputs as before (`rebuttal/analyses/host_runs.py`), so `make rebuttal` and
`make cells` reproduce every value: the 2026-05 analyses and `quality_drop.json` byte for
byte, `generated_values.json` and `cell_recipe.json` up to the folder names they record.
Run on the regenerated files, `make numbers` writes the same `numbers.tex` and tables.

### 10.3 Equivalence before removal

An old analysis file is removed only after `make analyze` reproduces its values. The
analysis lane adds a test that compares the new statistics with each superseded file at
its stored precision: the files marked superseded in 10.2 and the macros in
`paper/generated/numbers.tex`. A mismatch is a finding to explain, not a tolerance to
raise.

Done for the 2026-05 cell: `tests/test_analysis_equivalence.py` recomputes it from its
tracked tables and matches `paper/generated/numbers.json` (conditions, contrasts, worst
case, disclaimers, length quartiles, axes, themes, counts, the context-seeking anchor,
shift correlations, reply size, tau calibration) and the cell's `mimicry_split.json`,
`interference_mechanism.json` and `second_person_questions.json`. Two old values differ
by definition, and the test says why: the median of an even count, and a group count the
old script took over the base condition only. Not done: the cross-cell files
(`generated_values.json`, `quality_drop.json`, `cell_recipe.json`), which need the
2026-07 tables, so no old script or Makefile target has been removed.

## 11. Build lanes

The plan of seven lanes ran differently. One foundation step built the registry, the
package, the pipeline and the results root, and did the move of section 10, so the data,
protocols, pipeline and migrate lanes are done. Three lanes follow, in parallel, each in
its own worktree and local branch made from the foundation's last commit. Each lane
writes only the paths it owns, keeps `make test` and `make check` green at each commit,
and runs no GPU job and no paid compute: GPU work is written down as commands with a
cost estimate for the first author to approve (O6). A change to a path a lane does not
own goes in its report and lands at the integrate step.

| Lane | Branch | Worktree |
|---|---|---|
| analysis | `gen/lane-analysis` | `../_wt/analysis` (removed after the merge) |
| experiments | `gen/lane-experiments` | `../_wt/experiments` (removed after the merge) |
| plans | `gen/lane-plans` | `../_wt/plans` (removed after the merge) |

### 11.1 Ownership

| Lane | Owns (writes) | Tests it owns |
|---|---|---|
| foundation (done) | `bodhi_lora/{__init__,__main__,cli,paths,manifest}.py`, `bodhi_lora/registry/`, `bodhi_lora/engine/`, `configs/`, `results/` (the move, `PROVENANCE.md`, `provenance.json`, the recorded manifests), `scripts/migrate_results.py`, `.gitignore`, `Makefile`, `.github/workflows/` | `tests/test_registry_*.py`, `tests/test_no_hardcode.py`, `tests/test_engine_*.py`, `tests/test_manifest.py`, `tests/test_migration_*.py` |
| analysis | `bodhi_lora/analysis/`, `bodhi_lora/report/`, the `analyze` and `report` commands, `configs/report.yaml`, `paper/generated/`, macro names in `paper/main.tex` and `paper/checklist.tex`, `scripts/check_paper_macros.py`, the analysis scripts in `scripts/` and `rebuttal/analyses/` (to replace them), `results/*/analysis/`, `results/*/seed_*/table.csv`, `results/cross_cell/` | `tests/test_analysis_*.py`, the report's tests, and the old `tests/test_stats_*.py`, `tests/test_analyses_*.py`, `tests/test_aggregate_seeds.py` until their scripts go |
| experiments | `bodhi_lora/pipeline/`, `bodhi_lora/grading/`, `bodhi_lora/protocols/`, `bodhi_lora/data/`, new records in `configs/cells/`, `configs/protocols/`, `configs/benchmarks/` and `configs/models/` (status `planned`), `tests/fixtures/pipeline/` | `tests/test_pipeline_*.py`, `tests/test_protocols_*.py`, `tests/test_data_*.py`, `tests/test_grading_*.py` |
| plans | `docs/EXPERIMENTS.md`, `docs/DECISIONS.md`, `docs/CLAIMS.md`, drafts of the dated analysis plan in `docs/` (O4), `BLOCKED.md`, `KNOWN_ISSUES.md`, `RESULTS.md`, `REPRODUCIBILITY.md`, `README.md`, the records in `rebuttal/` | none |
| integrate (last) | `docs/ARCHITECTURE.md`, `scripts/check_doc_paths.py`, `scripts/README.md`, `setup.sh`, and any shared path a lane asked for | runs every test |

The plans lane drafts; the first author decides every methodology question (O4, O5,
O6). A record of status `planned` changes no number in the paper.

### 11.2 Order and hand-offs

1. **foundation** (done 2026-10-09): the registry with records for every model,
   benchmark, protocol and cell; `python -m bodhi_lora list`, `fetch` and `run`; the
   stub engine and the smoke cell; the move into the results root with its provenance
   check; `make check`; the no-hardcode test.
2. **analysis, experiments, plans** in parallel. Analysis builds against synthetic
   tables in its fixtures and then the migrated outputs. Experiments builds against the
   stub engine and dry runs. Plans needs only the docs.
3. Merge order: plans, experiments, analysis. A new cell record from experiments needs
   its plan entry; analysis reads every record at the end.
4. **integrate** last. It merges the lanes, runs `make test`, `make smoke`,
   `make check`, `make numbers` (or `make analyze` and `make report` once they exist) and
   `make paper`, updates this document and `PENDING` in `scripts/check_doc_paths.py`,
   and checks every staged diff against the clinician names file.

### 11.3 Done means

| Lane | Acceptance check, run and reported with its output |
|---|---|
| foundation | `python -m bodhi_lora list cells` lists every cell; `make test`, `make smoke`, `make check` pass; `make numbers` writes `paper/generated/` byte for byte as before the move; `tests/test_migration_provenance.py` passes. |
| analysis | `make analyze` and `make report` reproduce every value in `paper/generated/` and every superseded file at its stored precision (10.3), from the migrated outputs; the old analysis scripts and their Makefile targets are then removed. |
| experiments | Every planned cell in [EXPERIMENTS.md](EXPERIMENTS.md) with fixed settings has a record that loads, and `python -m bodhi_lora run <cell> --dry-run` prints its GPU commands; each GPU item is listed with its command and an H100-hour estimate. |
| plans | A draft of the dated analysis plan for the first author, and every EXPERIMENTS.md entry tied to a cell record, a decision or a BLOCKED.md item. |
| integrate | All of the above from a clean checkout of the merged branch, and `scripts/check_doc_paths.py` passes with no `PENDING` entry left for a path that now exists. |

### 11.4 How the lanes ended (2026-10-09)

- **plans** and **experiments** merged as their branches stood. The integrate step then
  brought the docs in line with the records (cell ids, loader keys, judges), stopped a
  stub or dry run from writing under the results root through `--results-root`, and
  made traces, filter and grade skip a request that does not fit the context window
  instead of asking for a rerun that would fail the same way.
- **analysis** stopped before its first commit. The integrate step reviewed its work,
  checked it against the outputs it supersedes (10.3), and committed the extract and
  analyze parts on `gen/lane-analysis` before the merge. It left out two parts: a post
  hoc red-flag label with an over-flagging rate computed on the 2026-05 outputs
  (convention 3: the analysis plan reads over-flagging from the physician labels, H7),
  and the unfinished report module. Both are kept outside the repo in
  `../_archive/analysis_lane_wip_2026-10-09/`.
- Not done, so its "done means" row above still applies: `make report`, the macro rename
  in `paper/main.tex`, and the removal of the old analysis scripts.

## 12. Open decisions for the first author

- **O1. Package name.** `bodhi_lora`, because `bodhi` is taken by the `bodhi-llm`
  dependency (section 1). Rename only together with the repository slug (D15).
- **O2. Raw outputs in git.** Default: one rule for every cell, so the submitted cell's
  raw eval JSON (80 MB) stops being tracked and stays in history; every cell gets a
  tracked response table. The alternative keeps those 80 MB tracked, which means one
  cell's raw files are in the tree and the others' are not.
- **O3. A versioned archive of all raw outputs** (about 450 MB with the submitted cell),
  so reviewers can check any table. Publishing it is a remote action.
- **O4. The dated analysis plan** (BLOCKED.md item 5) before any new run, and before the
  physician-label discrimination index is computed on existing outputs.
- **O5. Margins for the deployment-risk criterion** (EXPERIMENTS.md E07). Without them
  the criterion reports intervals only.
- **O6. GPU priorities.** EXPERIMENTS.md section 4 groups the runs in tiers with rough
  costs. No GPU runs until you say so.
- **O7. The MedGemma-27B rerun.** Its adapters are treated as not retained, so every new
  experiment on it needs a rerun. On one H100 that means QLoRA (4-bit), which is not the
  bf16 TPU recipe of the 2026-05 cell. The rerun is then a new cell, and the 2026-05
  cell stays the record.
