"""Schema of the experiment registry under configs/.

Four kinds of record, one YAML file each, named after the record's id:

  configs/models/<id>.yaml      a model the pipeline can load or grade with
  configs/benchmarks/<id>.yaml  a benchmark: loader, pinned source, split rules, labels
  configs/protocols/<id>.yaml   a teaching protocol or an inference-time prompt baseline
  configs/cells/<id>.yaml       one cell: benchmark x model x protocol over seeds, with
                                its graders and its trace, filter, train and eval settings

pydantic checks each file on its own. Unknown keys are errors, so a misspelled key
fails at load time instead of being ignored. validate_registry() then checks what one
file cannot: that references resolve, and the grader-family rule. docs/ARCHITECTURE.md
gives the design and an example of each record.

API used: pydantic 2.x (BaseModel, ConfigDict(extra="forbid", frozen=True),
model_validator(mode="after"), Field(discriminator=...)), per the pydantic v2 docs
pages "Models", "Validators" and "Unions". requirements*.txt pin pydantic>=2.10,<3,
the floor vLLM 0.9.2 already needs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

# Registry ids: lowercase letters and digits, with single "." or "-" between them,
# e.g. "llama-3.1-8b". One spelling rule for every id keeps file names, cell ids
# and table columns consistent.
_ID_BODY = r"[a-z0-9]+(?:[.-][a-z0-9]+)*"
RegistryId = Annotated[str, StringConstraints(pattern=rf"^{_ID_BODY}$")]
# A cell id is <benchmark>__<model>__<protocol>, plus __<tag> when another cell has
# the same three parts (a run month such as 2026-05, or a design tag such as
# unfiltered). The tag says what differs; the cell file says how.
CellId = Annotated[str, StringConstraints(pattern=rf"^{_ID_BODY}__{_ID_BODY}__{_ID_BODY}(?:__{_ID_BODY})?$")]
# Python registry keys (loader and protocol implementations) are code names.
CodeKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
HfRepo = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")]
GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
IsoDate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]
MacroToken = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Za-z]*$")]
PosInt = Annotated[int, Field(gt=0)]
PosFloat = Annotated[float, Field(gt=0)]
Share = Annotated[float, Field(ge=0, le=1)]

# A condition is the weights the model runs with and, optionally, a protocol applied
# at inference: "base", "lora", "base+bodhi", "lora+ask-first", "base+clarify".
_CONDITION_RE = re.compile(rf"^(base|lora)(?:\+({_ID_BODY}))?$")
WEIGHTS = ("base", "lora")


def parse_condition(name: str) -> tuple[str, Optional[str]]:
    """Split a condition name into (weights, protocol id or None)."""
    m = _CONDITION_RE.match(name)
    if not m:
        raise ValueError(f"condition {name!r}: expected base, lora, base+<protocol> or lora+<protocol>")
    return m.group(1), m.group(2)


def condition_from_legacy(name: str) -> str:
    """The condition that a name from the runs before this package stands for.

    Those runs (2026-04 to 2026-07) wrote ``<weights>_no_wrapper`` and
    ``<weights>_<protocol>``. Their files were renamed when they moved into the results
    root (see its provenance note), but the rows inside keep the old names, so every
    reader of a migrated file maps them with this one rule. A protocol id has hyphens
    where the old names had underscores.
    """
    weights, sep, rest = name.partition("_")
    if weights not in WEIGHTS or not sep or not rest:
        raise ValueError(f"legacy condition {name!r}: expected <base|lora>_no_wrapper or <base|lora>_<protocol>")
    condition = weights if rest == "no_wrapper" else f"{weights}+{rest.replace('_', '-')}"
    parse_condition(condition)
    return condition


class _Record(BaseModel):
    """Base of every registry record: unknown keys are errors, values are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatTemplate(_Record):
    """Facts about a model's chat template that the pipeline must work around."""

    # False when the template raises on a system message. Prompt baselines then put
    # their instructions at the top of the first user turn.
    system_role: bool = True
    # Marker before the assistant turn, for completion-only loss masking, when the
    # template exposes no assistant header to detect (e.g. "[/INST]").
    response_template: Optional[str] = None
    # Extra tokenizer arguments the model card asks for.
    tokenizer_kwargs: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(_Record):
    id: RegistryId
    hf_id: HfRepo
    # The HF commit every run loads. null only when the repo could not be read
    # (gated or private); unpinned_reason then says why, and manifests record it.
    revision: Optional[GitSha]
    last_modified: Optional[IsoDate] = None
    unpinned_reason: Optional[str] = None
    # Pretraining lineage, e.g. gemma, llama3, qwen2, mistral. The grader-family
    # rule compares these values.
    family: RegistryId
    domain: Literal["general", "clinical"]
    params_b: PosFloat
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    # Precision the model is served in when no cell setting says otherwise.
    # QLoRA training is a cell setting (train.quantization), not this field.
    quantization: Optional[Literal["4bit", "8bit"]] = None
    chat_template: ChatTemplate = Field(default_factory=ChatTemplate)
    gated: bool = False
    license: str
    notes: str = ""

    @model_validator(mode="after")
    def _pin_or_reason(self) -> "ModelConfig":
        if self.revision is None and not self.unpinned_reason:
            raise ValueError("revision is null: give unpinned_reason")
        return self


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class HfSource(_Record):
    kind: Literal["hf"]
    dataset: HfRepo
    revision: GitSha
    split: str = "train"
    config_name: Optional[str] = None


class UrlFile(_Record):
    url: Annotated[str, StringConstraints(pattern=r"^https://")]
    # A URL can change under us; the hash pins what the runs read. null only when the
    # file was not read when the record was written; unpinned_reason then says why,
    # `fetch` prints the hash to pin, and each run's inputs manifest records the hash
    # it read.
    sha256: Optional[Sha256]
    unpinned_reason: Optional[str] = None

    @model_validator(mode="after")
    def _pin_or_reason(self) -> "UrlFile":
        if self.sha256 is None and not self.unpinned_reason:
            raise ValueError("sha256 is null: give unpinned_reason")
        return self


class UrlSource(_Record):
    kind: Literal["url"]
    # Name -> file. Split rules and loaders refer to files by these names.
    files: dict[RegistryId, UrlFile]


class LocalFile(_Record):
    # Relative to the repository root, so a fixture benchmark works in any checkout.
    path: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._/-]+$")]
    sha256: Sha256

    @model_validator(mode="after")
    def _relative(self) -> "LocalFile":
        if self.path.startswith("/") or ".." in self.path.split("/"):
            raise ValueError("path must be relative to the repository root and stay inside it")
        return self


class LocalSource(_Record):
    """Files in the repository, for test and smoke benchmarks."""

    kind: Literal["local"]
    files: dict[RegistryId, LocalFile]


Source = Annotated[Union[HfSource, UrlSource, LocalSource], Field(discriminator="kind")]


class Draw(_Record):
    """How a set of rows is chosen from a larger set."""

    # all: every row. head: the first `size` rows in source order. seeded: a
    # random.Random(seed) sample of `size` rows.
    select: Literal["all", "head", "seeded"]
    size: Optional[PosInt] = None
    seed: int = 0

    @model_validator(mode="after")
    def _size_matches_select(self) -> "Draw":
        if (self.select == "all") != (self.size is None):
            raise ValueError("select: all takes no size; head and seeded need one")
        return self


class LabelGroups(_Record):
    """The positive rows, the negative rows, or both, of one label."""

    label: RegistryId
    groups: Annotated[list[Literal["positive", "negative"]], Field(min_length=1)] = ["positive", "negative"]


class SplitRule(_Record):
    """Which rows a cell may train on and which rows it may evaluate on.

    named_files: the eval pool is every row of source file `eval_file`, and the train
        pool is every row of `train_file` that is not in the eval pool.
    eval_ids_only: the eval pool is the cell's fixed eval ids, and the train pool is
        every other row of `train_file` and `eval_file`. It records a run made before
        the eval file was held out of training; no new cell should use it.
    holdout: `pool` draws rows from the source, `holdout` draws the eval pool from
        those rows, and the rest of the pool is the train pool.
    With either method, the rows of the label groups listed in `hold_out` leave the
    train pool and form the probe pool, so a labelled probe set is never trained on.
    The probe pool does not join the eval pool: a cell evaluates on it only when its
    eval ids name label groups (EvalIds.only).
    """

    method: Literal["named_files", "eval_ids_only", "holdout"]
    eval_file: Optional[RegistryId] = None
    train_file: Optional[RegistryId] = None
    pool: Optional[Draw] = None
    holdout: Optional[Draw] = None
    hold_out: list[LabelGroups] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fields_match_method(self) -> "SplitRule":
        named = self.eval_file is not None and self.train_file is not None
        drawn = self.pool is not None and self.holdout is not None
        if self.method in ("named_files", "eval_ids_only") and not (
                named and self.pool is None and self.holdout is None):
            raise ValueError(f"method {self.method} needs eval_file and train_file, and no pool or holdout")
        if self.method == "holdout" and not (drawn and self.eval_file is None and self.train_file is None):
            raise ValueError("method holdout needs pool and holdout, and no eval_file or train_file")
        return self


class LabelRule(_Record):
    """A binary label on benchmark rows, made from facts the graders never see.

    Tag or rubric rule: a row is positive when it carries any tag in `positive_tags`,
    or when any of its rubric criteria matches `rubric_pattern`. It is negative when
    it carries any tag in `negative_tags`; with no negative tags, every row that is
    not positive is negative. Other rows are unlabelled and left out of the label's
    analyses.

    Field rule (paired probes): the loader sets `field` on each row, and its value
    decides the label. Rows that share the value of `pair_field` form one pair.
    """

    description: str
    # Who made the label, e.g. "benchmark physicians" or "authors".
    source: str
    # pre_specified: fixed before any result it is used on was seen. post_hoc:
    # defined with or after the results; the report marks it exploratory (D9).
    timing: Literal["pre_specified", "post_hoc"]
    positive_tags: list[str] = Field(default_factory=list)
    negative_tags: list[str] = Field(default_factory=list)
    rubric_pattern: Optional[str] = None
    field: Optional[str] = None
    positive_values: list[str] = Field(default_factory=list)
    negative_values: list[str] = Field(default_factory=list)
    pair_field: Optional[str] = None

    @model_validator(mode="after")
    def _one_rule(self) -> "LabelRule":
        tag_rule = bool(self.positive_tags or self.rubric_pattern)
        field_rule = self.field is not None
        if tag_rule == field_rule:
            raise ValueError("give either positive_tags/rubric_pattern or field, not both and not neither")
        if field_rule and not (self.positive_values and self.negative_values):
            raise ValueError("a field rule needs positive_values and negative_values")
        if not field_rule and (self.positive_values or self.negative_values or self.pair_field):
            raise ValueError("positive_values, negative_values and pair_field belong to a field rule")
        if self.rubric_pattern is not None:
            re.compile(self.rubric_pattern)
        return self


class BenchmarkConfig(_Record):
    id: RegistryId
    title: str
    # Key of the loader in the package's loader registry. Loaders are named by data
    # format (e.g. a rubric JSONL, or an HF question-answer table), not by dataset.
    loader: CodeKey
    source: Source
    license: str
    audience: Literal["patient", "clinician", "mixed"]
    task: Literal["open_ended", "single_best_answer"]
    # native: expert rubrics come with each row. synthesized: the loader builds the
    # shared multi-criterion rubric with `reference_criterion` as its ground-truth item.
    # none: rows have no rubric (no reference answer); the eval records no rubric score,
    # and no cell may train on the benchmark, because the filter grades by the rubric.
    rubric: Literal["native", "synthesized", "none"]
    # Loader field -> candidate source columns, first non-empty wins.
    fields: dict[str, list[str]] = Field(default_factory=dict)
    # Other loader settings (e.g. how many characters of the reference to keep). The
    # loader validates them.
    loader_params: dict[str, Any] = Field(default_factory=dict)
    # Prompt ids are sha1("<id_prefix>:<question>")[:12] behind this prefix. Keep the
    # prefix fixed, or the eval ids of earlier runs stop matching. null when the
    # source rows carry their own ids (HealthBench prompt_id).
    id_prefix: Optional[RegistryId] = None
    # The user turn, with {question} (and any other loader field) filled in.
    prompt_template: Optional[str] = None
    # Synthesized rubric: the ground-truth criterion, with {reference} filled in.
    reference_criterion: Optional[str] = None
    # Tags added to every row, e.g. "persona:clinician".
    tags: list[str] = Field(default_factory=list)
    splits: dict[RegistryId, SplitRule]
    labels: dict[RegistryId, LabelRule] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "BenchmarkConfig":
        if "default" not in self.splits:
            raise ValueError("splits must include 'default'")
        if self.rubric == "synthesized" and not self.reference_criterion:
            raise ValueError("rubric: synthesized needs reference_criterion")
        if self.rubric != "synthesized" and self.reference_criterion:
            raise ValueError(f"rubric: {self.rubric} takes no reference_criterion")
        for name, rule in self.splits.items():
            missing = [h.label for h in rule.hold_out if h.label not in self.labels]
            if missing:
                raise ValueError(f"split {name}: hold_out labels not defined in labels: {missing}")
            if isinstance(self.source, (UrlSource, LocalSource)) and rule.eval_file is not None:
                unknown = {rule.eval_file, rule.train_file} - set(self.source.files)
                if unknown:
                    raise ValueError(f"split {name}: files not in source.files: {sorted(unknown)}")
        return self


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class LeakRule(_Record):
    """When a response counts as leaking the protocol's internal format."""

    description: str
    # Case-insensitive regular expressions.
    markers: Annotated[list[str], Field(min_length=1)]
    # A response leaks when at least this many markers match.
    threshold: PosInt = 2

    @model_validator(mode="after")
    def _compiles(self) -> "LeakRule":
        for m in self.markers:
            re.compile(m)
        return self


class ProtocolConfig(_Record):
    id: RegistryId
    title: str
    # wrapper: one or more model calls that end in a reply (BODHI, ask-first).
    # prompt: one instruction block added to the conversation, one call.
    kind: Literal["wrapper", "prompt"]
    impl: CodeKey
    # Pinned package that implements it, when it is not this repo, e.g. bodhi-llm.
    package: Optional[str] = None
    passes: PosInt
    instructions: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    leak_rules: dict[RegistryId, LeakRule] = Field(default_factory=dict)
    # Citation or design note for where the protocol comes from.
    reference: str

    @model_validator(mode="after")
    def _prompt_shape(self) -> "ProtocolConfig":
        if self.kind == "prompt" and (not self.instructions or self.passes != 1):
            raise ValueError("kind prompt needs instructions and passes: 1")
        return self


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

class Grading(_Record):
    # Grades teacher traces for the filter.
    filter: RegistryId
    # Grades eval responses: benchmark rubric and the seven epistemic dimensions.
    evaluator: RegistryId
    # Extra graders over the stored eval responses (robustness panel, cross-family
    # re-grade). They never change the evaluator's scores.
    judges: list[RegistryId] = Field(default_factory=list)
    # Reason for a cell that breaks the family rule (R1 or R2). The cell is then marked in
    # every table, and a cell with outputs to judge needs a judge from a fourth family.
    same_family_waiver: Optional[str] = None


class Serving(_Record):
    """vLLM serving settings of one engine: the model under test or a grader."""

    max_model_len: PosInt
    # Scheduler slots on the server.
    max_num_seqs: PosInt
    # Client threads that send requests to it.
    concurrency: PosInt
    request_timeout_s: PosInt
    # Pause between two engines. Only the old TPU runtime needed one.
    drain_s: Annotated[float, Field(ge=0)] = 0.0


class Traces(_Record):
    max_prompts: PosInt
    decoding: Literal["greedy", "sample"] = "greedy"
    temperature: Annotated[float, Field(ge=0)] = 0.0
    top_p: Share = 1.0
    seed: int = 0
    max_new_tokens_per_pass: PosInt
    # The teacher's engine.
    serving: Serving

    @model_validator(mode="after")
    def _decoding(self) -> "Traces":
        if (self.decoding == "greedy") != (self.temperature == 0):
            raise ValueError("greedy decoding needs temperature 0; sample needs temperature > 0")
        return self


class Tau(_Record):
    """The filter threshold rule."""

    # fixed: keep traces scoring >= value. percentile: tau is that percentile of the
    # scores. fixed_then_percentile: fixed, then the percentile when the train split
    # holds less than min_kept_share of the raw traces (the rule the 2026-07 launcher
    # applied to non-HealthBench cells; no cell met it). none: keep every trace (the
    # unfiltered arm). random_same_count: a random sample of as many traces as the
    # fixed rule keeps (the control for the unfiltered arm).
    rule: Literal["fixed", "percentile", "fixed_then_percentile", "none", "random_same_count"]
    value: Optional[float] = None
    percentile: Optional[Annotated[float, Field(gt=0, lt=100)]] = None
    min_kept_share: Optional[Share] = None
    seed: int = 0

    @model_validator(mode="after")
    def _params(self) -> "Tau":
        need = {
            "fixed": {"value"},
            "percentile": {"percentile"},
            "fixed_then_percentile": {"value", "percentile", "min_kept_share"},
            "none": set(),
            "random_same_count": {"value"},
        }[self.rule]
        given = {k for k in ("value", "percentile", "min_kept_share") if getattr(self, k) is not None}
        if given != need:
            raise ValueError(f"tau rule {self.rule} takes exactly {sorted(need)}, got {sorted(given)}")
        return self


class Filter(_Record):
    tau: Tau
    val_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.1
    # once: one train/val split shared by every seed. per_seed: the seed shuffles it.
    split_policy: Literal["once", "per_seed"]
    split_seed: Optional[int] = None
    # The filter grader's engine.
    serving: Serving

    @model_validator(mode="after")
    def _split_seed(self) -> "Filter":
        if (self.split_policy == "once") != (self.split_seed is not None):
            raise ValueError("split_policy once needs split_seed; per_seed takes none")
        return self


class Train(_Record):
    # peft: this package trains it. external: it was trained elsewhere and the block
    # only records the recipe (`record` says where it is documented); run refuses it.
    engine: Literal["peft", "external"]
    record: Optional[str] = None
    method: Literal["lora", "dora", "rslora"] = "lora"
    quantization: Optional[Literal["4bit", "8bit"]] = None
    r: PosInt
    alpha: PosFloat
    dropout: Annotated[float, Field(ge=0, lt=1)]
    target_modules: Annotated[list[str], Field(min_length=1)]
    epochs: Optional[PosFloat] = None
    max_steps: Optional[PosInt] = None
    per_device_batch: PosInt
    grad_accum: PosInt
    effective_batch: PosInt
    lr: PosFloat
    scheduler: Literal["cosine", "linear", "constant"]
    warmup_ratio: Share = 0.0
    max_seq_length: PosInt
    gradient_checkpointing: bool = False
    # Which checkpoint becomes the seed's adapter.
    checkpoint: Literal["best_val_loss", "last"]
    # When validation loss is measured and a checkpoint saved: each epoch, or every
    # eval_steps optimizer steps.
    eval_every: Literal["epoch", "steps"] = "epoch"
    eval_steps: Optional[PosInt] = None
    logging_steps: PosInt = 10
    hardware: str

    @model_validator(mode="after")
    def _consistent(self) -> "Train":
        if (self.epochs is None) == (self.max_steps is None):
            raise ValueError("give exactly one of epochs and max_steps")
        if self.engine == "external" and not self.record:
            raise ValueError("engine external needs record")
        if self.engine == "peft" and self.per_device_batch * self.grad_accum != self.effective_batch:
            raise ValueError("effective_batch must equal per_device_batch * grad_accum on one GPU")
        if self.method == "dora" and self.quantization:
            raise ValueError("DoRA does not work on a quantized base")
        if (self.eval_every == "steps") != (self.eval_steps is not None):
            raise ValueError("eval_every steps needs eval_steps; epoch takes none")
        return self


class EvalIds(_Record):
    # per_seed_draw: random.Random(seed) draws n ids from the eval pool for each seed.
    # fixed_pool: every seed uses the whole eval pool (n must equal its size).
    # fixed_file: every seed uses the ids in `file`, a JSON list in the cell's
    # shared/inputs folder (so no results path appears in a config).
    policy: Literal["per_seed_draw", "fixed_pool", "fixed_file"]
    n: PosInt
    # row: n counts rows. pair: n counts pairs of a paired benchmark (rows that share a
    # pair id), and both rows of each drawn pair are evaluated, so every seed keeps
    # whole pairs.
    unit: Literal["row", "pair"] = "row"
    # When set, draw from the rows of the eval pool and the probe pool that are in
    # these label groups, instead of from the eval pool (a labelled probe set).
    only: list[LabelGroups] = Field(default_factory=list)
    file: Optional[Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]+\.json$")]] = None

    @model_validator(mode="after")
    def _file(self) -> "EvalIds":
        if (self.policy == "fixed_file") != (self.file is not None):
            raise ValueError("policy fixed_file needs file; other policies take none")
        if self.unit == "pair" and self.policy == "fixed_file":
            raise ValueError("unit pair draws pairs; a fixed_file list takes unit row")
        return self


class Eval(_Record):
    conditions: Annotated[list[str], Field(min_length=1)]
    ids: EvalIds
    # train: serve the base the way it was trained (4-bit when train.quantization is
    # 4bit). model: serve it at the model record's precision.
    base_precision: Literal["train", "model"]
    base_precision_reason: Optional[str] = None
    max_new_tokens: PosInt
    max_new_tokens_per_pass: PosInt
    serving: Serving

    @model_validator(mode="after")
    def _conditions(self) -> "Eval":
        for c in self.conditions:
            parse_condition(c)
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions repeat")
        return self


class Probe(_Record):
    """Contamination probe: does the base complete benchmark prompts verbatim more often
    than control text it should not have memorised?"""

    # The benchmark whose rows give the control text.
    control: RegistryId
    n_prompts: PosInt = 100
    prefix_tokens: PosInt = 30
    continuation_tokens: PosInt = 50
    seed: int = 0
    serving: Serving


class Bench(_Record):
    """Measured time and tokens per condition on a fixed batch of eval prompts."""

    n_prompts: PosInt = 50
    max_new_tokens: PosInt = 512
    # Skip CUDA-graph capture, as the eval does, so the conditions compare like for like.
    enforce_eager: bool = True
    serving: Serving


# Statuses of a cell that records a past run. `run` refuses them: a new run of the same
# design is a new cell with its own tag, so outputs never mix.
RECORD_STATUSES = ("complete", "partial", "not_run", "pilot")


class Report(_Record):
    label: str
    # Letters only; it names the cell in LaTeX macros.
    token: MacroToken
    status: Literal["planned", "complete", "partial", "not_run", "pilot"]
    # Why the cell stays out of the main statements, if it does.
    set_aside: Optional[str] = None
    notes: str = ""


class CellConfig(_Record):
    id: CellId
    benchmark: RegistryId
    split: RegistryId = "default"
    model: RegistryId
    protocol: RegistryId
    seeds: Annotated[list[int], Field(min_length=1)]
    # Eval-only cell: reuse the adapters of another cell (same model and protocol).
    adapters_from: Optional[CellId] = None
    # Train on another cell's teacher traces (same benchmark, split and protocol).
    traces_from: Optional[CellId] = None
    grading: Grading
    traces: Optional[Traces] = None
    filter: Optional[Filter] = None
    train: Optional[Train] = None
    eval: Eval
    probe: Optional[Probe] = None
    bench: Optional[Bench] = None
    report: Report

    @model_validator(mode="after")
    def _consistent(self) -> "CellConfig":
        parts = self.id.split("__")
        if parts[:3] != [self.benchmark, self.model, self.protocol]:
            raise ValueError(f"id {self.id} must start with {self.benchmark}__{self.model}__{self.protocol}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds repeat")
        uses_lora = any(parse_condition(c)[0] == "lora" for c in self.eval.conditions)
        if self.adapters_from:
            if self.traces or self.filter or self.train or self.traces_from:
                raise ValueError("an adapters_from cell takes no traces, traces_from, filter or train")
        elif uses_lora and self.report.status not in ("not_run", "pilot"):
            # A run that stopped early, or a pilot whose settings were not kept, may leave
            # out the blocks it never ran or never recorded.
            if self.train is None or self.filter is None:
                raise ValueError("lora conditions need train and filter")
            if (self.traces is None) == (self.traces_from is None):
                raise ValueError("give exactly one of traces and traces_from")
        if (self.train is not None and self.train.quantization and self.eval.base_precision == "model"
                and not self.eval.base_precision_reason):
            # The adapter was trained on a quantized base but is served on a
            # full-precision one (audit REPRO-12). Allowed only with a stated reason.
            raise ValueError("train.quantization with eval.base_precision model needs base_precision_reason")
        return self


# ---------------------------------------------------------------------------
# Project settings (configs/project.yaml)
# ---------------------------------------------------------------------------

# A path inside the repository, written relative to its root.
RepoPath = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.][A-Za-z0-9_./-]*$")]


class Interpreters(_Record):
    """The Python of each environment.

    Training needs a newer transformers than the vLLM release that serves the models
    allows, so training and inference run in two environments (requirements.txt and
    requirements-infer.txt). CPU dry runs use the current interpreter instead.
    """

    train: RepoPath
    infer: RepoPath


class ProjectConfig(_Record):
    results_root: RepoPath
    data_root: RepoPath
    # Stub and dry runs write here, never under results_root: their stage manifests say
    # complete, and a GPU run under results_root would skip those stages.
    scratch_root: RepoPath = "outputs"
    python: Interpreters
    # Each GPU stage serves on port vllm_port_base + its GPU index, so stages on
    # different GPUs never collide.
    vllm_port_base: PosInt = 8000


# ---------------------------------------------------------------------------
# The registry and the checks across files
# ---------------------------------------------------------------------------

class RegistryError(ValueError):
    """One or more registry files are invalid. The message lists every problem."""


@dataclass(frozen=True)
class Registry:
    models: dict[str, ModelConfig] = field(default_factory=dict)
    benchmarks: dict[str, BenchmarkConfig] = field(default_factory=dict)
    protocols: dict[str, ProtocolConfig] = field(default_factory=dict)
    cells: dict[str, CellConfig] = field(default_factory=dict)

    def teacher(self, cell: CellConfig) -> ModelConfig:
        """The model whose traces the cell trains on."""
        source = self.cells[cell.traces_from] if cell.traces_from else cell
        return self.models[source.model]


def family_problems(cell: CellConfig, reg: Registry) -> list[str]:
    """The grader-family rule for one cell (docs/CONVENTIONS.md, convention 5).

    R1  the filter and the evaluator come from different families;
    R2  neither the base model nor the teacher shares a family with the filter or the
        evaluator;
        grading.same_family_waiver can waive R1 and R2 with a reason, and then
    R2b a planned, complete or partial cell lists a judge from a family outside all four
        (a pilot or a run that stopped early enters no table, so it needs none);
    R3  no judge shares a family with the base model, the teacher or the filter, and
        no judge is the evaluator itself.
    Assumes every referenced model exists; validate_registry checks that first.
    """
    fam = {k: m.family for k, m in reg.models.items()}
    g = cell.grading
    base, teacher = fam[cell.model], reg.teacher(cell).family
    filt, ev = fam[g.filter], fam[g.evaluator]
    out = []
    if filt == ev and not g.same_family_waiver:
        out.append(f"R1: filter {g.filter} and evaluator {g.evaluator} share family {filt}; "
                   "add grading.same_family_waiver with a reason, or change the cell")
    shared = sorted({base, teacher} & {filt, ev})
    if shared and not g.same_family_waiver:
        out.append(f"R2: base or teacher shares family {shared} with a grader; "
                   "add grading.same_family_waiver with a reason, or change the cell")
    has_outputs = cell.report.status in ("planned", "complete", "partial")
    if (shared or filt == ev) and g.same_family_waiver and has_outputs:
        outside = {base, teacher, filt, ev}
        if not any(fam[j] not in outside for j in g.judges):
            out.append("R2b: same_family_waiver needs a judge from a family outside "
                       f"{sorted(outside)}")
    for j in g.judges:
        if fam[j] in {base, teacher, filt}:
            out.append(f"R3: judge {j} shares family {fam[j]} with the base, the teacher or the filter")
        if j == g.evaluator:
            out.append(f"R3: judge {j} is the evaluator")
    return out


def validate_registry(reg: Registry) -> None:
    """Check references between records and the family rule; raise RegistryError."""
    problems: list[str] = []
    for cid, cell in sorted(reg.cells.items()):
        def bad(msg: str) -> None:
            problems.append(f"cells/{cid}: {msg}")

        refs = [("benchmark", cell.benchmark, reg.benchmarks), ("model", cell.model, reg.models),
                ("protocol", cell.protocol, reg.protocols), ("grading.filter", cell.grading.filter, reg.models),
                ("grading.evaluator", cell.grading.evaluator, reg.models)]
        refs += [("grading.judges", j, reg.models) for j in cell.grading.judges]
        refs += [(f"eval.conditions {c}", p, reg.protocols)
                 for c in cell.eval.conditions for p in [parse_condition(c)[1]] if p]
        refs += [(name, other, reg.cells) for name, other in
                 (("adapters_from", cell.adapters_from), ("traces_from", cell.traces_from)) if other]
        if cell.probe:
            refs.append(("probe.control", cell.probe.control, reg.benchmarks))
        missing = [f"{what} {ref!r} has no record" for what, ref, table in refs if ref not in table]
        for m in missing:
            bad(m)
        if missing:
            continue
        bench = reg.benchmarks[cell.benchmark]
        if cell.split not in bench.splits:
            bad(f"split {cell.split!r} is not a split of benchmark {cell.benchmark}")
        elif bench.splits[cell.split].method == "eval_ids_only" and cell.eval.ids.policy != "fixed_file":
            bad(f"split {cell.split} takes its eval pool from the eval ids, so eval.ids.policy must be fixed_file")
        if bench.rubric == "none" and (cell.traces is not None or cell.train is not None):
            bad(f"benchmark {cell.benchmark} has no rubric, so the filter cannot grade traces; "
                "evaluate on it with an adapters_from cell")
        for g in cell.eval.ids.only:
            if g.label not in bench.labels:
                bad(f"eval.ids.only label {g.label!r} is not a label of benchmark {cell.benchmark}")
        if cell.adapters_from:
            src = reg.cells[cell.adapters_from]
            if (src.model, src.protocol) != (cell.model, cell.protocol) or src.train is None:
                bad(f"adapters_from {src.id} must train the same model and protocol")
            elif src.grading.filter != cell.grading.filter:
                bad(f"grading.filter must match the filter of {src.id}")
        if cell.traces_from:
            src = reg.cells[cell.traces_from]
            if (src.benchmark, src.split, src.protocol) != (cell.benchmark, cell.split, cell.protocol):
                bad(f"traces_from {src.id} must have the same benchmark, split and protocol")
            elif src.traces is None:
                bad(f"traces_from {src.id} must generate its own traces")
        problems += [f"cells/{cid}: {p}" for p in family_problems(cell, reg)]
    if problems:
        raise RegistryError("registry check failed:\n  " + "\n  ".join(problems))


_KINDS = (("models", ModelConfig), ("benchmarks", BenchmarkConfig),
          ("protocols", ProtocolConfig), ("cells", CellConfig))


def load_registry(configs_dir: Path) -> Registry:
    """Read and check every record under configs_dir/{models,benchmarks,protocols,cells}.

    A file's name must be its id plus ".yaml". Every problem in every file is
    collected and raised together, so one run shows the whole list.
    """
    configs_dir = Path(configs_dir)
    tables: dict[str, dict] = {}
    problems: list[str] = []
    for kind, cls in _KINDS:
        tables[kind] = {}
        for path in sorted((configs_dir / kind).glob("*.yaml")):
            where = f"{kind}/{path.name}"
            try:
                rec = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            except (ValidationError, yaml.YAMLError) as e:
                problems.append(f"{where}: {e}")
                continue
            if rec.id != path.stem:
                problems.append(f"{where}: id {rec.id!r} does not match the file name")
            tables[kind][rec.id] = rec
    if problems:
        raise RegistryError("registry files failed to load:\n  " + "\n  ".join(problems))
    reg = Registry(**tables)
    validate_registry(reg)
    return reg
