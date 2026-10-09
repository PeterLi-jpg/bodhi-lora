"""The registry schema (bodhi_lora/registry/schema.py): what it accepts and what it refuses.

The records below follow the examples in docs/ARCHITECTURE.md. They are built in
memory so the test does not depend on the configs/ files the build lanes write.
"""
import copy

import pytest
import yaml

pydantic = pytest.importorskip("pydantic")

from bodhi_lora.registry import (  # noqa: E402
    BenchmarkConfig,
    CellConfig,
    ModelConfig,
    ProtocolConfig,
    Registry,
    RegistryError,
    family_problems,
    load_registry,
    parse_condition,
    validate_registry,
)

SHA = "0" * 40


def model(id_, family, domain="general"):
    return {"id": id_, "hf_id": f"org/{id_}", "revision": SHA, "family": family,
            "domain": domain, "params_b": 8, "license": "test"}


MODELS = [model("base-24b", "mistral"), model("filter-14b", "qwen2"), model("eval-8b", "llama3"),
          model("clinical-8b", "llama3", "clinical"), model("judge-27b", "gemma", "clinical")]

BENCHMARK = {
    "id": "rubric-bench", "title": "A rubric benchmark", "loader": "rubric_jsonl",
    "source": {"kind": "url", "files": {
        "full": {"url": "https://example.org/full.jsonl", "sha256": "a" * 64},
        "hard": {"url": "https://example.org/hard.jsonl", "sha256": "b" * 64}}},
    "license": "MIT", "audience": "patient", "task": "open_ended", "rubric": "native",
    "splits": {
        "default": {"method": "named_files", "eval_file": "hard", "train_file": "full",
                    "hold_out": [{"label": "needs-info", "groups": ["positive"]}]},
        "hard": {"method": "named_files", "eval_file": "hard", "train_file": "full"}},
    "labels": {"needs-info": {
        "description": "physicians judged that the prompt lacks context",
        "source": "benchmark physicians", "timing": "pre_specified",
        "positive_tags": ["category:not-enough-context"],
        "negative_tags": ["category:enough-context"]}},
}

PROTOCOLS = [
    {"id": "bodhi", "title": "Two-pass analysis then reply", "kind": "wrapper", "impl": "bodhi_llm",
     "package": "bodhi-llm==0.1.4", "passes": 2, "reference": "test",
     "leak_rules": {"headers": {"description": "analysis headers in the reply",
                                "markers": ["RED FLAGS", "KEY UNCERTAINTIES"], "threshold": 2}}},
    {"id": "clarify", "title": "Clarifying-question prompt", "kind": "prompt", "impl": "instruction",
     "passes": 1, "instructions": "Ask for missing facts first.", "reference": "test"},
]

SERVING = {"max_model_len": 4096, "max_num_seqs": 32, "concurrency": 32, "request_timeout_s": 900}
GRADER_SERVING = {"max_model_len": 8192, "max_num_seqs": 64, "concurrency": 16, "request_timeout_s": 180}

CELL = {
    "id": "rubric-bench__base-24b__bodhi",
    "benchmark": "rubric-bench", "model": "base-24b", "protocol": "bodhi",
    "seeds": [42, 7, 13, 99, 101],
    "grading": {"filter": "filter-14b", "evaluator": "eval-8b", "judges": ["judge-27b"]},
    "traces": {"max_prompts": 4200, "max_new_tokens_per_pass": 2048, "serving": GRADER_SERVING},
    "filter": {"tau": {"rule": "fixed", "value": 0.4}, "split_policy": "per_seed",
               "serving": GRADER_SERVING},
    "train": {"engine": "peft", "quantization": "4bit", "r": 16, "alpha": 32, "dropout": 0.05,
              "target_modules": ["q_proj", "k_proj", "v_proj"], "epochs": 3, "per_device_batch": 2,
              "grad_accum": 8, "effective_batch": 16, "lr": 1.0e-4, "scheduler": "cosine",
              "warmup_ratio": 0.05, "max_seq_length": 4096, "checkpoint": "best_val_loss",
              "hardware": "1x H100 80GB"},
    "eval": {"conditions": ["base", "base+bodhi", "lora", "lora+bodhi", "base+clarify"],
             "ids": {"policy": "per_seed_draw", "n": 200}, "base_precision": "train",
             "max_new_tokens": 1024, "max_new_tokens_per_pass": 2048, "serving": SERVING},
    "report": {"label": "Rubric bench, base 24B", "token": "RubricBase", "status": "planned"},
}


def registry(cells=(CELL,), models=MODELS):
    return Registry(
        models={m["id"]: ModelConfig.model_validate(m) for m in models},
        benchmarks={BENCHMARK["id"]: BenchmarkConfig.model_validate(BENCHMARK)},
        protocols={p["id"]: ProtocolConfig.model_validate(p) for p in PROTOCOLS},
        cells={c["id"]: CellConfig.model_validate(c) for c in cells},
    )


def edited(base, **changes):
    """A deep copy of base with dotted keys replaced, e.g. edited(CELL, **{"grading.judges": []})."""
    out = copy.deepcopy(base)
    for dotted, value in changes.items():
        node = out
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        node[last] = value
    return out


def test_valid_registry_passes():
    validate_registry(registry())


def test_parse_condition():
    assert parse_condition("lora") == ("lora", None)
    assert parse_condition("base+ask-first") == ("base", "ask-first")
    with pytest.raises(ValueError):
        parse_condition("lora_bodhi")


@pytest.mark.parametrize("changes, message", [
    ({"traces.extra_key": 1}, "Extra inputs are not permitted"),
    ({"id": "rubric-bench__base-24b__clarify"}, "must start with"),
    ({"eval.conditions": ["lora_no_wrapper"]}, "expected base, lora"),
    ({"train": None}, "lora conditions need train"),
    ({"eval.base_precision": "model"}, "base_precision_reason"),
    ({"filter.tau": {"rule": "fixed"}}, "takes exactly"),
    ({"train.grad_accum": 4}, "effective_batch"),
    ({"eval.ids": {"policy": "fixed_file", "n": 200}}, "needs file"),
    ({"train.eval_every": "steps"}, "needs eval_steps"),
    ({"filter.serving": None}, "filter.serving"),
])
def test_cell_record_refuses(changes, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        CellConfig.model_validate(edited(CELL, **changes))


def test_family_rule_same_family_base_needs_waiver():
    # A Llama-family base graded by a Llama-family evaluator: R2 unless waived.
    models = MODELS + [model("llama-base-8b", "llama3", "clinical")]
    cell = edited(CELL, id="rubric-bench__llama-base-8b__bodhi", model="llama-base-8b",
                  **{"grading.judges": []})
    with pytest.raises(RegistryError, match="R2"):
        validate_registry(registry([cell], models))
    waived = edited(cell, **{"grading.same_family_waiver": "kept as run; regraded"})
    with pytest.raises(RegistryError, match="R2b"):
        validate_registry(registry([waived], models))
    regraded = edited(waived, **{"grading.judges": ["judge-27b"]})
    validate_registry(registry([regraded], models))


def test_family_rule_waiver_covers_r1_and_records_without_outputs():
    # A filter and evaluator from one family break R1. A waiver records the reason; a
    # pilot needs no judge because it enters no table, but a complete cell does.
    models = MODELS + [model("qwen-eval-14b", "qwen2")]
    cell = edited(CELL, **{"grading.evaluator": "qwen-eval-14b", "grading.judges": []})
    with pytest.raises(RegistryError, match="R1"):
        validate_registry(registry([cell], models))
    waived = edited(cell, **{"grading.same_family_waiver": "pilot before the design"})
    with pytest.raises(RegistryError, match="R2b"):
        validate_registry(registry([edited(waived, **{"report.status": "complete"})], models))
    validate_registry(registry([edited(waived, **{"report.status": "pilot"})], models))


def test_records_that_stopped_early_may_omit_stage_blocks():
    stopped = edited(CELL, train=None, filter=None, **{"report.status": "not_run"})
    validate_registry(registry([stopped]))
    with pytest.raises(pydantic.ValidationError, match="lora conditions need train"):
        CellConfig.model_validate(edited(stopped, **{"report.status": "complete"}))


def test_eval_ids_only_split_needs_fixed_eval_ids():
    bench = edited(BENCHMARK, **{"splits.pilot": {"method": "eval_ids_only", "eval_file": "hard",
                                                  "train_file": "full"}})
    cell = edited(CELL, split="pilot")
    reg = Registry(models=registry().models, protocols=registry().protocols,
                   benchmarks={"rubric-bench": BenchmarkConfig.model_validate(bench)},
                   cells={cell["id"]: CellConfig.model_validate(cell)})
    with pytest.raises(RegistryError, match="fixed_file"):
        validate_registry(reg)


def test_local_source_paths_stay_in_the_repo():
    local = {"kind": "local", "files": {"rows": {"path": "tests/fixtures/x.jsonl", "sha256": "c" * 64}}}
    bench = edited(BENCHMARK, source=local, splits={"default": {
        "method": "holdout", "pool": {"select": "all"}, "holdout": {"select": "head", "size": 2}}})
    BenchmarkConfig.model_validate(bench)
    for bad in ("/etc/passwd", "../outside.jsonl"):
        with pytest.raises(pydantic.ValidationError):
            BenchmarkConfig.model_validate(edited(bench, **{"source.files.rows.path": bad}))


def test_family_rule_judges():
    in_filter_family = MODELS + [model("qwen-judge", "qwen2", "clinical")]
    cell = edited(CELL, **{"grading.judges": ["qwen-judge"]})
    with pytest.raises(RegistryError, match="R3"):
        validate_registry(registry([cell], in_filter_family))
    # A clinical judge from the evaluator's family is allowed: the panel compares it
    # with the general evaluator of the same family.
    reg = registry([edited(CELL, **{"grading.judges": ["clinical-8b"]})])
    assert family_problems(reg.cells[CELL["id"]], reg) == []


def test_family_rule_teacher_counts():
    # A student trained on traces from a teacher in the evaluator's family breaks R2.
    models = MODELS + [model("llama-teacher-8b", "llama3")]
    teacher = edited(CELL, id="rubric-bench__llama-teacher-8b__bodhi", model="llama-teacher-8b",
                     **{"grading.same_family_waiver": "teacher only", "grading.judges": ["judge-27b"]})
    student = edited(CELL, id="rubric-bench__base-24b__bodhi__teacher-llama-teacher-8b",
                     traces=None, traces_from=teacher["id"])
    with pytest.raises(RegistryError, match="R2"):
        validate_registry(registry([teacher, student], models))


def test_cross_references():
    eval_only = {k: v for k, v in CELL.items() if k not in ("traces", "filter", "train")}
    eval_only = edited(eval_only, id="rubric-bench__base-24b__bodhi__probe", adapters_from=CELL["id"])
    validate_registry(registry([CELL, eval_only]))
    dangling = edited(eval_only, adapters_from="rubric-bench__base-24b__bodhi__gone")
    with pytest.raises(RegistryError, match="has no record"):
        validate_registry(registry([CELL, dangling]))
    bad_split = edited(CELL, split="missing")
    with pytest.raises(RegistryError, match="is not a split"):
        validate_registry(registry([bad_split]))
    probe = edited(eval_only, **{"eval.ids": {"policy": "fixed_pool", "n": 50,
                                              "only": [{"label": "needs-info"}]}})
    validate_registry(registry([CELL, probe]))
    bad_label = edited(probe, **{"eval.ids.only": [{"label": "no-such-label"}]})
    with pytest.raises(RegistryError, match="is not a label"):
        validate_registry(registry([CELL, bad_label]))


def test_benchmark_refuses_unknown_hold_out_label():
    bench = edited(BENCHMARK, **{"splits.default.hold_out": [{"label": "not-defined"}]})
    with pytest.raises(pydantic.ValidationError, match="hold_out labels not defined"):
        BenchmarkConfig.model_validate(bench)


def test_load_registry_from_files(tmp_path):
    def write(kind, rec, name=None):
        d = tmp_path / kind
        d.mkdir(exist_ok=True)
        (d / f"{name or rec['id']}.yaml").write_text(yaml.safe_dump(rec))

    for m in MODELS:
        write("models", m)
    write("benchmarks", BENCHMARK)
    for p in PROTOCOLS:
        write("protocols", p)
    write("cells", CELL)
    reg = load_registry(tmp_path)
    assert set(reg.cells) == {CELL["id"]}
    write("models", MODELS[0], name="renamed")
    with pytest.raises(RegistryError, match="does not match the file name"):
        load_registry(tmp_path)


def test_an_unpinned_url_file_needs_a_reason():
    unpinned = copy.deepcopy(BENCHMARK)
    unpinned["source"]["files"]["hard"] = {"url": "https://example.org/hard.jsonl", "sha256": None}
    with pytest.raises(pydantic.ValidationError, match="unpinned_reason"):
        BenchmarkConfig.model_validate(unpinned)
    unpinned["source"]["files"]["hard"]["unpinned_reason"] = "not read yet"
    BenchmarkConfig.model_validate(unpinned)


def test_a_benchmark_without_a_rubric_takes_no_reference_and_no_training():
    bench = dict(copy.deepcopy(BENCHMARK), rubric="none")
    BenchmarkConfig.model_validate(bench)
    with pytest.raises(pydantic.ValidationError, match="takes no reference_criterion"):
        BenchmarkConfig.model_validate(dict(bench, reference_criterion="matches {reference}"))
    reg = Registry(models={m["id"]: ModelConfig.model_validate(m) for m in MODELS},
                   benchmarks={"rubric-bench": BenchmarkConfig.model_validate(bench)},
                   protocols={p["id"]: ProtocolConfig.model_validate(p) for p in PROTOCOLS},
                   cells={CELL["id"]: CellConfig.model_validate(CELL)})
    with pytest.raises(RegistryError, match="has no rubric"):
        validate_registry(reg)


def test_pair_draws_do_not_take_a_fixed_file():
    bad = copy.deepcopy(CELL)
    bad["eval"]["ids"] = {"policy": "fixed_file", "n": 200, "file": "eval_ids.json", "unit": "pair"}
    with pytest.raises(pydantic.ValidationError, match="unit pair"):
        CellConfig.model_validate(bad)
