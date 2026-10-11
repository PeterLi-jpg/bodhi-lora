"""Stage probe: does the base model continue benchmark text verbatim more often than
control text (issue #72, EXPERIMENTS.md E11)?

    python -m bodhi_lora.pipeline.probe --cell <cell id> [--stub-engine]

For each probed text the base model gets the first prefix_tokens tokens as a raw prompt
and continues it greedily for continuation_tokens tokens. The prompt is token ids on the
completions route, with no chat template: a chat template would make the prefix a user
turn, and an instruction-tuned model would answer it instead of continuing it. A text
counts as memorised when the continuation equals the text's next continuation_tokens
tokens after lowercasing and collapsing spaces. The probe also records the longest
common token prefix. The served tokenizer splits the text, so the token counts are the
model's own.

The probed text is the source text of a row: its user turns, with the fixed words of
the benchmark's prompt_template removed, so a templated benchmark is not probed on the
template. Rows are taken in a seeded shuffle, and only rows with at least
prefix_tokens + continuation_tokens tokens count, so every drawn row is probed:
n_prompts from the cell's eval pool and n_prompts from every row of the control
benchmark (probe.control). The two exact-match rates are compared with a two-sided
Fisher exact test (scipy.stats.fisher_exact). A set with fewer long-enough rows than
n_prompts stops the stage: the cell record then needs a smaller n_prompts or another
control.

Output, in shared/probe/: probe.json, the rates, counts and test, which git tracks; and
samples.jsonl, one line per probed text with its prefix, its true continuation and the
model's, which git ignores like every JSONL under results/, because it copies benchmark
text (open decision O8). The stage only adds that folder, so it also runs on a recorded
cell.
"""
from __future__ import annotations

import json
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import fisher_exact

from bodhi_lora import manifest as mf
from bodhi_lora.data import load_rows
from bodhi_lora.engine import make_engine
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser

INTERPRETATION = ("p < 0.05 with a benchmark rate well above the control rate is evidence that "
                  "the benchmark text was in the base model's pretraining data. Equal rates do "
                  "not rule it out: the probe sees only verbatim recall of the drawn rows.")


# -- the text of a row -----------------------------------------------------------------

def normalize_for_exact_match(text: str) -> str:
    """Lowercase + collapse whitespace. Used for the exact_match metric."""
    return " ".join(text.lower().split())


def compute_prefix_overlap_tokens(toks_a, toks_b) -> int:
    """Length of the longest common prefix of two token-id sequences."""
    n = 0
    for a, b in zip(toks_a, toks_b):
        if a != b:
            break
        n += 1
    return n


def extract_user_prompt_text(prompt_messages) -> str:
    """The user turns of a prompt, joined by newlines, so a multi-turn prompt
    flattens to one string for tokenization."""
    return "\n".join(
        m["content"] for m in prompt_messages if m.get("role") == "user"
    )


def strip_template(text: str, template) -> str:
    """The text without the template's fixed words before its first field and after its
    last field. A text that does not start or end with them is returned as it is."""
    parts = list(string.Formatter().parse(template or ""))
    if not any(field is not None for _, field, _, _ in parts):
        return text
    # parse() yields (literal, field, spec, conversion); a trailing literal has no field.
    head = parts[0][0]
    tail = parts[-1][0] if parts[-1][1] is None else ""
    if head and text.startswith(head):
        text = text[len(head):]
    if tail and text.endswith(tail):
        text = text[:-len(tail)]
    return text


def source_text(row, bench) -> str:
    return strip_template(extract_user_prompt_text(row.messages), bench.prompt_template)


# -- drawing and probing ------------------------------------------------------------------

def draw_long_enough(items: list, engine, need: int, n: int, seed: int, name: str) -> list:
    """n (id, text, token ids) of at least `need` tokens, in a seeded shuffle of `items`,
    and how many rows were tokenized to find them.

    items are (id, text) pairs, sorted by id first so the draw does not depend on the
    source order. Tokenizing stops once n are found.
    """
    order = sorted(items)
    random.Random(seed).shuffle(order)
    out, seen = [], 0
    for pid, text in order:
        seen += 1
        ids = engine.tokenize(text)
        if len(ids) >= need:
            out.append((pid, text, ids))
            if len(out) == n:
                return out, seen
    raise SystemExit(f"the {name} set has {len(out)} rows of at least {need} tokens among "
                     f"{seen}; the probe needs {n}. Lower probe.n_prompts or pick another control.")


def probe_one(pid, ids: list, engine, prefix_tokens: int, continuation_tokens: int) -> dict:
    prefix, truth = ids[:prefix_tokens], ids[prefix_tokens:prefix_tokens + continuation_tokens]
    done = engine.continue_ids(prefix, max_new_tokens=continuation_tokens, temperature=0.0)
    truth_text = engine.detokenize(truth)
    return {
        "prompt_id_or_index": pid,
        "leak_prefix": engine.detokenize(prefix),
        "ground_truth_continuation": truth_text,
        "model_output": done.text,
        "exact_match": normalize_for_exact_match(done.text) == normalize_for_exact_match(truth_text),
        "prefix_overlap_tokens": compute_prefix_overlap_tokens(engine.tokenize(done.text), truth),
    }


def aggregate(samples: list, n_rows: int, n_tokenized: int) -> dict:
    """The summary of one probed set. It holds no text, so git can track it."""
    n = len(samples)
    hits = sum(s["exact_match"] for s in samples)
    return {
        "n": n,
        "n_rows_in_set": n_rows,
        # Rows read in the seeded order until n were long enough; the rest were too short.
        "n_rows_tokenized": n_tokenized,
        "n_exact_match": hits,
        "exact_match_rate": hits / n if n else None,
        "mean_prefix_overlap_tokens": float(np.mean([s["prefix_overlap_tokens"] for s in samples])) if n else None,
    }


def probe_set(drawn: list, engine, probe) -> list:
    with ThreadPoolExecutor(max_workers=probe.serving.concurrency) as pool:
        return list(pool.map(lambda d: probe_one(d[0], d[2], engine, probe.prefix_tokens,
                                                 probe.continuation_tokens), drawn))


def run(ctx: StageContext) -> dict:
    probe = ctx.cell.probe
    if probe is None:
        raise SystemExit(f"{ctx.cell.id} has no probe block")
    out_dir = ctx.paths.shared("probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    rows = ctx.rows_by_id()
    control_bench = ctx.reg.benchmarks[probe.control]
    control_rows = load_rows(control_bench, ctx.project.data_root, fetch=False)
    sets = {
        "benchmark": [(pid, source_text(rows[pid], ctx.bench)) for pid in ctx.split().eval],
        "control": [(r.prompt_id, source_text(r, control_bench)) for r in control_rows],
    }
    need = probe.prefix_tokens + probe.continuation_tokens
    # The stub's tokenizer is its own; the reply repeats nothing, so stub rates are 0.
    engine = make_engine(ctx.model, serving=probe.serving, stub=ctx.stub, stub_reply="stub continuation",
                         log_path=ctx.log_path("vllm_probe.log"))
    results, samples = {}, {}
    with engine as eng:
        for name, items in sets.items():
            drawn, tokenized = draw_long_enough(items, eng, need, probe.n_prompts, probe.seed, name)
            samples[name] = probe_set(drawn, eng, probe)
            results[name] = aggregate(samples[name], len(items), tokenized)
            print(f"  {name}: {results[name]['n']} probed, exact-match rate "
                  f"{results[name]['exact_match_rate']}", flush=True)
    hits = {k: v["n_exact_match"] for k, v in results.items()}
    _, p_value = fisher_exact([[hits["benchmark"], probe.n_prompts - hits["benchmark"]],
                               [hits["control"], probe.n_prompts - hits["control"]]],
                              alternative="two-sided")
    out = {"benchmark_id": ctx.bench.id, "control_id": probe.control, "model": ctx.model.hf_id,
           "model_revision": ctx.model.revision, "method": "raw continuation of token ids, greedy",
           "benchmark": results["benchmark"], "control": results["control"],
           "fisher_p_value": float(p_value), "test_used": "fisher_exact",
           "settings": probe.model_dump(), "interpretation": INTERPRETATION,
           "samples_file": "samples.jsonl"}
    path = out_dir / "probe.json"
    path.write_text(json.dumps(out, indent=2))
    # The per-item text stays out of git (.gitignore ignores results/**/*.jsonl); the
    # manifest records its sha256, so the tracked summary names the file it came from.
    samples_path = out_dir / out["samples_file"]
    with samples_path.open("w") as f:
        for name in ("benchmark", "control"):
            for s in samples[name]:
                f.write(json.dumps({"set": name, **s}) + "\n")
    counts = {"benchmark_hits": hits["benchmark"], "control_hits": hits["control"],
              "n_per_set": probe.n_prompts, "fisher_p_value": float(p_value)}
    ctx.write_manifest(out_dir, "probe", timer=timer, models={"base": ctx.model},
                       engines={"base": engine.settings()}, outputs=[path, samples_path], counts=counts,
                       config={"probe": probe.model_dump(), "control": control_bench.id},
                       extra={"status": "complete"})
    print(f"probe: {hits['benchmark']}/{probe.n_prompts} benchmark, {hits['control']}/{probe.n_prompts} "
          f"control, p={p_value:.4g} -> {path}", flush=True)
    return counts


def main(argv=None) -> int:
    args = stage_parser(__doc__.splitlines()[0]).parse_args(argv)
    run(context_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
