"""Grade BOHDI traces with the HealthBench rubric grader and filter by score."""

import argparse
import json
import random
import re
import statistics
import sys
import os
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _vllm_engine import VLLMEngine
from generate_traces import load_exclude_ids

# same template as healthbench_eval.py in the upstream repo
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


class LocalGrader:
    """Thin wrapper around VLLMEngine for grading rubric items."""

    def __init__(self, engine: VLLMEngine):
        self.engine = engine

    def grade(self, prompt: str, max_new_tokens: int = 512) -> str:
        msgs = [{"role": "user", "content": prompt}]
        return self.engine.chat(msgs, max_new_tokens=max_new_tokens, temperature=0.0)


def parse_json_response(text):
    text = text.strip()
    # direct parse (no fences)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # extract from code fences
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # last resort: find a JSON object containing criteria_met
    m = re.search(r"\{[^{}]*\"criteria_met\"\s*:\s*(?:true|false)[^{}]*\}", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _escape_grader_delimiters(text: str) -> str:
    """Defang grader-template delimiters in untrusted text.

    The grader prompt embeds the conversation between ``<<conversation>>``
    and uses section headers like ``# Final instruction`` / ``# Example``.
    A trace whose response contains those literal substrings could trick
    the grader into treating injected text as instructions (e.g. forcing
    ``criteria_met: true`` regardless of the real answer).

    We backslash-escape the dangerous markers in the trace so the literal
    delimiters never appear in the interpolated prompt. This is mitigation,
    not full prevention (a determined attacker can still find delimiters
    that work), but it raises the bar significantly without changing the
    semantics of legitimate text.
    """
    if not isinstance(text, str):
        return text
    # Escape the template's interpolation markers.
    text = text.replace("<<", r"\<\<").replace(">>", r"\>\>")
    # Escape the template's section-header strings so injected content
    # cannot impersonate "# Final instruction" / "# Example" blocks.
    text = text.replace("# Final instruction", r"\# Final instruction")
    text = text.replace("# Example", r"\# Example")
    return text


def grade_trace(grader, prompt_messages, response_text, rubric_items, max_retries=3):
    """Grade a single trace. Returns score + per-criterion results + a
    ``parse_failures`` count so callers can surface grader reliability
    (see issue #5 — previously silent parse failures were folded in as
    genuine rubric misses).

    Prompt-injection hardening: the grader template uses ``<<conversation>>``
    / ``<<rubric_item>>`` as interpolation delimiters and ``# Final instruction``
    / ``# Example`` as section headers. Untrusted trace content (both the
    prompt messages and the assistant response) is run through
    :func:`_escape_grader_delimiters` so a malicious trace cannot inject
    fake template sections that flip the grader's decision.
    """
    # Escape delimiters in BOTH the user-side messages and the assistant
    # response. Either side could contain attacker-controlled text.
    safe_messages = [
        {"role": m["role"], "content": _escape_grader_delimiters(m.get("content", ""))}
        for m in prompt_messages
    ]
    safe_response = _escape_grader_delimiters(response_text)
    convo = safe_messages + [{"role": "assistant", "content": safe_response}]
    convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo)

    results = []
    parse_failures = 0
    raw_failures = []
    for item in rubric_items:
        crit_str = f"[{item['points']}] {item['criterion']}"
        grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", crit_str)

        parsed = None
        last_raw = None
        for _ in range(max_retries):
            raw = grader.grade(grader_prompt)
            last_raw = raw
            parsed = parse_json_response(raw)
            if "criteria_met" in parsed and isinstance(parsed["criteria_met"], bool):
                break
        else:
            parsed = {"criteria_met": False, "explanation": "grader parse failed"}
            parse_failures += 1
            raw_failures.append({"criterion": item["criterion"], "raw": last_raw})

        results.append({
            "criterion": item["criterion"],
            "points": item["points"],
            "tags": item.get("tags", []),
            "criteria_met": parsed["criteria_met"],
            "explanation": parsed.get("explanation", ""),
            "parse_failed": parsed.get("explanation") == "grader parse failed",
        })

    total_pos = sum(r["points"] for r in results if r["points"] > 0)
    total_neg = sum(r["points"] for r in results if r["points"] < 0)
    total_abs = sum(abs(r["points"]) for r in results)
    earned = sum(r["points"] for r in results if r["criteria_met"])
    positive_items = [r for r in results if r["points"] > 0]
    positive_items_met = sum(1 for r in positive_items if r["criteria_met"])

    # Per issue #4: ``overall_score`` is now the symmetric normalized form
    # ``(earned - neg) / (pos - neg)`` — bounded in [0, 1], comparable
    # across prompts with different penalty structure. The legacy
    # positive-only view (``earned / total_pos``) stays under
    # ``positive_score`` for backward analysis, and the alternate audit
    # views from earlier (``absolute_point_score``, ``positive_criteria_rate``)
    # remain available so anyone can re-derive results under the old metric.
    positive_score = earned / total_pos if total_pos > 0 else 0.0
    score_range = total_pos - total_neg
    normalized_score = (
        (earned - total_neg) / score_range if score_range > 0 else 0.0
    )
    absolute_score = earned / total_abs if total_abs > 0 else 0.0
    positive_rate = (
        positive_items_met / len(positive_items) if positive_items else 0.0
    )

    # per-tag breakdown
    tag_items = defaultdict(list)
    for r in results:
        for tag in r.get("tags", []):
            tag_items[tag].append(r)
    tag_scores = {}
    for tag, items in tag_items.items():
        pos = sum(r["points"] for r in items if r["points"] > 0)
        if pos > 0:
            tag_scores[tag] = sum(r["points"] for r in items if r["criteria_met"]) / pos

    return {
        "overall_score": normalized_score,
        "normalized_score": normalized_score,
        "positive_score": positive_score,
        "absolute_point_score": absolute_score,
        "positive_criteria_rate": positive_rate,
        "score_components": {
            "earned_points": earned,
            "positive_point_total": total_pos,
            "negative_point_total": total_neg,
            "absolute_point_total": total_abs,
            "positive_criteria_total": len(positive_items),
            "positive_criteria_met": positive_items_met,
        },
        "criteria_results": results,
        "tag_scores": tag_scores,
        "parse_failures": parse_failures,
        "raw_parse_failures": raw_failures,
    }


def load_rubrics(paths):
    """Load rubrics from one or more HealthBench JSONL files."""
    rubrics = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                rubrics[ex["prompt_id"]] = ex["rubrics"]
        print(f"  {path}: {len(rubrics)} total rubrics")
    print(f"Loaded rubrics for {len(rubrics)} prompts")
    return rubrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--healthbench-data", required=True, nargs="+")
    parser.add_argument(
        "--grader-model",
        # Asymmetric design: filter uses Qwen2.5-14B-Instruct, eval uses
        # Llama-3.1-8B-Instruct (see scripts/eval_healthbench.py default).
        # Decoupling the filter grader from the eval grader breaks the
        # "graded by your own evaluator" critique — training data is
        # selected under one family, results are reported under another.
        default="Qwen/Qwen2.5-14B-Instruct",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument(
        "--score-field",
        default="overall_score",
        choices=["overall_score", "absolute_point_score", "positive_criteria_rate"],
        help="which grade field to threshold on. "
             "Default keeps the historical behavior; the alternatives make "
             "issue #4 easier to audit without changing the default pipeline.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graded-output", default=None, help="save all graded traces for debugging")
    parser.add_argument(
        "--resume-from", default=None,
        help="Path to a previous graded JSONL (e.g. an earlier --graded-output "
             "file or graded_partial.jsonl). prompt_ids found there will be "
             "skipped and their grades carried forward, mirroring the resume "
             "logic in scripts/generate_traces.py.",
    )
    # Belt-and-suspenders: even if a stale raw_traces.jsonl resumed past
    # the Stage-1 --exclude-ids change, this drops contaminated rows here
    # so they never reach training (issue #60).
    parser.add_argument(
        "--exclude-ids", nargs="+", default=None,
        help="One or more .json/.jsonl files of prompt_ids to drop from the "
             "input traces. Same format as scripts/generate_traces.py.",
    )
    args = parser.parse_args()

    # The grader runs Qwen-14B via vLLM (HF transformers under the hood);
    # without set_seed(), grader sampling drifts across runs even with the
    # same --seed. Mirrors train_lora.py's seeding block (audit N4).
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rubrics_by_id = load_rubrics(args.healthbench_data)

    # JSONL guard: mirror scripts/convert_traces_to_maxtext.py:69-83. A
    # single malformed line silently breaking the run (or worse, producing
    # a half-graded output) is much harder to debug than failing fast with
    # file:line context.
    traces = []
    with open(args.input) as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                traces.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"filter_traces: malformed JSON at {args.input}:{line_no}: {e}"
                ) from e
    print(f"Loaded {len(traces)} raw traces")

    # Empty-file guard: an empty input would silently produce empty
    # train.jsonl/val.jsonl and Stage 3 would quietly train on nothing.
    if len(traces) == 0:
        raise SystemExit("filter_traces: input has 0 rows; check Stage 1 output")

    if args.exclude_ids:
        exclude = load_exclude_ids(args.exclude_ids)
        before = len(traces)
        traces = [t for t in traces if t.get("prompt_id") not in exclude]
        dropped = before - len(traces)
        print(f"Excluded {dropped} traces via --exclude-ids ({len(exclude)} ids); "
              f"{len(traces)} remain")

    # Schema validation: catch missing required fields up-front with a clear
    # error message rather than letting a KeyError surface from inside a
    # ThreadPoolExecutor worker (where the prompt_id context is lost).
    REQUIRED_KEYS = ("prompt_id", "messages", "response")
    for i, trace in enumerate(traces):
        for key in REQUIRED_KEYS:
            if key not in trace:
                raise KeyError(
                    f"trace at row {i} missing required key {key!r}: "
                    f"prompt_id={trace.get('prompt_id', '?')}"
                )

    # Stage 2 resume (mirrors generate_traces.py:194-241). If a previous
    # run wrote graded rows, load them and skip those prompt_ids in the
    # grading loop. Anything we already graded is carried forward into
    # ``graded`` so train/val splits below see the full corpus.
    done_ids = set()
    graded = []
    if args.resume_from and Path(args.resume_from).exists():
        with open(args.resume_from) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Skip corrupt lines from interrupted runs (matches
                    # generate_traces.py's permissive resume).
                    continue
                pid = row.get("prompt_id")
                if pid is None or "grade" not in row:
                    continue
                done_ids.add(pid)
                graded.append(row)
        print(
            f"Resuming from {args.resume_from}: skipping "
            f"{len(done_ids)} already-graded prompt_ids"
        )

    # Concurrent grading — vLLM batches concurrent requests server-side, so
    # submitting N at once gives ~Nx throughput up to its scheduling limit.
    # 16 is a sweet spot: with 32 we saw vLLM crash from KV-cache pressure
    # mid-run on Qwen2.5-14B with TP=8 (issue: single bad trace took down
    # the whole pipeline). Keep this conservative.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    GRADER_CONCURRENCY = int(os.environ.get("GRADER_CONCURRENCY", "16"))

    # Periodic partial-write: append every N graded rows to a sidecar so a
    # crash mid-run doesn't lose all progress. Sits next to --graded-output
    # if set, otherwise under --output-dir. Append mode so resume runs add
    # to existing rows. No lock needed: only the main thread writes here
    # (workers return values via as_completed).
    PARTIAL_FLUSH_EVERY = 50
    if args.graded_output:
        partial_path = Path(args.graded_output).with_suffix(".partial.jsonl")
    else:
        partial_path = out_dir / "graded_partial.jsonl"
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    failed_traces = []
    with VLLMEngine(args.grader_model) as engine:
        grader = LocalGrader(engine)
        # Filter to traces that have rubrics (preserve original skip-and-print behavior)
        # and skip anything we've already graded (resume path).
        tasks = []
        for trace in traces:
            if trace["prompt_id"] in done_ids:
                continue
            rubrics = rubrics_by_id.get(trace["prompt_id"])
            if rubrics is None:
                print(f"  no rubrics for {trace['prompt_id']}, skipping")
                continue
            tasks.append((trace, rubrics))

        def _grade_one(args_pair):
            trace, rubrics = args_pair
            try:
                result = grade_trace(grader, trace["messages"], trace["response"], rubrics)
                trace["grade"] = result
                return trace, None
            except Exception:
                # Don't let one bad trace (e.g. transient vLLM HTTP error,
                # malformed prompt) take down the whole pipeline. Log and
                # skip; keep the last 30 lines of the traceback so the
                # exact failure is visible without overwhelming logs.
                tb = "\n".join(traceback.format_exc().strip().splitlines()[-30:])
                return None, (trace.get("prompt_id", "?"), tb)

        with open(partial_path, "a") as partial_f:
            with ThreadPoolExecutor(max_workers=GRADER_CONCURRENCY) as ex:
                futures = [ex.submit(_grade_one, t) for t in tasks]
                new_since_flush = 0
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Grading"):
                    trace, err = fut.result()
                    if trace is not None:
                        graded.append(trace)
                        partial_f.write(json.dumps(trace) + "\n")
                        new_since_flush += 1
                        if new_since_flush >= PARTIAL_FLUSH_EVERY:
                            partial_f.flush()
                            new_since_flush = 0
                    else:
                        failed_traces.append(err)

    if failed_traces:
        print(f"\nWARNING: {len(failed_traces)} traces failed grading and were skipped:")
        for pid, err in failed_traces[:10]:
            print(f"  {pid}: {err}")
        if len(failed_traces) > 10:
            print(f"  ... and {len(failed_traces) - 10} more")

    # Aggregate per-trace parse_failures into a single end-of-run summary so
    # grader instability shows up without grepping stage logs (audit N7).
    # Each parse failure is otherwise silently folded in as a "criteria not met"
    # rubric item — under heavy load Qwen-14B can hit 10-30% parse-failure
    # rates and still produce a plausible-looking score distribution.
    total_parse_failures = sum(t["grade"]["parse_failures"] for t in graded)
    total_rubric_items = sum(len(t["grade"]["criteria_results"]) for t in graded)
    print(f"Graded {len(graded)}/{len(traces)}")
    if total_rubric_items:
        pct = 100.0 * total_parse_failures / total_rubric_items
        print(
            f"  parse failures: {total_parse_failures}/{total_rubric_items} "
            f"({pct:.1f}%)",
            file=sys.stderr,
        )
        if total_parse_failures > 0.05 * total_rubric_items:
            print(
                f"  WARNING: parse-failure rate {pct:.1f}% > 5% "
                f"— grader may be unstable",
                file=sys.stderr,
            )

    if args.graded_output:
        p = Path(args.graded_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for item in graded:
                f.write(json.dumps(item) + "\n")
        print(f"All graded traces -> {p}")

    # ``--score-field`` defaults to overall_score, which is now the
    # normalized formula. Existing audits using --score-field absolute_point_score
    # / positive_criteria_rate / positive_score still work.
    kept = [t for t in graded if t["grade"][args.score_field] >= args.min_score]
    print(
        f"Kept {len(kept)}/{len(graded)} "
        f"(score_field={args.score_field}, threshold={args.min_score})"
    )

    scores = [t["grade"][args.score_field] for t in graded]
    if scores:
        print(f"Normalized scores: min={min(scores):.3f} max={max(scores):.3f} "
              f"mean={sum(scores)/len(scores):.3f} median={statistics.median(scores):.3f}")

    random.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_ratio)) if len(kept) > 1 else 0
    val, train = kept[:n_val], kept[n_val:]

    for name, data in [("train", train), ("val", val)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"{name}: {len(data)} -> {path}")

    print(f"\n--- summary ---")
    print(f"raw={len(traces)} graded={len(graded)} kept={len(kept)} "
          f"train={len(train)} val={len(val)}")


if __name__ == "__main__":
    main()
