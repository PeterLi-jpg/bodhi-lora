"""Grade BOHDI traces with the HealthBench rubric grader and filter by score."""

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# TPU detection — same pattern as train_lora.py.
# When True: use XLA device, bfloat16, no device_map="auto".
# Non-AWQ Qwen2.5-14B-Instruct (28 GB in bfloat16) fits on one v4 chip (32 GB).
try:
    import torch_xla.core.xla_model as _xm
    _ON_TPU = True
except ImportError:
    _xm = None
    _ON_TPU = False

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
    def __init__(self, model_name, device="auto"):
        print(f"Loading grader: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if _ON_TPU:
            # TPU: load in bfloat16 (native precision), move to XLA device.
            # Qwen2.5-14B bfloat16 = 28 GB — fits on one v4 chip (32 GB).
            # device_map="auto" targets CUDA and fails on TPU; don't use it.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            )
            self._device = _xm.xla_device()
            self.model = self.model.to(self._device)
        else:
            # GPU: bfloat16 + automatic device placement across available GPUs.
            # (was float16 previously; bfloat16 is more stable and matches training)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device,
            )
            self._device = next(self.model.parameters()).device

        self.model.eval()

    def grade(self, prompt, max_new_tokens=512):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_toks = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_toks, skip_special_tokens=True)


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


def grade_trace(grader, prompt_messages, response_text, rubric_items, max_retries=3):
    """Grade a single trace. Returns score + per-criterion results + a
    ``parse_failures`` count so callers can surface grader reliability
    (see issue #5 — previously silent parse failures were folded in as
    genuine rubric misses)."""
    convo = prompt_messages + [{"role": "assistant", "content": response_text}]
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
    earned = sum(r["points"] for r in results if r["criteria_met"])
    score = earned / total_pos if total_pos > 0 else 0.0

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
        "overall_score": score,
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
        default="Qwen/Qwen2.5-14B-Instruct",  # full bfloat16, works on GPU and TPU
        # Use Qwen/Qwen2.5-14B-Instruct-AWQ on GPU if VRAM is tight (needs autoawq, CUDA only)
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graded-output", default=None, help="save all graded traces for debugging")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rubrics_by_id = load_rubrics(args.healthbench_data)

    traces = []
    with open(args.input) as f:
        for line in f:
            traces.append(json.loads(line))
    print(f"Loaded {len(traces)} raw traces")

    grader = LocalGrader(args.grader_model)

    graded = []
    for trace in tqdm(traces, desc="Grading"):
        rubrics = rubrics_by_id.get(trace["prompt_id"])
        if rubrics is None:
            print(f"  no rubrics for {trace['prompt_id']}, skipping")
            continue
        result = grade_trace(grader, trace["messages"], trace["response"], rubrics)
        trace["grade"] = result
        graded.append(trace)

    total_parse_failures = sum(t["grade"]["parse_failures"] for t in graded)
    total_rubric_items = sum(len(t["grade"]["criteria_results"]) for t in graded)
    print(f"Graded {len(graded)}/{len(traces)}")
    if total_rubric_items:
        pct = 100.0 * total_parse_failures / total_rubric_items
        print(f"Grader parse failures: {total_parse_failures}/{total_rubric_items} "
              f"rubric items ({pct:.2f}%) — high values indicate grader unreliability, "
              f"not model failure (see issue #5)")

    if args.graded_output:
        p = Path(args.graded_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for item in graded:
                f.write(json.dumps(item) + "\n")
        print(f"All graded traces -> {p}")

    kept = [t for t in graded if t["grade"]["overall_score"] >= args.min_score]
    print(f"Kept {len(kept)}/{len(graded)} (threshold={args.min_score})")

    scores = [t["grade"]["overall_score"] for t in graded]
    if scores:
        print(f"Scores: min={min(scores):.3f} max={max(scores):.3f} "
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
