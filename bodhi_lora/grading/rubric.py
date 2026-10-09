"""The rubric grader: one grading call per rubric item, then the normalized score.

The grader template is the one of the upstream rubric grader (simple-evals); a response
meets a criterion when the grader returns criteria_met true. The score of a response is
(earned - negative total) / (positive total - negative total), in [0, 1]. The filter
grades teacher traces with it, and the eval grades every response with it.
"""

import json
import re
from collections import defaultdict

# Tokens of one grading call: room for the JSON verdict and a short explanation.
RUBRIC_GRADE_TOKENS = 512

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


class Grader:
    """Greedy grading calls to an engine: one user message in, the grader's text out."""

    def __init__(self, engine):
        self.engine = engine

    def grade(self, prompt: str, max_new_tokens: int = RUBRIC_GRADE_TOKENS) -> str:
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


def grade_rubric(grader, prompt_messages, response_text, rubric_items, max_retries=3):
    """Grade one response against its rubric items. Returns the score, per-criterion
    results and a ``parse_failures`` count, so callers can report grader reliability
    (issue #5: parse failures used to fold in silently as rubric misses).

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
