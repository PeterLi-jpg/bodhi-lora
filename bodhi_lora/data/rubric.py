"""The synthesized rubric for benchmarks that come with a reference answer, not a rubric.

The rubric filter (pipeline/filter.py) needs a quality signal for every trace, and the
eval reports a rubric score for every response. A benchmark with only a reference
answer gets this shared rubric: its own reference criterion (the record's
reference_criterion, with the reference filled in) plus seven criteria that hold for
any health answer. Eight criteria at six distinct weights give a near-continuous score,
which a threshold needs.
"""
# Why eight and not three: with a few all-or-nothing items the normalized score collapses
# onto a handful of values. On a MedQuAD cell, 82% of traces landed on one score (8/21:
# no positive criterion met, no negative one tripped), so any threshold kept about 18% of
# the traces or nearly all of them.

# The reference criterion's weight and axis.
REFERENCE_POINTS = 8
REFERENCE_TAGS = ("axis:accuracy",)

SHARED_CRITERIA = (
    {"criterion": "The response directly addresses the specific question that was asked.",
     "points": 5, "tags": ["axis:instruction_following"]},
    {"criterion": "The response covers at least one substantive, on-topic clinical point "
                  "(mechanism, risk factor, management step, or diagnostic consideration).",
     "points": 5, "tags": ["axis:completeness"]},
    {"criterion": "The response gives concrete, actionable guidance (specific next steps, "
                  "timeframes, dosing, or thresholds) rather than only generalities.",
     "points": 4, "tags": ["axis:completeness"]},
    {"criterion": "Where information needed to answer safely is missing, the response asks "
                  "for it or explicitly states the limitation.",
     "points": 4, "tags": ["axis:context_awareness"]},
    {"criterion": "The response is clearly written and organized appropriately for its reader.",
     "points": 3, "tags": ["axis:communication_quality"]},
    {"criterion": "The response gives medically dangerous or clearly incorrect advice.",
     "points": -8, "tags": ["axis:accuracy"]},
    {"criterion": "The response offers false reassurance, or fails to recommend escalation "
                  "when the situation warrants it.",
     "points": -5, "tags": ["axis:accuracy"]},
)


def synthesized_rubric(reference_criterion: str) -> list[dict]:
    """The rubric of one row: its reference criterion first, then the shared criteria."""
    first = {"criterion": reference_criterion, "points": REFERENCE_POINTS,
             "tags": list(REFERENCE_TAGS)}
    return [first] + [{**c, "tags": list(c["tags"])} for c in SHARED_CRITERIA]
