"""Inference-time prompt baselines for the eval.

Each baseline adds one instruction block to the unchanged base model at eval
time. Nothing is trained. eval_healthbench.py runs them with --baseline, on the
same eval prompts and with the same graders as the 2x2, so a baseline row is
directly comparable with the base and LoRA rows of the same cell.

clarify
    Tells the model to ask short clarifying questions when facts that would
    change the advice are missing. This is the cheapest way to get context
    seeking without training.

verbalized_confidence
    Tells the model to state its confidence in words and as a number, and to
    name what would change its answer. This follows the verbalized-confidence
    setup of Lin et al. (2022), "Teaching Models to Express Their Uncertainty
    in Words", and Tian et al. (2023), "Just Ask for Calibration". Those papers
    ask for the probability that a short answer is correct. The questions here
    are open-ended and have no single correct answer, so the prompt asks for
    confidence in the main advice.
"""

BASELINE_PROMPTS = {
    "clarify": (
        "You are answering a health question. If facts that would change your "
        "advice are missing, ask the person for them with short, direct "
        "questions. Ask before you give advice that depends on those facts. "
        "If you can give safe general advice now, give it, and say how the "
        "answers to your questions would change it. If no key facts are "
        "missing, answer directly and do not ask questions."
    ),
    "verbalized_confidence": (
        "You are answering a health question. Give your best answer. Then, on "
        "a new line, state how confident you are that your main advice is "
        "correct and safe for this person, in words and as a number from 0 to "
        "100, in this form: \"Confidence: fairly confident (70/100)\". After "
        "that, name the facts that would most change your answer."
    ),
}

PLACEMENTS = ("system", "first_user")


def apply_instructions(messages, instructions, placement="system"):
    """Return a new message list with the instructions added.

    placement="system" puts the instructions in a system message at the
    start. If the conversation already has a system message, the
    instructions go before its text.

    placement="first_user" puts the instructions at the top of the first user
    turn. Use it for chat templates that raise on a system message
    (system_role: false in configs/families.yaml, for example BioMistral).
    Gemma 3 templates do the same merge themselves.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement={placement!r}; use one of {PLACEMENTS}")
    msgs = [dict(m) for m in messages]
    if placement == "system":
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{instructions}\n\n{msgs[0]['content']}"
            return msgs
        return [{"role": "system", "content": instructions}] + msgs
    for m in msgs:
        if m.get("role") == "user":
            m["content"] = f"{instructions}\n\n{m['content']}"
            return msgs
    raise ValueError("first_user placement needs at least one user message")
