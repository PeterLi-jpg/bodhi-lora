"""Prompt-level conditions that run on the existing eval and grader path.

prompts.py    Inference-time baselines: an instruction block added to the
              unchanged base model at eval time (no training).
ask_first.py  A second chain-of-thought teaching protocol that a cell can
              use in place of BODHI for trace generation and for the
              "+CoT" eval conditions.

Both live here because both are prompts applied to the same models, the same
eval prompts and the same graders as the 2x2.
"""
