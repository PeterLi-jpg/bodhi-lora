"""Binary labels on benchmark rows, from facts the graders never see (LabelRule).

A label is True (positive), False (negative) or None (unlabelled). Unlabelled rows stay
out of the label's analyses.

A label reads only structured fields that ship with the dataset: the row's tags or one
of its fields. It never reads free text (a prompt or a rubric criterion) with a keyword
list or a regular expression; docs/CONVENTIONS.md rule 15 says why.
"""
from typing import Optional

from bodhi_lora.registry.schema import LabelRule


def label_value(rule: LabelRule, tags: list[str], fields: Optional[dict] = None) -> Optional[bool]:
    """The value of one label on one row.

    Tag rule: positive when the row carries a positive tag; negative when it carries a
    negative tag, or, if the rule lists no negative tags, when it is not positive. A row
    that is both positive and negative is left unlabelled, so a conflict never decides a
    group.
    Field rule: the row field's value decides, by positive_values and negative_values.
    """
    if rule.field is not None:
        value = (fields or {}).get(rule.field)
        if value in rule.positive_values:
            return True
        if value in rule.negative_values:
            return False
        return None
    tagset = set(tags)
    positive = bool(tagset & set(rule.positive_tags))
    if not rule.negative_tags:
        return positive
    negative = bool(tagset & set(rule.negative_tags))
    if positive and negative:
        return None
    if positive:
        return True
    if negative:
        return False
    return None


def in_groups(value: Optional[bool], groups: list[str]) -> bool:
    """Whether a label value falls in the listed groups ("positive", "negative")."""
    return (value is True and "positive" in groups) or (value is False and "negative" in groups)
