"""Alternative reward designs, kept solely so their weaknesses can be measured.

Every function here is a reward this environment could plausibly have shipped. Three
of them are, in my experience, the *default* choices a first implementation reaches
for, which is exactly why they are worth keeping around: claiming a mitigation matters
is cheap, and showing what an attacker earns without it is not.

None of these is used to score anything real. `indic_extraction_v1.score` is the
shipped reward; this module exists to produce the counterfactual column in the
reward-hacking table.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from indic_extraction_v1.corpus import FIELDS, Document
from indic_extraction_v1.score import format_violation, verbosity
from indic_extraction_v1.verify import (
    _CHECKS,
    _balanced_objects,
    _strip_fences,
    verify_document,
)

RewardFn = Callable[[str, Document, int], float]


def shipped(reply: str, doc: Document, output_tokens: int) -> float:
    """The reward this environment actually uses. Baseline for comparison."""
    from indic_extraction_v1.score import total_reward

    return total_reward(verify_document(reply, doc), output_tokens)


def with_format_bonus(reply: str, doc: Document, output_tokens: int) -> float:
    """Correctness plus a POSITIVE bonus for well-formed output.

    The single most common way an extraction reward becomes hackable. It looks like
    encouragement to follow the schema; it is in fact a payment for producing braces,
    collectable without reading the document at all.
    """
    verdict = verify_document(reply, doc)
    accuracy = verdict.n_correct / len(FIELDS)
    bonus = 0.3 if verdict.schema_clean else 0.0
    return accuracy + bonus - 0.10 * verbosity(output_tokens)


def lenient_extractor(reply: str, doc: Document, output_tokens: int) -> float:
    """Scores the BEST-matching JSON object in the reply instead of the first.

    Superficially this is generosity toward models that think out loud before
    answering. What it actually does is let the model submit a slate of candidates and
    have the grader perform the extraction on its behalf.
    """
    spans = _balanced_objects(_strip_fences(reply))
    best = 0.0
    for span in spans:
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        hits = 0
        for field in FIELDS:
            if field in obj and _CHECKS[field](obj[field], doc.record, doc.lang)[0]:
                hits += 1
        best = max(best, hits / len(FIELDS))
    return best - 0.10 * verbosity(output_tokens)


def membership_credit(reply: str, doc: Document, output_tokens: int) -> float:
    """Accepts a list per field and credits it if the truth appears anywhere in it.

    Reads as robustness to models that hedge. Turns the task into "name every candidate
    in the document", which requires no discrimination whatsoever -- and discrimination
    between competing candidates is the entire task.
    """
    spans = _balanced_objects(_strip_fences(reply))
    if not spans:
        return 0.0
    try:
        obj = json.loads(spans[0])
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(obj, dict):
        return 0.0

    hits = 0
    for field in FIELDS:
        if field not in obj:
            continue
        value = obj[field]
        candidates = value if isinstance(value, list) else [value]
        if any(_CHECKS[field](c, doc.record, doc.lang)[0] for c in candidates):
            hits += 1
    return hits / len(FIELDS) - 0.10 * verbosity(output_tokens)


def char_based_verbosity(reply: str, doc: Document, output_tokens: int) -> float:
    """Identical to the shipped reward except the length penalty counts characters.

    Included because it is the version I would have written without measuring. The
    same content costs between 1.16x and 4.52x more tokens per character in an Indic
    script than in Latin, depending on script and tokeniser, so a character budget
    applies a different standard to the two ways of answering correctly -- a script
    bias living inside the reward rather than the data.

    Note that the offline attack matrix cannot exercise this: the adversarial policies
    all answer in one script. The evidence is the direct token measurement in
    `research/script_bias.py`, not this column.
    """
    verdict = verify_document(reply, doc)
    accuracy = verdict.n_correct / len(FIELDS)
    # Budget chosen to be roughly equivalent to TOKEN_BUDGET for Latin-script output.
    penalty = min(1.0, max(0.0, (len(reply) - 480) / 2400))
    return accuracy - 0.25 * format_violation(verdict) - 0.10 * penalty


ABLATIONS: dict[str, RewardFn] = {
    "shipped": shipped,
    "format_bonus": with_format_bonus,
    "lenient_extractor": lenient_extractor,
    "membership_credit": membership_credit,
    "char_verbosity": char_based_verbosity,
}
