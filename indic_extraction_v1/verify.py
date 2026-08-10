"""Deterministic verification of a model response against a `Record`.

Split into two stages that must not be confused:

* **Extraction** -- recover a single JSON object from the model's raw reply.
* **Field verification** -- decide, per field, whether the value is right.

The separation exists because the two failure modes deserve different treatment. A
model that read the document correctly but wrapped its JSON in prose has a formatting
problem; a model that emitted perfect JSON with the wrong amount has a comprehension
problem. Collapsing both into one score destroys the signal that makes this
environment useful for diagnosis, and makes the reward much easier to game.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from indic_extraction_v1.corpus import FIELDS, Document, Record
from indic_extraction_v1.lang import LANGUAGES, LanguageCode
from indic_extraction_v1.normalize import (
    name_tokens,
    normalise_reference,
    parse_amount,
    parse_date,
)


@dataclass(frozen=True)
class FieldVerdict:
    name: str
    correct: bool
    reason: str
    """Why it failed, for error attribution. Never used in scoring."""


@dataclass(frozen=True)
class Verdict:
    parsed: bool
    """A single JSON object was recoverable from the reply."""

    fields: tuple[FieldVerdict, ...]
    extra_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    n_candidate_objects: int
    """How many top-level JSON objects the reply contained. >1 is a format violation:
    see `extract_json`."""

    @property
    def n_correct(self) -> int:
        return sum(1 for f in self.fields if f.correct)

    @property
    def all_correct(self) -> bool:
        return bool(self.fields) and all(f.correct for f in self.fields)

    @property
    def schema_clean(self) -> bool:
        return self.parsed and not self.extra_keys and not self.missing_keys


# ------------------------------------------------------------------------ extraction


def _strip_fences(text: str) -> str:
    """Remove a single leading ```json / ``` fence pair if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]  # drop the opening fence and any language tag
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _balanced_objects(text: str) -> list[str]:
    """Return every top-level balanced {...} span, respecting string literals.

    We count *all* of them rather than stopping at the first because the count itself
    is a signal. A reply carrying several candidate objects is trying, deliberately or
    not, to have the grader pick the best one -- and a grader that obliges is trivially
    exploitable by emitting one object per plausible reading of the document.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
                start = -1
    return spans


def extract_json(reply: str) -> tuple[dict[str, Any] | None, int]:
    """Recover exactly one JSON object from `reply`; return it and the candidate count.

    Deliberately strict. We take the *first* balanced object, never the best-matching
    one, and we report how many were present so the reward can penalise multiplicity.
    Choosing the object that scores highest would convert this environment from an
    extraction task into a multiple-choice task the model writes itself.
    """
    body = _strip_fences(reply)
    spans = _balanced_objects(body)
    if not spans:
        return None, 0
    for span in spans[:1]:
        try:
            loaded = json.loads(span)
        except json.JSONDecodeError:
            return None, len(spans)
        if isinstance(loaded, dict):
            return loaded, len(spans)
    return None, len(spans)


# ---------------------------------------------------------------- field verification


def _check_name(value: Any, record: Record, lang: LanguageCode) -> tuple[bool, str]:
    if not isinstance(value, str) or not value.strip():
        return False, "not-a-nonempty-string"
    honorifics = LANGUAGES[lang].honorifics
    got = name_tokens(value, honorifics)
    if got == name_tokens(record.name_native, honorifics):
        return True, "native"
    lowered = tuple(t.lower() for t in got)
    for roman in record.name_roman:
        if lowered == tuple(t.lower() for t in name_tokens(roman, honorifics)):
            return True, "roman"
    return False, "no-accepted-spelling-matched"


def _check_amount(value: Any, record: Record, lang: LanguageCode) -> tuple[bool, str]:
    parsed = parse_amount(value)
    if parsed is None:
        return False, "unparseable"
    if parsed == record.amount_inr:
        return True, "exact"
    return False, "wrong-value"


def _check_due_date(value: Any, record: Record, lang: LanguageCode) -> tuple[bool, str]:
    if not isinstance(value, str):
        return False, "not-a-string"
    parsed = parse_date(value, LANGUAGES[lang].months)
    if parsed is None:
        return False, "unparseable"
    if parsed == record.due_date:
        return True, "exact"
    # Distinguishing a day/month transposition from an ordinary wrong answer is worth
    # the four lines: it is the single most informative error in this task, and it is
    # invisible in an aggregate accuracy number.
    truth = record.due_date
    if parsed.day == truth.month and parsed.month == truth.day and parsed.year == truth.year:
        return False, "day-month-transposed"
    return False, "wrong-date"


def _check_reference(value: Any, record: Record, lang: LanguageCode) -> tuple[bool, str]:
    if not isinstance(value, str) or not value.strip():
        return False, "not-a-nonempty-string"
    if normalise_reference(value) == normalise_reference(record.reference):
        return True, "exact"
    return False, "wrong-value"


_CHECKS = {
    "name": _check_name,
    "amount_inr": _check_amount,
    "due_date": _check_due_date,
    "reference": _check_reference,
}


def verify(reply: str, record: Record, lang: LanguageCode) -> Verdict:
    """Verify a raw model reply against a record's ground truth."""
    obj, n_objects = extract_json(reply)
    if obj is None:
        return Verdict(
            parsed=False,
            fields=tuple(FieldVerdict(f, False, "no-json") for f in FIELDS),
            extra_keys=(),
            missing_keys=FIELDS,
            n_candidate_objects=n_objects,
        )

    keys = tuple(obj.keys())
    extra = tuple(k for k in keys if k not in FIELDS)
    missing = tuple(f for f in FIELDS if f not in obj)

    verdicts = []
    for field in FIELDS:
        if field not in obj:
            verdicts.append(FieldVerdict(field, False, "missing"))
            continue
        ok, reason = _CHECKS[field](obj[field], record, lang)
        verdicts.append(FieldVerdict(field, ok, reason))

    return Verdict(
        parsed=True,
        fields=tuple(verdicts),
        extra_keys=extra,
        missing_keys=missing,
        n_candidate_objects=n_objects,
    )


def render_prompt(doc: Document) -> str:
    """The exact instruction shown to the model.

    The schema is stated in English while the document stays in its source script.
    That is the realistic deployment shape -- an Indian engineer writes the schema in
    English and the documents arrive in whatever language the citizen used -- and it
    keeps the task about reading the document rather than decoding the instructions.
    """
    return (
        "Extract the following fields from the document below and reply with a single "
        "JSON object and nothing else.\n\n"
        "Schema:\n"
        '  "name"        - the addressee\'s name (the person the document is directed to)\n'
        '  "amount_inr"  - the amount payable, as an integer number of rupees\n'
        '  "due_date"    - the payment due date, as "YYYY-MM-DD"\n'
        '  "reference"   - the document\'s own reference number\n\n'
        "The document may contain other people, other amounts, other dates and other "
        "reference numbers. Extract only the four fields described above.\n"
        "Numeric dates in this document are day-first (DD/MM/YYYY).\n\n"
        "Document:\n"
        f"{doc.text}\n"
    )


def verify_document(reply: str, doc: Document) -> Verdict:
    """Convenience wrapper for the offline paths that already hold a whole Document."""
    return verify(reply, doc.record, doc.lang)
