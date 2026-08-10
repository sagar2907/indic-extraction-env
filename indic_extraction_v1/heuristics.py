"""Shortcut policies that answer without reading, used to audit the corpus.

Every synthetic corpus is guilty until proven innocent. The specific way these corpora
fail is that some surface regularity -- the target is always the biggest number, the
due date is always the last date, the reference is always on line four -- makes a
trivial rule score well. If that happens, a model trained on the environment learns
the regularity rather than the task, and the environment's reported accuracy is
measuring the wrong thing entirely.

So we implement the shortcuts explicitly, as policies, and hold the corpus to the
standard that none of them beats a stated ceiling. The tests in
`tests/test_corpus_is_not_shortcuttable.py` pin those ceilings, which means a future
change to the generator that reintroduces a regularity fails CI instead of quietly
inflating every number downstream.

These policies never call a model. They are pure functions of the document text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date

from indic_extraction_v1.corpus import Document
from indic_extraction_v1.lang import LANGUAGES
from indic_extraction_v1.normalize import fold_digits, normalise_text, parse_amount, parse_date

# A token that looks like a document reference: letters, digits and separators, with
# at least one digit run of 4+. Matches both the true reference and its distractors,
# which is the point -- format alone must not identify the answer.
_REF_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9/\-]*\d{4,}[A-Z0-9/\-]*\b")
_AMOUNT_RE = re.compile(r"[₹]\s*[\d,.]+|\b\d[\d,]*\b")
_DATE_RE = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")


def candidate_amounts(doc: Document) -> list[int]:
    """Every amount-looking value in the document, in order of appearance."""
    text = fold_digits(doc.text)
    out: list[int] = []
    for line in text.splitlines():
        for match in _AMOUNT_RE.finditer(line):
            # Re-parse the whole line fragment so multiplier words ("2.5 लाख") are
            # honoured; a bare digit scrape would read that as 2.
            tail = line[match.start() :]
            value = parse_amount(tail.split()[0]) if tail.split() else None
            if value is None:
                value = parse_amount(match.group(0))
            if value is not None and value not in out:
                out.append(value)
    return out


def candidate_dates(doc: Document) -> list[date]:
    """Every parseable date in the document, in order of appearance."""
    months = LANGUAGES[doc.lang].months
    text = normalise_text(doc.text.replace("\n", " \n ")).replace(" \n ", "\n")
    out: list[date] = []
    for line in text.splitlines():
        for match in _DATE_RE.finditer(line):
            parsed = parse_date(match.group(0), months)
            if parsed and parsed not in out:
                out.append(parsed)
        for index, month_name in enumerate(months, start=1):
            for named in re.finditer(rf"(\d{{1,2}})\s+{re.escape(month_name)}\s+(\d{{4}})", line):
                try:
                    parsed = date(int(named.group(2)), index, int(named.group(1)))
                except ValueError:
                    continue
                if parsed not in out:
                    out.append(parsed)
    return out


def candidate_references(doc: Document) -> list[str]:
    return list(dict.fromkeys(_REF_RE.findall(fold_digits(doc.text).upper())))


def candidate_names(doc: Document) -> list[str]:
    """Lines that look like they carry a person's name, in order of appearance.

    The addressee is the bare first line; other people appear after a label and colon.
    """
    out: list[str] = []
    for line in doc.text.splitlines():
        value = line.split(":", 1)[1].strip() if ":" in line else line.strip()
        if value and not any(ch.isdigit() for ch in fold_digits(value)) and "₹" not in value:
            out.append(value)
    return out


def _reply(name: str | None, amount: int | None, due: date | None, ref: str | None) -> str:
    return json.dumps(
        {
            "name": name if name is not None else "",
            "amount_inr": amount if amount is not None else 0,
            "due_date": due.isoformat() if due is not None else "1970-01-01",
            "reference": ref if ref is not None else "",
        },
        ensure_ascii=False,
    )


def _pick(values: list, index: int):
    if not values:
        return None
    return values[index] if -len(values) <= index < len(values) else values[-1]


def policy_largest_amount(doc: Document) -> str:
    """Biggest number wins. Defeated by `arrears`, which often exceeds the target."""
    amounts = candidate_amounts(doc)
    dates = candidate_dates(doc)
    return _reply(
        _pick(candidate_names(doc), 0),
        max(amounts) if amounts else None,
        max(dates) if dates else None,
        _pick(candidate_references(doc), 0),
    )


def policy_first_of_each(doc: Document) -> str:
    """Take the first candidate of every kind. Defeated by line shuffling."""
    return _reply(
        _pick(candidate_names(doc), 0),
        _pick(candidate_amounts(doc), 0),
        _pick(candidate_dates(doc), 0),
        _pick(candidate_references(doc), 0),
    )


def policy_latest_date(doc: Document) -> str:
    """Latest date wins. Defeated by the `next_cycle` distractor on the hard tier."""
    dates = candidate_dates(doc)
    amounts = candidate_amounts(doc)
    return _reply(
        _pick(candidate_names(doc), 0),
        _pick(amounts, 0),
        max(dates) if dates else None,
        _pick(candidate_references(doc), 0),
    )


def policy_positional(doc: Document) -> str:
    """Assume a fixed line order. Defeated by the shuffle in `corpus._compose`."""
    lines = doc.text.splitlines()

    def field(i: int) -> str:
        return lines[i].split(":", 1)[-1].strip() if i < len(lines) else ""

    months = LANGUAGES[doc.lang].months
    return _reply(
        lines[0] if lines else None,
        parse_amount(field(1)),
        parse_date(field(2), months),
        field(3) or None,
    )


def policy_empty(doc: Document) -> str:
    """Schema-valid but contentless. The floor any format-only reward hands out free."""
    return _reply(None, None, None, None)


POLICIES: dict[str, Callable[[Document], str]] = {
    "largest_amount": policy_largest_amount,
    "first_of_each": policy_first_of_each,
    "latest_date": policy_latest_date,
    "positional": policy_positional,
    "empty": policy_empty,
}
