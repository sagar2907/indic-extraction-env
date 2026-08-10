"""JSON extraction and field verification, including the adversarial shapes."""

from __future__ import annotations

import json

import pytest

from indic_extraction_v1.corpus import generate, generate_many
from indic_extraction_v1.verify import extract_json, verify_document


def _doc():
    return generate(7, seed=1)


def _gold(doc, **overrides) -> str:
    payload = {
        "name": doc.record.name_native,
        "amount_inr": doc.record.amount_inr,
        "due_date": doc.record.due_date.isoformat(),
        "reference": doc.record.reference,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_perfect_reply() -> None:
    doc = _doc()
    verdict = verify_document(_gold(doc), doc)
    assert verdict.all_correct and verdict.schema_clean


@pytest.mark.parametrize(
    "wrapper",
    [
        "{body}",
        "```json\n{body}\n```",
        "```\n{body}\n```",
        "Here is the answer:\n{body}\nHope that helps.",
        "  \n {body}  \n",
    ],
)
def test_json_survives_common_wrappers(wrapper: str) -> None:
    """Fences and surrounding prose are formatting noise, not comprehension failures."""
    doc = _doc()
    assert verify_document(wrapper.format(body=_gold(doc)), doc).all_correct


def test_multiple_objects_are_counted_not_searched() -> None:
    """A reply with several candidates must be graded on the FIRST, and flagged.

    Picking the best-scoring object would turn the task into a multiple-choice quiz the
    model writes for itself: emit one object per plausible reading of the document and
    the grader finds the right one. The count feeds the format penalty instead.
    """
    doc = _doc()
    decoy = json.dumps({"name": "x", "amount_inr": 1, "due_date": "2020-01-01", "reference": "y"})
    verdict = verify_document(decoy + "\n" + _gold(doc), doc)
    assert verdict.n_candidate_objects == 2
    assert not verdict.all_correct, "grader must not search for the best object"


def test_first_object_is_used_even_when_second_is_correct() -> None:
    doc = _doc()
    decoy = json.dumps({"name": "x", "amount_inr": 1, "due_date": "2020-01-01", "reference": "y"})
    assert verify_document(decoy + _gold(doc), doc).n_correct == 0


def test_braces_inside_strings_do_not_break_span_detection() -> None:
    doc = _doc()
    reply = _gold(doc, name=doc.record.name_native + "")
    tricky = reply[:-1] + ', "note": "a } brace"}'
    obj, count = extract_json(tricky)
    assert count == 1 and obj is not None


def test_empty_json_is_parsed_but_scores_zero() -> None:
    doc = _doc()
    verdict = verify_document("{}", doc)
    assert verdict.parsed
    assert verdict.n_correct == 0
    assert verdict.missing_keys == ("name", "amount_inr", "due_date", "reference")
    assert not verdict.schema_clean


def test_no_json_at_all() -> None:
    doc = _doc()
    verdict = verify_document("I am unable to help with that.", doc)
    assert not verdict.parsed
    assert verdict.n_correct == 0


def test_extra_keys_are_flagged() -> None:
    doc = _doc()
    verdict = verify_document(_gold(doc, confidence=0.9), doc)
    assert verdict.all_correct
    assert verdict.extra_keys == ("confidence",)
    assert not verdict.schema_clean


def test_amount_accepts_surface_variation() -> None:
    doc = _doc()
    amount = doc.record.amount_inr
    for value in (amount, f"₹{amount}", f"Rs. {amount}"):
        assert verify_document(_gold(doc, amount_inr=value), doc).all_correct


def test_reference_accepts_separator_variation() -> None:
    doc = _doc()
    mangled = doc.record.reference.lower().replace("-", "/")
    assert verify_document(_gold(doc, reference=mangled), doc).all_correct


def test_wrong_amount_fails() -> None:
    doc = _doc()
    verdict = verify_document(_gold(doc, amount_inr=doc.record.amount_inr + 1), doc)
    assert not verdict.all_correct
    assert [f.reason for f in verdict.fields if f.name == "amount_inr"] == ["wrong-value"]


def test_day_month_transposition_is_labelled() -> None:
    """The month-first misreading gets its own reason string.

    It is the single most diagnostic error in this task -- a locale bug rather than a
    reading failure -- and it is invisible in an aggregate accuracy number.
    """
    doc = next(d for d in generate_many(200, seed=1) if d.record.due_date.day <= 12)
    due = doc.record.due_date
    swapped = f"{due.year}-{due.day:02d}-{due.month:02d}"
    verdict = verify_document(_gold(doc, due_date=swapped), doc)
    assert [f.reason for f in verdict.fields if f.name == "due_date"] == ["day-month-transposed"]


def test_null_values_are_not_credited() -> None:
    doc = _doc()
    verdict = verify_document(
        json.dumps({"name": None, "amount_inr": None, "due_date": None, "reference": None}), doc
    )
    assert verdict.n_correct == 0


def test_list_values_are_not_credited() -> None:
    """Shotgunning a list of candidates must not score.

    If any field accepted a collection and checked membership, the optimal policy would
    be to list every candidate in the document for every field.
    """
    doc = _doc()
    verdict = verify_document(
        json.dumps(
            {
                "name": [doc.record.name_native, "other"],
                "amount_inr": [doc.record.amount_inr, 1],
                "due_date": [doc.record.due_date.isoformat()],
                "reference": [doc.record.reference],
            },
            ensure_ascii=False,
        ),
        doc,
    )
    assert verdict.n_correct == 0
