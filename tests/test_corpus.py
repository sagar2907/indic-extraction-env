"""Corpus generation must be reproducible, self-consistent and free of script leaks."""

from __future__ import annotations

import collections

import pytest

from indic_extraction_v1.corpus import FIELDS, TIERS, generate, generate_many
from indic_extraction_v1.lang import DIGITS, LANGUAGES
from indic_extraction_v1.verify import verify_document


def test_generation_is_reproducible() -> None:
    assert [d.text for d in generate_many(50, seed=3)] == [
        d.text for d in generate_many(50, seed=3)
    ]


def test_generation_is_index_independent() -> None:
    """generate(n) must not depend on whether 0..n-1 were generated first.

    Independence is what allows a run to resume, shard across providers, or evaluate a
    subset without silently producing a different corpus than the full run would have.
    """
    batch = generate_many(40, seed=5)
    for i in (0, 7, 23, 39):
        assert generate(i, seed=5).text == batch[i].text


def test_different_seeds_give_different_corpora() -> None:
    assert generate(0, seed=1).text != generate(0, seed=2).text


def test_no_clock_dependence() -> None:
    """Dates are anchored to a fixed epoch, so the corpus cannot drift over time.

    A generator that read today's date would produce a different corpus on every run,
    making every measurement in this repository unreproducible after the fact.
    """
    for doc in generate_many(200, seed=11):
        assert doc.record.due_date.year in (2026, 2027)


@pytest.mark.parametrize("tier", TIERS)
def test_tier_filter(tier: str) -> None:
    assert all(d.tier == tier for d in generate_many(30, seed=2, tier=tier))


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_lang_filter(lang: str) -> None:
    assert all(d.lang == lang for d in generate_many(30, seed=2, lang=lang))


def test_ground_truth_appears_in_document() -> None:
    """Every answer must actually be present in the text that is shown to the model.

    Without this, an unanswerable row would be indistinguishable from a model failure.
    """
    from indic_extraction_v1.normalize import fold_digits, normalise_reference

    for doc in generate_many(200, seed=13):
        assert doc.record.name_native in doc.text
        flat = normalise_reference(doc.text)
        assert normalise_reference(doc.record.reference) in flat
        # The amount may be rendered in words ("2.5 लाख"), so assert the digits are
        # present only when the plain or grouped form was used.
        folded = fold_digits(doc.text).replace(",", "")
        if (
            "लाख" not in doc.text
            and "करोड़" not in doc.text
            and "कोटी" not in doc.text
            and "லட்சம்" not in doc.text
            and "கோடி" not in doc.text
            and "লক্ষ" not in doc.text
        ):
            assert str(doc.record.amount_inr) in folded


@pytest.mark.parametrize("lang_code", sorted(LANGUAGES))
def test_no_cross_script_currency_leaks(lang_code: str) -> None:
    """A Tamil document must not contain Devanagari currency or multiplier words.

    Regression: `_render_amount` hardcoded the Hindi forms, so every language's
    documents carried 'रु.' and 'लाख'. Besides being unreadable, it is a spurious cue
    that a model could learn instead of the task.
    """
    others = [lang.currency for code, lang in LANGUAGES.items() if code != lang_code] + [
        lang.lakh for code, lang in LANGUAGES.items() if code != lang_code
    ]
    for doc in generate_many(120, seed=17, lang=lang_code):  # type: ignore[arg-type]
        for foreign in others:
            if foreign in (LANGUAGES[lang_code].currency, LANGUAGES[lang_code].lakh):
                continue
            assert foreign not in doc.text, f"{foreign!r} leaked into {lang_code}"


def test_native_digits_only_from_own_script() -> None:
    """A document may use ASCII or its own script's digits, never another script's."""
    for doc in generate_many(200, seed=19):
        own = DIGITS[LANGUAGES[doc.lang].script]
        for script, glyphs in DIGITS.items():
            if script in ("latn", LANGUAGES[doc.lang].script):
                continue
            assert not any(g in doc.text for g in glyphs), f"{script} digits in {doc.lang}"
        assert own, f"no digit set defined for {doc.lang}"


def test_gold_answer_scores_perfectly() -> None:
    """The verifier must accept its own reference answer on every row.

    This is the model-free validation hook expressed as a test. A row the verifier
    cannot grade would look like a permanent model failure and quietly cap the ceiling.
    """
    import json

    for doc in generate_many(300, seed=23):
        gold = json.dumps(
            {
                "name": doc.record.name_native,
                "amount_inr": doc.record.amount_inr,
                "due_date": doc.record.due_date.isoformat(),
                "reference": doc.record.reference,
            },
            ensure_ascii=False,
        )
        verdict = verify_document(gold, doc)
        assert verdict.all_correct, (doc.idx, [(f.name, f.reason) for f in verdict.fields])


def test_gold_answer_in_roman_also_scores_perfectly() -> None:
    """Answering with an accepted transliteration must score identically to the native form."""
    import json

    for doc in generate_many(200, seed=29):
        gold = json.dumps(
            {
                "name": doc.record.name_roman[0],
                "amount_inr": doc.record.amount_inr,
                "due_date": doc.record.due_date.isoformat(),
                "reference": doc.record.reference,
            },
            ensure_ascii=False,
        )
        assert verify_document(gold, doc).all_correct, doc.idx


def test_corpus_is_balanced() -> None:
    docs = generate_many(1200, seed=31)
    langs = collections.Counter(d.lang for d in docs)
    tiers = collections.Counter(d.tier for d in docs)
    for count in langs.values():
        assert count > len(docs) / len(LANGUAGES) * 0.7
    for count in tiers.values():
        assert count > len(docs) / len(TIERS) * 0.7


def test_fields_constant_is_the_schema() -> None:
    assert FIELDS == ("name", "amount_inr", "due_date", "reference")
