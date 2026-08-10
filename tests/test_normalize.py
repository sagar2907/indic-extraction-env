"""Normalisation is pure and deterministic; these tests pin its exact behaviour."""

from __future__ import annotations

from datetime import date

import pytest

from indic_extraction_v1.lang import BENGALI, HINDI, MARATHI, TAMIL
from indic_extraction_v1.normalize import (
    fold_digits,
    format_indian,
    name_tokens,
    normalise_reference,
    normalise_text,
    parse_amount,
    parse_date,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("१२,५००", "12,500"),
        ("௧௫/௦௩/௨௦௨௬", "15/03/2026"),
        ("০১২৩", "0123"),
        ("no digits here", "no digits here"),
        ("mixed १2३", "mixed 123"),
    ],
)
def test_fold_digits(raw: str, expected: str) -> None:
    assert fold_digits(raw) == expected


def test_normalise_text_is_nfc_stable() -> None:
    """Decomposed and composed Devanagari must compare equal after normalisation.

    Model output arrives in whichever normalisation form the provider's tokeniser and
    the transport happened to produce. Without NFC folding these are unequal strings
    that render identically, which would surface as a phantom model error.
    """
    composed = "नि"  # ni, already composed
    decomposed = "नि"
    assert normalise_text(composed) == normalise_text(decomposed)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("₹12,500", 12500),
        ("₹१२,५००", 12500),
        ("12,34,567", 1234567),  # Indian grouping
        ("1,234,567", 1234567),  # Western grouping
        ("1.25 लाख", 125000),
        ("₹2.5 करोड़", 25000000),
        ("₹3 कोटी", 30000000),
        ("₹2.4 லட்சம்", 240000),
        ("₹1.5 কোটি", 15000000),
        ("5 हजार", 5000),
        ("Rs. 12,500", 12500),
        ("INR 12500", 12500),
        ("12500 रुपये", 12500),
        ("২৫০০ টাকা", 2500),
        ("ரூ 12500", 12500),
        (12500, 12500),
        (12500.4, 12500),
    ],
)
def test_parse_amount_accepts(raw, expected: int) -> None:
    assert parse_amount(raw) == expected


def test_parse_amount_handles_indic_currency_prefix_with_period() -> None:
    """Regression: 'रु. 12500/-' returned None.

    The currency pattern ended in \\b after an Indic token. Python defines \\b through
    str.isalnum(), which is False for combining marks (category Mn), so there is no
    boundary between the vowel sign in 'रु' and the following '.' and the assertion
    failed -- leaving a stray '.' that broke the final numeric match. Word-boundary
    assertions are not reliable at the end of Indic tokens.
    """
    assert parse_amount("रु. 12500/-") == 12500
    assert parse_amount("ரூ. 12,500/-") == 12500
    assert parse_amount("টা. 12,500/-") == 12500


@pytest.mark.parametrize("raw", ["garbage", "", "12 और 500", None, True, False, [1]])
def test_parse_amount_rejects(raw) -> None:
    """Unparseable input must be None, never 0.

    0 is a legitimate amount. Collapsing failure into 0 would hand out credit on any
    document whose true amount happened to be zero, and would hide parser bugs.
    """
    assert parse_amount(raw) is None


def test_parse_amount_rejects_bool_specifically() -> None:
    """bool subclasses int; without an explicit guard True would parse as 1."""
    assert parse_amount(True) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-03-15", date(2026, 3, 15)),
        ("15/03/2026", date(2026, 3, 15)),
        ("15-03-2026", date(2026, 3, 15)),
        ("15.03.26", date(2026, 3, 15)),
        ("२०२६-०३-१५", date(2026, 3, 15)),
        ("௧௫/௦௩/௨௦௨௬", date(2026, 3, 15)),
    ],
)
def test_parse_date_numeric(raw: str, expected: date) -> None:
    assert parse_date(raw) == expected


def test_parse_date_is_day_first() -> None:
    """03/04/2026 is 3 April, not 4 March.

    The corpus is written in the Indian day-first convention, so a month-first reading
    is a genuine comprehension error the reward should catch. The parser deliberately
    does not try both orderings -- doing so would forgive exactly the mistake we most
    want to measure.
    """
    assert parse_date("03/04/2026") == date(2026, 4, 3)


@pytest.mark.parametrize(
    ("lang", "raw"),
    [
        (HINDI, "१५ मार्च २०२६"),
        (MARATHI, "15 मार्च 2026"),
        (TAMIL, "15 மார்ச் 2026"),
        (BENGALI, "15 মার্চ 2026"),
    ],
)
def test_parse_date_named_months(lang, raw: str) -> None:
    assert parse_date(raw, lang.months) == date(2026, 3, 15)


@pytest.mark.parametrize("raw", ["31/02/2026", "nonsense", "", "2026-13-01"])
def test_parse_date_rejects(raw: str) -> None:
    assert parse_date(raw) is None


def test_format_indian_grouping() -> None:
    assert format_indian(1234567) == "12,34,567"
    assert format_indian(12500) == "12,500"
    assert format_indian(500) == "500"
    assert format_indian(0) == "0"


def test_normalise_reference_ignores_separator_style() -> None:
    assert normalise_reference("MH/2026/44871") == normalise_reference("mh-2026-44871")
    assert normalise_reference("MH 2026 44871") == "MH202644871"


def test_name_tokens_strip_roman_honorifics_too() -> None:
    """Regression: only native honorifics were stripped.

    'श्री' was removed but 'Shri' survived, so a model that transliterated the entire
    name -- honorific included -- was compared against a token list one element longer
    than the reference and scored wrong. That penalised precisely the cross-script
    answer this environment exists to treat as correct.
    """
    assert name_tokens("श्री रमेश शर्मा", HINDI.honorifics) == ("रमेश", "शर्मा")
    assert name_tokens("Shri Ramesh Sharma", HINDI.honorifics) == ("Ramesh", "Sharma")
    assert name_tokens("Dr. Ramesh Sharma", HINDI.honorifics) == ("Ramesh", "Sharma")
    assert name_tokens("Ramesh Sharma", HINDI.honorifics) == ("Ramesh", "Sharma")


ZWNJ = "\u200c"
ZWJ = "\u200d"
ZWSP = "\u200b"
BOM = "\ufeff"
SOFT_HYPHEN = "\u00ad"
WORD_JOINER = "\u2060"


def test_zero_width_characters_do_not_break_name_matching() -> None:
    """Regression: an invisible character scored a correct answer as wrong.

    ZWNJ (U+200C) and ZWJ (U+200D) are legitimately used in Devanagari and Bengali to
    control conjunct formation. NFC does not remove them, so "रमेश शर्मा" and the same
    name carrying a ZWNJ render identically, mean the same thing, and compared unequal --
    the model was marked down for how its output happened to be encoded.

    Same class of defect as the honorific and romanisation bugs: penalising a correct
    answer for an invisible presentation difference, which is the one thing this
    verifier exists not to do.
    """
    plain = "रमेश शर्मा"
    with_zwnj = "रमेश" + ZWNJ + " शर्मा"
    assert name_tokens(plain, HINDI.honorifics) == name_tokens(with_zwnj, HINDI.honorifics)


@pytest.mark.parametrize(
    ("name", "char"),
    [
        ("ZWSP", ZWSP),
        ("ZWNJ", ZWNJ),
        ("ZWJ", ZWJ),
        ("BOM", BOM),
        ("SOFT HYPHEN", SOFT_HYPHEN),
        ("WORD JOINER", WORD_JOINER),
    ],
)
def test_invisible_characters_are_stripped(name: str, char: str) -> None:
    """Every zero-width or format character must be gone after normalisation.

    ZWSP is category Zs rather than Cf, so it survives a category-Cf sweep and needs
    naming explicitly -- which is exactly the kind of gap a parametrised test catches.
    """
    assert char not in normalise_text("अ" + char + "ब")
    assert normalise_text("अ" + char + "ब") == "अब"


@pytest.mark.parametrize(
    ("plain", "marked"),
    [
        ("अन्तर", "अन्" + ZWNJ + "तर"),
        ("क्ष", "क्" + ZWJ + "ष"),
        ("অন্য", "অন্" + ZWNJ + "য"),
    ],
)
def test_conjunct_control_characters_compare_equal(plain: str, marked: str) -> None:
    """Strings differing only by conjunct-control marks are the same string."""
    assert normalise_text(plain) == normalise_text(marked)


def test_stripping_invisibles_does_not_eat_visible_text() -> None:
    """The sweep must remove nothing a reader can see.

    A category-based filter is blunt, so this pins that ordinary Indic text, Latin text,
    digits, currency and punctuation all survive intact.
    """
    assert normalise_text("श्री रमेश कुमार शर्मा") == "श्री रमेश कुमार शर्मा"
    assert normalise_text("₹12,500/-") == "₹12,500/-"
    assert normalise_text("திரு லட்சுமி") == "திரு லட்சுமி"
    assert normalise_text("MH/2026/44871") == "MH/2026/44871"
    # A string made only of invisibles collapses to empty rather than to whitespace.
    assert normalise_text(ZWNJ + ZWJ + BOM) == ""
