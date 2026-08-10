"""Deterministic normalisation of extracted field values.

Everything here is a pure function of its input: no clock reads, no locale lookups,
no network, no randomness. That matters because these functions sit directly under
the reward, and a reward that varies with the machine it runs on is not a reward.

The design rule throughout is *normalise, then compare exactly*. We never fall back
to fuzzy string distance. Fuzzy matching would silently hand out partial credit for
wrong answers, which is precisely the kind of soft reward that gets hacked -- and it
would make the reward a function of a similarity threshold nobody can justify.
Instead every accepted spelling is enumerated at generation time (see `lang.Name`).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from indic_extraction_v1.lang import ROMAN_HONORIFICS

# Every Unicode decimal digit maps to its ASCII counterpart. Built from unicodedata
# rather than hand-listed so that a script we have not thought about still folds
# correctly; `str.maketrans` needs codepoint keys, hence the dict comprehension.
_DIGIT_FOLD: dict[int, str] = {}
for _cp in range(0x110000):
    _ch = chr(_cp)
    if unicodedata.category(_ch) == "Nd":
        _DIGIT_FOLD[_cp] = str(unicodedata.digit(_ch))


def fold_digits(text: str) -> str:
    """Map every Unicode decimal digit to ASCII 0-9, leaving all else untouched.

    >>> fold_digits("१२,५०० तक")
    '12,500 तक'
    >>> fold_digits("௧௫/௦௩/௨௦௨௬")
    '15/03/2026'
    """
    return text.translate(_DIGIT_FOLD)


def strip_invisibles(text: str) -> str:
    """Remove zero-width and formatting characters that carry no visible meaning.

    NFC does *not* remove these, which is the trap. ZWNJ (U+200C) and ZWJ (U+200D) are
    legitimately used in Devanagari and Bengali to control conjunct formation, so
    "रमेश शर्मा" and the same string carrying a ZWNJ render identically and mean the same
    thing -- but compare unequal, and were scored as a comprehension failure.

    Everything in Unicode category Cf (format) goes, plus ZWSP, which is category Zs and
    therefore survives both the Cf sweep and NFC. Soft hyphen, BOM and word joiner are Cf
    and are covered.

    This is the same principle as stripping honorifics and accepting romanisations: the
    verifier scores what the model read, never how the text happened to be encoded.
    """
    return "".join(ch for ch in text if ch != "​" and unicodedata.category(ch) != "Cf")


def normalise_text(text: str) -> str:
    """NFC-normalise, strip invisibles, fold digits, collapse whitespace, strip.

    NFC matters for Indic scripts: the same visible string can arrive decomposed or
    composed depending on the model's tokeniser and the client's encoding, and the two
    forms are not equal under `==`. Normalising both sides removes an entire class of
    spurious mismatches that would otherwise look like model errors.

    Order is load-bearing. NFC runs first because composition can itself alter which
    codepoints are present; invisibles are removed next so they cannot survive into the
    digit fold or split a token during whitespace collapse.
    """
    text = unicodedata.normalize("NFC", text)
    text = strip_invisibles(text)
    text = fold_digits(text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- names

_PUNCT = re.compile(r"[.,।৷\-_/]+")  # includes danda (।) and Bengali danda


def name_tokens(raw: str, honorifics: tuple[str, ...]) -> tuple[str, ...]:
    """Reduce a name to comparable tokens, dropping honorifics and punctuation.

    Honorifics are stripped from *both* the reference and the prediction so that
    answering "Ramesh Sharma" for "श्री रमेश शर्मा" is not penalised. Whether the model
    echoes the honorific is a stylistic choice, not a comprehension signal.

    Romanised honorifics are always stripped alongside the language's native ones.
    Dropping only the native set was a real bug: "श्री" vanished but "Shri" survived,
    so a model that transliterated the whole name -- honorific included -- was scored
    against a token list one element longer than the reference. That penalised exactly
    the cross-script answer this environment exists to accept.
    """
    text = normalise_text(raw)
    text = _PUNCT.sub(" ", text)
    drop = {h.strip(". ").lower() for h in honorifics}
    drop |= {h.strip(". ").lower() for h in ROMAN_HONORIFICS}
    return tuple(p for p in text.split() if p and p.strip(". ").lower() not in drop)


# ------------------------------------------------------------------------- amounts

# Indian numbering multipliers, in the spellings the generator emits and the common
# Latin forms a model may answer with.
_MULTIPLIERS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("लाख", "लक्ष", "লক্ষ", "லட்சம்", "lakh", "lakhs", "lac", "lacs"), 100_000),
    (("करोड़", "करोड", "कोटी", "কোটি", "கோடி", "crore", "crores", "cr"), 10_000_000),
    (("हजार", "हज़ार", "हजार", "হাজার", "ஆயிரம்", "thousand"), 1_000),
)

# Latin currency words need word boundaries so they do not eat letters inside a
# larger token. Indic currency words must NOT use a trailing \b: Python's \b is
# defined via str.isalnum(), which is False for combining marks (category Mn), so
# "रु" followed by "." has no boundary between "ु" and "." and the assertion fails.
# This bit us on "रु. 12500/-" returning None. Longest-first alternation keeps
# "रुपये" from being partially consumed by "रु".
_CURRENCY_LATIN = re.compile(r"\b(?:rupees|rs|inr)\b\.?", re.I)
_CURRENCY_SYMBOL = re.compile(
    "[₹$]|" + "|".join(re.escape(w) for w in ("रुपये", "रूपये", "रु", "টাকা", "টা", "ரூபாய்", "ரூ"))
)


def parse_amount(raw: str | int | float) -> int | None:
    """Parse an Indian-format currency amount to whole rupees, or None if unparseable.

    Handles ASCII and Indic digits, Indian lakh/crore grouping (12,34,567), Western
    grouping (1,234,567), currency symbols and words, and multiplier suffixes
    ("1.25 लाख" -> 125000).

    Returns None rather than 0 on failure. This distinction is load-bearing: 0 is a
    legitimate amount, and collapsing "I could not parse this" into "the answer is 0"
    would let a model score on documents whose true amount happened to be zero.

    >>> parse_amount("₹१२,५००")
    12500
    >>> parse_amount("1.25 लाख")
    125000
    >>> parse_amount("12,34,567")
    1234567
    """
    if isinstance(raw, bool):  # bool is an int subclass; reject it explicitly
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        # Half-up, matching the string path below. round() would use banker's
        # rounding and disagree with it on exact .5 values.
        return int(raw + 0.5)
    if not isinstance(raw, str):
        return None

    text = normalise_text(raw).lower()
    text = _CURRENCY_SYMBOL.sub(" ", _CURRENCY_LATIN.sub(" ", text))
    # A trailing "." left behind by "रु." would break the final fullmatch.
    text = text.replace(".", " ", 1) if re.match(r"^\s*\.", text) else text

    multiplier = 1
    for spellings, value in _MULTIPLIERS:
        for spelling in spellings:
            if re.search(rf"(?<![\w]){re.escape(spelling)}(?![\w])", text):
                multiplier = value
                text = text.replace(spelling, " ")
                break
        if multiplier != 1:
            break

    # Strip grouping commas only when they separate digits, so "12,500" folds to
    # "12500" but a stray comma between two numbers does not silently join them.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = text.strip()

    match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s*/-)?", text)
    if not match:
        return None
    value = float(match.group(1)) * multiplier
    # Sub-rupee precision is never meaningful in these documents; round half-up.
    return int(value + 0.5)


def format_indian(amount: int) -> str:
    """Render an integer with Indian digit grouping (last 3, then pairs).

    >>> format_indian(1234567)
    '12,34,567'
    """
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return sign + ",".join([*groups, tail])


# --------------------------------------------------------------------------- dates

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_NUMERIC = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")


def parse_date(raw: str, months: tuple[str, ...] | None = None) -> date | None:
    """Parse a date to a `datetime.date`, or None if unparseable/ambiguous-and-invalid.

    Accepts ISO (YYYY-MM-DD), Indian numeric (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YY) and
    "DD <month-name> YYYY" in any language whose month list is supplied.

    Numeric dates are read **day-first**, which is the Indian convention and the
    convention the corpus is written in. This is not a detail: a model that defaults
    to US month-first ordering will read 03/04/2026 as 4 March instead of 3 April, and
    that is a real, deterministic comprehension failure we want the reward to catch
    rather than paper over. We therefore do *not* try both orderings.
    """
    text = normalise_text(raw)

    iso = _ISO.match(text)
    if iso:
        return _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))

    numeric = _NUMERIC.match(text)
    if numeric:
        day, month, year = (int(g) for g in numeric.groups())
        if year < 100:
            # Two-digit years in these documents are always 21st century.
            year += 2000
        return _safe_date(year, month, day)

    if months:
        for index, month_name in enumerate(months, start=1):
            pattern = rf"^(\d{{1,2}})\s+{re.escape(month_name)}\s+(\d{{4}})$"
            named = re.match(pattern, text)
            if named:
                return _safe_date(int(named.group(2)), index, int(named.group(1)))
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


# ----------------------------------------------------------------------- reference

_REF_STRIP = re.compile(r"[\s\-/]+")


def normalise_reference(raw: str) -> str:
    """Upper-case and remove separators, so 'MH/2026/44871' == 'mh-2026-44871'.

    Separator style is a transcription choice, not a comprehension signal. The digits
    and letters are what the model had to actually read.
    """
    return _REF_STRIP.sub("", normalise_text(raw)).upper()
