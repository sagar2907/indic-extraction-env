"""Seeded, deterministic generation of Indic-script documents with exact ground truth.

Why generate rather than scrape or label? Three reasons, in order of weight:

1. The ground truth is exact *by construction*. There is no annotation noise, so a
   disagreement between model and reference is always the model's, never the label's.
2. It is reproducible from an integer. Anyone can regenerate byte-identical tasks
   without downloading anything, which is what makes this environment installable
   with zero dependencies and runnable offline.
3. We control the difficulty axes independently -- digit system, date format, amount
   wording, distractor count -- so a failure can be attributed to a cause instead of
   being a single opaque accuracy number.

The cost is realism, and the mitigation is adversarial design: every distractor here
exists to defeat a specific shortcut that would otherwise score well without any
reading. See `heuristics.py`, which implements those shortcuts as explicit baselines,
and the tests that pin them near chance.

Determinism rules observed throughout: no clock reads, no `hash()` (which is salted
per process), no iteration over sets or dicts whose order could vary, and one
independently seeded RNG per task so that generating task N never depends on
whether tasks 0..N-1 were generated first.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from indic_extraction_v1.lang import DIGITS, LANGUAGES, Language, LanguageCode, Name
from indic_extraction_v1.normalize import format_indian

Tier = Literal["easy", "medium", "hard"]
TIERS: tuple[Tier, ...] = ("easy", "medium", "hard")

# The four fields a model must extract. Ordering is fixed and load-bearing: it is the
# order shown in the prompt schema and the order used for per-field reporting.
FIELDS: tuple[str, ...] = ("name", "amount_inr", "due_date", "reference")


@dataclass(frozen=True)
class Record:
    """The ground-truth answer for one document."""

    name_native: str
    name_roman: tuple[str, ...]
    """Every romanisation of `name_native` the verifier will accept."""

    amount_inr: int
    due_date: date
    reference: str


@dataclass(frozen=True)
class Document:
    """One generated task: the rendered text plus its exact answer."""

    idx: int
    lang: LanguageCode
    tier: Tier
    text: str
    record: Record
    issue_date: date
    """Kept so the surface can be re-rendered from the same ground truth.

    `render_variant` needs every input `_compose` consumed, and the issue date is the
    only one not already carried on the record."""

    features: dict[str, str]
    """Which difficulty knobs fired, for per-feature error attribution in analysis."""


# --------------------------------------------------------------------------- digits


def render_digits(value: str, script: str) -> str:
    """Rewrite ASCII digits in `value` into `script`'s digit set."""
    table = str.maketrans(DIGITS["latn"], DIGITS[script])
    return value.translate(table)


# --------------------------------------------------------------------------- pieces


def _pick_name(rng: random.Random, lang: Language) -> tuple[str, tuple[str, ...]]:
    """Build a two-part name and the cross-product of its accepted romanisations."""
    given: Name = rng.choice(lang.given)
    surname: Name = rng.choice(lang.surnames)
    native = f"{given.native} {surname.native}"
    roman = tuple(f"{g} {s}" for g in given.roman for s in surname.roman)
    return native, roman


def _render_amount(rng: random.Random, amount: int, lang: Language, tier: Tier) -> str:
    """Render an amount in one of several real-world surface forms.

    On the hard tier an amount that is a clean multiple of a lakh or crore may be
    written in words ("2.5 करोड़"), which is the form that most often defeats a naive
    digit-scraping regex.

    Currency and multiplier words come from `lang`, never hardcoded. An earlier version
    emitted Devanagari "रु." inside Tamil and Bengali documents, which is not a form any
    reader would encounter and would have taught a model a spurious cue.
    """
    styles = ["plain", "grouped"]
    if tier != "easy":
        styles += ["symbol", "native"]
    if tier == "hard" and amount >= 100_000 and amount % 10_000 == 0:
        styles.append("multiplier")
    style = rng.choice(styles)

    if style == "multiplier":
        if amount >= 10_000_000 and amount % 100_000 == 0:
            return f"₹{amount / 10_000_000:g} {lang.crore}"
        return f"₹{amount / 100_000:g} {lang.lakh}"
    if style == "plain":
        return f"₹{amount}"
    if style == "grouped":
        return f"₹{format_indian(amount)}"
    if style == "symbol":
        return f"{lang.currency} {format_indian(amount)}/-"
    return "₹" + render_digits(format_indian(amount), lang.script)


def _render_date(rng: random.Random, value: date, lang: Language, tier: Tier) -> str:
    """Render a date in one of the formats these documents actually use.

    Note that every numeric form here is day-first. That is deliberate: the corpus
    commits to the Indian convention so that a month-first misreading is a detectable
    error rather than an ambiguity the verifier has to forgive.
    """
    styles = ["slash", "dash"]
    if tier != "easy":
        styles += ["named", "dot"]
    if tier == "hard":
        styles += ["native_slash", "native_named"]
    style = rng.choice(styles)

    dd, mm, yyyy = f"{value.day:02d}", f"{value.month:02d}", f"{value.year:04d}"
    month_name = lang.months[value.month - 1]

    if style == "slash":
        return f"{dd}/{mm}/{yyyy}"
    if style == "dash":
        return f"{dd}-{mm}-{yyyy}"
    if style == "dot":
        return f"{dd}.{mm}.{yyyy[2:]}"
    if style == "named":
        return f"{value.day} {month_name} {yyyy}"
    if style == "native_slash":
        return render_digits(f"{dd}/{mm}/{yyyy}", lang.script)
    day = render_digits(str(value.day), lang.script)
    year = render_digits(yyyy, lang.script)
    return f"{day} {month_name} {year}"


def _render_reference(rng: random.Random, lang: Language) -> str:
    """A document reference in a plausible Indian government/utility format."""
    state = rng.choice(("MH", "TN", "WB", "DL", "KA", "UP"))
    dept = rng.choice(("EB", "LIC", "MUN", "REV", "GST"))
    serial = rng.randint(10_000, 99_999)
    style = rng.choice(("slash", "dash", "mixed"))
    if style == "slash":
        return f"{state}/{2026}/{serial}"
    if style == "dash":
        return f"{state}-{dept}-{2026}-{serial}"
    return f"{state}/{dept}-{serial}"


# ------------------------------------------------------------------------ generator


def generate(
    idx: int, seed: int = 0, lang: LanguageCode | None = None, tier: Tier | None = None
) -> Document:
    """Generate document number `idx` under master `seed`.

    Each task gets its own RNG seeded from (seed, idx) so that `generate(7, ...)` is
    identical whether or not tasks 0..6 were generated first. That independence is
    what lets a run resume, shard, or sample a subset without perturbing the corpus.
    """
    # A large odd multiplier keeps consecutive idx values far apart in the seed space.
    # Deliberately arithmetic rather than hash()-based: hash() is salted per process
    # unless PYTHONHASHSEED is pinned, which would silently break reproducibility.
    rng = random.Random(seed * 1_000_003 + idx)

    lang_code = lang if lang is not None else rng.choice(tuple(LANGUAGES))
    language = LANGUAGES[lang_code]
    chosen_tier: Tier = tier if tier is not None else rng.choice(TIERS)

    name_native, name_roman = _pick_name(rng, language)
    amount = rng.choice(
        (
            rng.randrange(500, 99_999),
            rng.randrange(1, 40) * 10_000,
            rng.randrange(1, 20) * 100_000,
        )
    )
    # Anchor the calendar to a fixed epoch rather than today's date: reading the clock
    # would make the corpus non-reproducible, which is the one thing it must not be.
    issue = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 300))
    due = issue + timedelta(days=rng.randrange(10, 60))
    reference = _render_reference(rng, language)

    record = Record(
        name_native=name_native,
        name_roman=name_roman,
        amount_inr=amount,
        due_date=due,
        reference=reference,
    )

    lines, features = _compose(rng, language, chosen_tier, record, issue)
    return Document(
        idx=idx,
        lang=lang_code,
        tier=chosen_tier,
        text="\n".join(lines),
        record=record,
        issue_date=issue,
        features=features,
    )


def render_variant(doc: Document, variant: int) -> Document:
    """Re-render `doc` with a different surface form but identical ground truth.

    The record -- name, amount, due date, reference -- is held fixed while the digit
    system, date format, amount wording, label choice and line order are all redrawn.
    The correct answer is therefore exactly the same for every variant.

    This exists to test a property the reward silently depends on: that the verifier
    scores the *content* and not the *presentation*. Helff et al. (arXiv:2604.15149)
    show that RLVR verifiers checking only extensional correctness admit false
    positives, and propose isomorphic perturbation testing -- evaluating invariance
    under logically equivalent transformations -- as the check that separates genuine
    competence from surface exploitation. Variants are the isomorphic transformation
    for this task: a verifier whose verdict moves across them is grading formatting,
    and a model whose accuracy moves across them is reading formatting.
    """
    language = LANGUAGES[doc.lang]
    # Offset far from the generation seed space so a variant's surface never coincides
    # with one some other document already drew.
    rng = random.Random(0x5EED_0000 + doc.idx * 977 + variant)
    lines, features = _compose(rng, language, doc.tier, doc.record, doc.issue_date)
    features["variant"] = str(variant)
    return Document(
        idx=doc.idx,
        lang=doc.lang,
        tier=doc.tier,
        text="\n".join(lines),
        record=doc.record,
        issue_date=doc.issue_date,
        features=features,
    )


def _compose(
    rng: random.Random, lang: Language, tier: Tier, record: Record, issue: date
) -> tuple[list[str], dict[str, str]]:
    """Lay out the document body and its distractor lines.

    Every one of the four target fields gets exactly one competing candidate of the
    same kind, at every tier, and each competitor is positioned so that the obvious
    shortcut for that field is a coin flip:

    * amount    -- arrears drawn log-uniformly around the target, so it is larger
                   half the time and "pick the largest number" is 50/50.
    * due_date  -- the rival date falls after the due date half the time, so neither
                   "pick the latest" nor "pick the earliest" is informative.
    * name      -- a second person, and the addressee is behind a label like everyone
                   else, so the reader must use the label rather than the position.
    * reference -- a second reference in the same format family, so format-sniffing
                   narrows the field to two and no further.

    This replaced an earlier design in which difficulty came from the *number* of
    distractors, easy documents having none. That was measurably unsound: with no
    rival candidate a field is free, and shortcut policies scored 0.72 field accuracy
    and 0.33 exact match while reading nothing. Tier now controls surface format
    difficulty only -- digit system, date format, amount wording -- and the
    discrimination burden is constant across tiers.
    """
    label = lang.labels
    honorific = rng.choice(lang.honorifics)
    features = {"tier": tier, "lang": lang.code}

    # Log-uniform around 1.0: median multiplier is exactly 1, so P(arrears > target)
    # is 0.5 by construction rather than by a guessed range.
    arrears = max(1, int(record.amount_inr * math.exp(rng.uniform(-0.9, 0.9))))
    features["arrears_exceeds_target"] = str(arrears > record.amount_inr)

    # Half the documents put the rival date after the due date, half before.
    if rng.random() < 0.5:
        rival_date = record.due_date + timedelta(days=rng.randrange(30, 120))
        rival_date_label = label["next_cycle"][0]
        features["rival_date_after"] = "True"
    else:
        rival_date = issue
        rival_date_label = label["issue_date"][0]
        features["rival_date_after"] = "False"

    officer_native, _ = _pick_name(rng, lang)

    lines = [
        f"{label['addressee'][0]}: {honorific} {record.name_native}",
        f"{label['payable'][0]}: {_render_amount(rng, record.amount_inr, lang, tier)}",
        f"{label['due_date'][0]}: {_render_date(rng, record.due_date, lang, tier)}",
        f"{label['reference'][0]}: {record.reference}",
        f"{label['arrears'][0]}: {_render_amount(rng, arrears, lang, tier)}",
        f"{rival_date_label}: {_render_date(rng, rival_date, lang, tier)}",
        f"{label['officer'][0]}: {officer_native}",
        f"{label['helpline'][0]}: {_render_reference(rng, lang)}",
    ]

    if tier == "hard":
        # Extra noise that competes with nothing, to test robustness to clutter rather
        # than to add discrimination difficulty.
        late_fee = rng.randrange(50, 900)
        lines.append(f"{label['late_fee'][0]}: {_render_amount(rng, late_fee, lang, tier)}")

    # Shuffling defeats positional shortcuts: without it, "the reference is always the
    # fourth line" would be a perfect extraction rule for that field. The addressee is
    # shuffled along with everything else -- pinning it to line 0 made "take line 0" a
    # perfect name extractor, which is how the name field came to be worth nothing.
    rng.shuffle(lines)
    return lines, features


def generate_many(
    count: int, seed: int = 0, lang: LanguageCode | None = None, tier: Tier | None = None
) -> list[Document]:
    return [generate(i, seed=seed, lang=lang, tier=tier) for i in range(count)]
