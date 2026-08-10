"""The corpus must not be solvable by rules that never read the document.

This is the most important test file in the repository, and the reason is worth
stating plainly. If a shortcut scores well, then every accuracy number this
environment reports is partly measuring the shortcut rather than the task, and any
model trained against it learns the regularity instead of the skill. That failure is
silent: nothing crashes, the numbers just mean something other than what they claim.

The ceilings below are therefore load-bearing, not decorative. They were set after
measuring an earlier generator that failed badly -- shortcut policies reached 0.717
field accuracy and 0.327 exact match because the addressee was always the first line
and the easy and medium tiers had no competing candidates at all. The generator was
redesigned so that every field has exactly one rival at every tier, positioned to make
each shortcut a coin flip.

With four fields and one rival each, a guesser should land near 0.5 per field and
0.5**4 = 0.0625 exact. The ceilings allow headroom over that for sampling noise while
still failing loudly if a regularity is reintroduced.
"""

from __future__ import annotations

import pytest

from indic_extraction_v1.corpus import FIELDS, TIERS, generate_many
from indic_extraction_v1.heuristics import POLICIES
from indic_extraction_v1.verify import verify_document

# Fixed corpus for every assertion here: a seeded sample large enough that the
# measurement is stable, and identical on every machine and every run.
DOCS = generate_many(600, seed=1)

MAX_FIELD_ACCURACY = 0.55
MAX_EXACT_MATCH = 0.12
MAX_PER_FIELD_ACCURACY = 0.62


def _score(policy) -> tuple[float, float, dict[str, float]]:
    total = 0
    exact = 0
    per: dict[str, int] = dict.fromkeys(FIELDS, 0)
    for doc in DOCS:
        verdict = verify_document(policy(doc), doc)
        total += verdict.n_correct
        exact += verdict.all_correct
        for field in verdict.fields:
            if field.correct:
                per[field.name] += 1
    n = len(DOCS)
    return total / (len(FIELDS) * n), exact / n, {k: v / n for k, v in per.items()}


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_shortcut_policy_stays_near_chance(name: str) -> None:
    field_accuracy, exact, per_field = _score(POLICIES[name])
    assert field_accuracy <= MAX_FIELD_ACCURACY, (name, field_accuracy)
    assert exact <= MAX_EXACT_MATCH, (name, exact)
    for field, value in per_field.items():
        assert value <= MAX_PER_FIELD_ACCURACY, (name, field, value)


def test_no_field_is_free() -> None:
    """No single field may be extractable by any shortcut at high accuracy.

    Regression: the addressee was pinned to line 0 while every other line was shuffled,
    so 'take the first line' extracted the name with accuracy 1.000. Partial credit
    then paid one quarter of full marks for reading nothing.
    """
    for name, policy in POLICIES.items():
        _, _, per_field = _score(policy)
        for field, value in per_field.items():
            assert value <= MAX_PER_FIELD_ACCURACY, (
                f"{name} extracts {field} at {value:.3f} without reading"
            )


def test_empty_but_valid_json_scores_zero() -> None:
    """A schema-shaped but contentless reply must earn nothing.

    This is the canonical reward hack for extraction tasks: emit the right keys with
    empty values and collect whatever the format term pays. It must be worth zero on
    the correctness axis, which is the only axis that pays positively.
    """
    field_accuracy, exact, _ = _score(POLICIES["empty"])
    assert field_accuracy == 0.0
    assert exact == 0.0


@pytest.mark.parametrize("tier", TIERS)
def test_shortcuts_stay_near_chance_within_every_tier(tier: str) -> None:
    """Aggregate safety can hide a tier that is individually trivial.

    The earlier generator passed nothing like this: on the easy tier there were no
    competing candidates at all, so shortcuts scored perfectly there while the mixed
    average still looked tolerable.
    """
    docs = generate_many(300, seed=41, tier=tier)  # type: ignore[arg-type]
    for name, policy in POLICIES.items():
        correct = sum(verify_document(policy(d), d).n_correct for d in docs)
        accuracy = correct / (len(FIELDS) * len(docs))
        assert accuracy <= MAX_FIELD_ACCURACY, (tier, name, accuracy)
