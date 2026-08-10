"""Exploit ceilings, pinned so a reward change cannot quietly reopen a hole.

Each assertion here corresponds to an exploit that was found by measurement against an
earlier or alternative reward design, and to the specific mitigation that closed it.
The numbers are not arbitrary thresholds: they are the measured attack scores, with
enough slack to absorb the sampling noise of a fixed 400-document corpus.

Entirely offline and deterministic -- no model, no network, no clock -- so these run in
the same suite as everything else and cost nothing.
"""

from __future__ import annotations

import statistics

import pytest

from indic_extraction_v1.corpus import generate_many
from research.ablations import ABLATIONS
from research.adversarial import POLICIES, estimate_tokens

DOCS = generate_many(400, seed=1)


def mean_score(policy_name: str, reward_name: str) -> float:
    policy = POLICIES[policy_name]
    reward = ABLATIONS[reward_name]
    return statistics.fmean(
        reward(reply, doc, estimate_tokens(reply))
        for reply, doc in ((policy(doc), doc) for doc in DOCS)
    )


def test_honest_extraction_scores_full_marks() -> None:
    assert mean_score("honest_oracle", "shipped") == pytest.approx(1.0)


def test_contentless_reply_earns_nothing() -> None:
    """Exploit 1: a positive reward for well-formed output creates a free floor.

    Under a design that pays 0.3 for schema conformance, a reply containing the right
    keys and no information scores 0.300 -- 23% of what full extraction earns -- while
    reading nothing. The shipped reward makes every non-correctness term a penalty
    bounded above by zero, which removes the floor entirely.
    """
    assert mean_score("empty_schema", "shipped") == 0.0
    assert mean_score("empty_schema", "format_bonus") > 0.29


def test_shotgun_objects_is_contained() -> None:
    """Exploit 2: a grader that searches for the best-matching JSON object.

    Emitting one object per combination of candidates scores 0.732 -- 73% of honest
    extraction -- against a lenient extractor, with no discrimination performed at all.
    Grading the FIRST object only, and penalising multiplicity, cuts it to 0.243.
    """
    contained = mean_score("shotgun_objects", "shipped")
    exploited = mean_score("shotgun_objects", "lenient_extractor")
    assert exploited > 0.70
    assert contained < 0.30
    assert exploited - contained > 0.45


def test_shotgun_lists_is_contained() -> None:
    """Exploit 3: crediting a field when the truth appears anywhere in a collection.

    The most severe of the three. Listing every candidate per field scores 0.846 --
    85% of honest extraction -- against a membership-based grader, and the task's whole
    difficulty is discriminating between those candidates. Requiring a scalar value
    drops it to exactly zero.
    """
    assert mean_score("shotgun_lists", "shipped") == 0.0
    assert mean_score("shotgun_lists", "membership_credit") > 0.80


def test_no_attack_beats_honest_extraction_under_shipped_reward() -> None:
    """The property that has to hold no matter what else changes."""
    honest = mean_score("honest_oracle", "shipped")
    for name in POLICIES:
        if name in ("honest_oracle", "correct_but_padded"):
            continue
        assert mean_score(name, "shipped") < honest, name


def test_padding_never_overturns_correctness() -> None:
    """A correct answer buried in filler must still beat every attack.

    Calibration check on the verbosity penalty: it should discourage padding without
    ever making a guess preferable to a correct answer.
    """
    padded = mean_score("correct_but_padded", "shipped")
    worst_attack = max(
        mean_score(name, "shipped")
        for name in POLICIES
        if name not in ("honest_oracle", "correct_but_padded")
    )
    assert padded > worst_attack
    assert padded < mean_score("honest_oracle", "shipped")


@pytest.mark.parametrize("reward_name", sorted(ABLATIONS))
def test_every_reward_variant_is_bounded(reward_name: str) -> None:
    """Sanity: no reward design may return unbounded or NaN scores."""
    for policy_name in POLICIES:
        value = mean_score(policy_name, reward_name)
        assert value == value  # not NaN
        assert -2.0 <= value <= 2.0, (reward_name, policy_name, value)
