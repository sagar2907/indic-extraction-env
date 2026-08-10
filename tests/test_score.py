"""The reward's structural guarantees, tested as properties rather than examples.

These are the tests that would fail if someone "improved" the reward by adding a bonus
for well-formed output. That change looks harmless and is the single most common way
an extraction reward becomes hackable, so the invariant is pinned explicitly.
"""

from __future__ import annotations

import json

from indic_extraction_v1.corpus import generate, generate_many
from indic_extraction_v1.heuristics import POLICIES
from indic_extraction_v1.score import (
    TOKEN_BUDGET,
    TOKEN_CEILING,
    RewardTerms,
    reward_terms,
    total_reward,
    verbosity,
)
from indic_extraction_v1.verify import verify_document


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


def test_perfect_answer_scores_one() -> None:
    doc = _doc()
    assert total_reward(verify_document(_gold(doc), doc), 60) == 1.0


def test_only_correctness_can_be_positive() -> None:
    """No combination of formatting and brevity can produce positive reward alone.

    This is the property that makes the reward hard to hack. If format or brevity paid
    positively, the reward-maximising policy would be a terse well-formed reply with no
    content -- cheap to produce and requiring no reading at all.
    """
    for output_tokens in (0, 10, TOKEN_BUDGET, TOKEN_CEILING, 5000):
        for reply in ("{}", "", "not json", '{"name":"","amount_inr":0}'):
            doc = _doc()
            terms = reward_terms(verify_document(reply, doc), output_tokens)
            assert terms.field_accuracy == 0.0
            assert terms.total <= 0.0, (reply, output_tokens, terms)


def test_empty_json_is_punished_not_rewarded() -> None:
    """The canonical degenerate answer must sit strictly below zero.

    The exact figure is -0.25/3. `{}` trips exactly one of the three format violations
    -- its keys do not match the schema -- while remaining valid JSON containing a
    single object. An earlier docstring claimed it scored the full -0.25; that number
    was reasoned about rather than measured, and this assertion is what caught it.
    """
    doc = _doc()
    terms = reward_terms(verify_document("{}", doc), 20)
    assert terms.total < 0.0
    assert abs(terms.total - (-0.25 / 3)) < 1e-12
    assert terms.format_violation == 1 / 3


def test_correct_answer_always_beats_every_degenerate_one() -> None:
    """Even a verbose, badly formatted correct answer outranks a clean empty one."""
    doc = _doc()
    sloppy_but_right = "Here you go!\n" + _gold(doc, extra="noise")
    right = total_reward(verify_document(sloppy_but_right, doc), TOKEN_CEILING)
    clean_but_empty = total_reward(verify_document("{}", doc), 10)
    assert right > clean_but_empty


def test_penalties_are_bounded() -> None:
    for doc in generate_many(50, seed=3):
        for reply in ("{}", _gold(doc), "garbage", _gold(doc) + _gold(doc)):
            terms = reward_terms(verify_document(reply, doc), 10_000)
            assert 0.0 <= terms.format_violation <= 1.0
            assert 0.0 <= terms.verbosity <= 1.0
            assert 0.0 <= terms.field_accuracy <= 1.0


def test_verbosity_is_free_within_budget_and_saturates() -> None:
    assert verbosity(0) == 0.0
    assert verbosity(TOKEN_BUDGET) == 0.0
    assert verbosity(TOKEN_BUDGET + 1) > 0.0
    assert verbosity(TOKEN_CEILING) == 1.0
    assert verbosity(TOKEN_CEILING * 10) == 1.0


def test_multiple_candidate_objects_are_penalised() -> None:
    """Emitting several guesses must cost, not pay.

    Without this the optimal policy under a lenient extractor is to enumerate every
    plausible reading of the document and let the grader find the right one.
    """
    doc = _doc()
    decoy = json.dumps({"name": "x", "amount_inr": 1, "due_date": "2020-01-01", "reference": "y"})
    shotgun = reward_terms(verify_document(decoy + "\n" + _gold(doc), doc), 80)
    assert shotgun.format_violation > 0.0
    assert shotgun.total < total_reward(verify_document(_gold(doc), doc), 80)


def test_no_shortcut_policy_earns_positive_reward_on_average() -> None:
    """Shortcut policies must not be a viable strategy under the real reward.

    They can score above zero on individual documents by luck. What must not happen is
    a policy that never reads the document earning a materially positive average, which
    would make "guess plausibly" competitive with "extract correctly".
    """
    docs = generate_many(400, seed=1)
    best_correct = sum(total_reward(verify_document(_gold(d), d), 70) for d in docs) / len(docs)
    for name, policy in POLICIES.items():
        mean = sum(total_reward(verify_document(policy(d), d), 70) for d in docs) / len(docs)
        assert mean < 0.55, (name, mean)
        assert mean < best_correct * 0.6, (name, mean, best_correct)


def test_reward_terms_total_matches_weights() -> None:
    terms = RewardTerms(field_accuracy=1.0, format_violation=1.0, verbosity=1.0)
    assert abs(terms.total - (1.0 - 0.25 - 0.10)) < 1e-12


def test_verbosity_budget_does_not_tax_correct_answers() -> None:
    """The length penalty must read zero for normal correct output.

    Regression on a calibration error. TOKEN_BUDGET was originally 160, below the ~197
    output tokens gpt-oss-120b actually spends on a correct answer, so 170 of 199
    correct rollouts were charged a small penalty. A term that fires on the overwhelming
    majority of correct answers adds noise to the reward instead of shaping behaviour.
    The budget now sits above measured correct-answer length for every model evaluated.
    """
    measured_correct_answer_lengths = {
        "llama-3.1-8b-instant": 68,
        "gemini-3.6-flash": 76,
        "openai/gpt-oss-120b": 197,
    }
    for model, tokens in measured_correct_answer_lengths.items():
        assert verbosity(tokens) == 0.0, model


def test_verbosity_still_bites_on_real_padding() -> None:
    """Raising the budget must not disarm the penalty against degenerate verbosity."""
    doc = _doc()
    from research.adversarial import correct_but_padded, estimate_tokens

    padded = correct_but_padded(doc)
    assert verbosity(estimate_tokens(padded)) > 0.5
