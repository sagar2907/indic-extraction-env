"""The verifier must grade content, not presentation.

Motivated by Helff et al., *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking*
(arXiv:2604.15149), which shows that RLVR verifiers checking only extensional
correctness admit false positives, and proposes isomorphic perturbation testing --
evaluating invariance under logically equivalent task transformations -- as the check
that distinguishes genuine competence from surface exploitation.

`corpus.render_variant` is the isomorphic transformation for this task: identical
ground truth, completely redrawn surface. Two properties must hold across variants.

* The verifier's verdict on the gold answer is invariant. If it is not, the reward
  depends on how the document happened to be typeset, and any accuracy number the
  environment reports is partly a measurement of formatting luck.
* A shortcut policy does not become reliable on some particular surface. Aggregate
  shortcut resistance can hide a rendering on which a trivial rule works perfectly.
"""

from __future__ import annotations

import json

import pytest

from indic_extraction_v1.corpus import FIELDS, generate_many, render_variant
from indic_extraction_v1.heuristics import POLICIES
from indic_extraction_v1.verify import verify_document

VARIANTS = 4
DOCS = generate_many(120, seed=1)


def _gold(doc) -> str:
    return json.dumps(
        {
            "name": doc.record.name_native,
            "amount_inr": doc.record.amount_inr,
            "due_date": doc.record.due_date.isoformat(),
            "reference": doc.record.reference,
        },
        ensure_ascii=False,
    )


def test_variants_preserve_ground_truth_exactly() -> None:
    """The transformation must be isomorphic: same answer, different presentation."""
    for doc in DOCS[:40]:
        for variant in range(VARIANTS):
            other = render_variant(doc, variant)
            assert other.record == doc.record
            assert other.lang == doc.lang
            assert other.tier == doc.tier


def test_variants_actually_change_the_surface() -> None:
    """A test of invariance is worthless if the transformation is a no-op."""
    changed = 0
    for doc in DOCS[:40]:
        surfaces = {render_variant(doc, v).text for v in range(VARIANTS)}
        if len(surfaces) == VARIANTS:
            changed += 1
    assert changed >= 38, f"only {changed}/40 documents produced distinct variants"


def test_gold_answer_verifies_identically_across_variants() -> None:
    """The core invariance property: the verdict must not move with the typesetting.

    A verifier that scores the gold answer differently depending on whether the amount
    was written '₹58,348' or '₹५८,३४८' is grading presentation. Every accuracy figure
    this environment produces would then be contaminated by formatting.
    """
    for doc in DOCS:
        answer = _gold(doc)
        baseline = verify_document(answer, doc)
        assert baseline.all_correct, doc.idx
        for variant in range(VARIANTS):
            verdict = verify_document(answer, render_variant(doc, variant))
            assert verdict.all_correct, (doc.idx, variant)
            assert verdict.n_correct == baseline.n_correct


@pytest.mark.parametrize("policy_name", sorted(POLICIES))
def test_no_variant_makes_a_shortcut_reliable(policy_name: str) -> None:
    """Shortcut resistance must hold per surface, not only on average.

    Aggregate safety can conceal one rendering on which a trivial rule is near-perfect.
    Because a training run would sample every surface, a single exploitable rendering
    is enough for a policy to find and exploit it.
    """
    policy = POLICIES[policy_name]
    for variant in range(VARIANTS):
        correct = 0
        for doc in DOCS:
            other = render_variant(doc, variant)
            correct += verify_document(policy(other), other).n_correct
        accuracy = correct / (len(FIELDS) * len(DOCS))
        assert accuracy <= 0.60, (policy_name, variant, accuracy)
