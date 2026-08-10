"""Reward composition, independent of the verifiers runtime.

Kept separate from `taskset.py` for two reasons. It lets the reward be tested and
analysed offline on any platform -- `verifiers.v1` imports `fcntl` and cannot be
installed on Windows -- and it keeps a single definition of the reward, so the numbers
in the reward-hacking analysis are produced by exactly the code that trains a model
rather than by a re-implementation that might drift.

The design rule, stated once here and enforced by `tests/test_score.py`:

    Exactly one term can be positive, and it is the one that requires reading the
    document. Every other term is a penalty bounded in [0, 1] with a negative weight,
    so its best possible contribution is zero.

The consequence is that no amount of formatting discipline or brevity can substitute
for being right. A terse, perfectly-formed, contentless reply scores the penalty floor
and can never overtake a correct answer. This is not a stylistic preference; it is the
property that makes the reward hard to hack, and the asymmetry is the whole mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from indic_extraction_v1.corpus import FIELDS
from indic_extraction_v1.verify import Verdict

# Output-token allowance before the verbosity penalty starts.
#
# Set from measurement, not from a guess. The first value here was 160, chosen because a
# bare four-field JSON answer is well under that. Measuring 400 real rollouts showed the
# figure was wrong in a way that mattered: gpt-oss-120b answers correctly in about 197
# tokens, so 170 of 199 *correct* rollouts were being charged a small penalty. That is a
# tax on correct behaviour rather than a guard against padding -- pure noise in the
# reward, applied to exactly the answers that should score full marks.
#
# 256 sits above normal correct output for every model measured (llama-3.1-8b: 68
# tokens, gemini-3.6-flash: 76, gpt-oss-120b: 197) while still biting hard on genuine
# padding: the `correct_but_padded` attack spends roughly 670 output tokens and is
# charged accordingly. The penalty is a guardrail against degenerate verbosity, not a
# brevity incentive, and it should read zero for any well-behaved answer.
TOKEN_BUDGET = 256

# Ceiling used to normalise the verbosity penalty into [0, 1].
TOKEN_CEILING = 800

WEIGHT_FIELD_ACCURACY = 1.0
WEIGHT_FORMAT_VIOLATION = -0.25
WEIGHT_VERBOSITY = -0.10


@dataclass(frozen=True)
class RewardTerms:
    field_accuracy: float
    format_violation: float
    verbosity: float

    @property
    def total(self) -> float:
        return (
            WEIGHT_FIELD_ACCURACY * self.field_accuracy
            + WEIGHT_FORMAT_VIOLATION * self.format_violation
            + WEIGHT_VERBOSITY * self.verbosity
        )


def field_accuracy(verdict: Verdict) -> float:
    """Fraction of the four fields extracted correctly. The only positive term.

    Partial credit rather than all-or-nothing. That is only safe because no field can
    be guessed for free -- a property established by measurement in `heuristics.py` and
    pinned by `tests/test_corpus_is_not_shortcuttable.py`, not assumed. If a generator
    change ever made some field trivially guessable, partial credit would begin paying
    for it, which is why that test is a hard gate rather than a nicety.
    """
    return verdict.n_correct / len(FIELDS)


def format_violation(verdict: Verdict) -> float:
    """Penalty in [0, 1] for anything wrong with the response's shape.

    Three independent violations, each worth a third: no recoverable JSON, keys that do
    not match the schema exactly, and more than one candidate JSON object. The last is
    what stops a model from listing several guesses and letting the grader choose.
    """
    violations = 0
    if not verdict.parsed:
        violations += 1
    if verdict.extra_keys or verdict.missing_keys:
        violations += 1
    if verdict.n_candidate_objects > 1:
        violations += 1
    return min(1.0, violations / 3.0)


def verbosity(output_tokens: int) -> float:
    """Penalty in [0, 1] for output tokens spent beyond `TOKEN_BUDGET`.

    Measured in tokens the provider billed, never in characters.

    This environment accepts an answer in the document's own script *or* in Latin
    transliteration, and the correctness term cannot distinguish them. A
    character-denominated budget would nonetheless charge them at very different
    effective rates. Measured tokens-per-character relative to Latin, for identical
    content: Devanagari 1.16x and Tamil 1.66x on gpt-oss-120b, rising to 1.94x and
    4.52x on llama-3.1-8b-instant. A character budget is therefore up to four and a
    half times more generous to a Tamil answer than to the same answer romanised --
    a script preference expressed through the length term, invisible in the
    correctness term, and unsatisfiable except by switching scripts. Counting tokens
    removes the effect by construction. See `research/script_bias.py`.
    """
    if output_tokens <= TOKEN_BUDGET:
        return 0.0
    return min(1.0, (output_tokens - TOKEN_BUDGET) / (TOKEN_CEILING - TOKEN_BUDGET))


def reward_terms(verdict: Verdict, output_tokens: int) -> RewardTerms:
    return RewardTerms(
        field_accuracy=field_accuracy(verdict),
        format_violation=format_violation(verdict),
        verbosity=verbosity(output_tokens),
    )


def total_reward(verdict: Verdict, output_tokens: int) -> float:
    return reward_terms(verdict, output_tokens).total
