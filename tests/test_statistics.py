"""Interval estimation and paired comparison.

These tests exist because the project previously reported every accuracy as a bare point
estimate from a single seed. The assertions below pin the two properties that made that
unsound: an interval must stay inside [0, 1] at any sample size, and a single seed must
never claim to have measured variance.
"""

from __future__ import annotations

import math

import pytest

from research.statistics_ import (
    Interval,
    mcnemar,
    seed_variance,
    wilson,
    wilson_from_scores,
)


def normal_approximation(successes: int, n: int, z: float = 1.959963984540054):
    """The method we deliberately do not use, kept to demonstrate why."""
    p = successes / n
    half = z * math.sqrt(p * (1 - p) / n)
    return p - half, p + half


def test_wilson_matches_known_values() -> None:
    """Pinned against the project's own committed results."""
    assert str(wilson(184, 200)) == "0.920 [0.874, 0.950] (n=200)"
    assert str(wilson(124, 200)) == "0.620 [0.551, 0.684] (n=200)"
    assert str(wilson(38, 40)) == "0.950 [0.835, 0.986] (n=40)"


def test_normal_approximation_would_report_impossible_accuracy() -> None:
    """The concrete reason this project uses Wilson.

    At 38 of 40 correct -- the real gemini-3.6-flash sample -- the normal approximation
    puts the upper bound at 1.018, an accuracy above 100 per cent. That is the
    small-sample failure arXiv:2503.01747 describes, reproduced on our own data. Wilson
    stays inside [0, 1] because it inverts the score test instead of assuming the
    estimate is normally distributed.
    """
    _, naive_high = normal_approximation(38, 40)
    assert naive_high > 1.0, "the demonstration itself has broken"
    assert wilson(38, 40).high <= 1.0


@pytest.mark.parametrize(("successes", "n"), [(0, 1), (1, 1), (0, 40), (40, 40), (1, 3), (7, 9)])
def test_intervals_stay_inside_the_unit_range(successes: int, n: int) -> None:
    """Including the degenerate all-right and all-wrong cases.

    A normal-approximation interval at 0/n or n/n has zero width, which claims perfect
    certainty from a finite sample. Wilson keeps a sensible width at both extremes.
    """
    interval = wilson(successes, n)
    assert 0.0 <= interval.low <= interval.point <= interval.high <= 1.0
    if 0 < successes < n:
        assert interval.width > 0.0


def test_perfect_score_still_has_an_interval() -> None:
    """40/40 is not evidence of 100% accuracy."""
    interval = wilson(40, 40)
    assert interval.point == 1.0
    assert interval.low < 1.0
    assert interval.width > 0.05


def test_interval_narrows_as_n_grows() -> None:
    widths = [wilson(round(0.9 * n), n).width for n in (20, 50, 200, 1000)]
    assert widths == sorted(widths, reverse=True)


@pytest.mark.parametrize(("successes", "n"), [(-1, 10), (11, 10), (5, 0), (5, -3)])
def test_wilson_rejects_impossible_inputs(successes: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson(successes, n)


def test_wilson_from_scores_treats_field_accuracy_as_field_decisions() -> None:
    """Field accuracy is a mean of quarters, i.e. a binomial over 4n field decisions."""
    scores = [1.0, 0.75, 0.5, 1.0]
    interval = wilson_from_scores(scores)
    assert interval.n == 4
    assert interval.point == pytest.approx(sum(scores) / 4, abs=0.13)


def test_wilson_from_scores_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        wilson_from_scores([0.5, 1.5])
    with pytest.raises(ValueError):
        wilson_from_scores([])


def test_mcnemar_uses_only_discordant_pairs() -> None:
    """Documents both models get right carry no comparative information.

    This is the efficiency the pairing buys and the reason two marginal intervals are the
    wrong comparison: they discard which documents each model got right.
    """
    a = [True, True, False, True, False]
    b = [True, False, False, False, True]
    result = mcnemar("a", a, "b", b)
    assert result.a_only == 2
    assert result.b_only == 1
    assert result.both == 1
    assert result.neither == 1
    assert result.discordant == 3
    assert result.n == 5


def test_mcnemar_detects_a_real_difference() -> None:
    a = [True] * 60 + [False] * 40
    b = [True] * 20 + [False] * 80
    assert mcnemar("a", a, "b", b).significant


def test_mcnemar_does_not_invent_a_difference() -> None:
    """Identical outcomes have zero discordant pairs and cannot be significant."""
    outcomes = [True, False] * 50
    result = mcnemar("a", outcomes, "b", list(outcomes))
    assert result.discordant == 0
    assert result.p_value == 1.0
    assert not result.significant


def test_paired_test_can_separate_models_whose_intervals_overlap() -> None:
    """The concrete reason to prefer pairing over comparing marginal intervals.

    Two models can have overlapping marginal intervals while every document they
    disagree on falls the same way -- which is unambiguous evidence, and is invisible if
    you only look at the two intervals.
    """
    a = [True] * 55 + [False] * 45
    b = [True] * 45 + [False] * 55
    # Constructed so the discordant pairs all favour A.
    a_outcomes = [True] * 10 + [True] * 45 + [False] * 45
    b_outcomes = [False] * 10 + [True] * 45 + [False] * 45

    assert wilson(sum(a), len(a)).overlaps(wilson(sum(b), len(b)))
    assert mcnemar("a", a_outcomes, "b", b_outcomes).significant


def test_mcnemar_requires_aligned_samples() -> None:
    with pytest.raises(ValueError):
        mcnemar("a", [True, False], "b", [True])
    with pytest.raises(ValueError):
        mcnemar("a", [], "b", [])


def test_single_seed_reports_unmeasured_variance_not_zero() -> None:
    """One observation has no spread; claiming 0.0 would assert the opposite.

    Every result in this project came from seed 1 alone. Reporting a standard deviation
    of zero there would have implied the corpus draw contributes nothing, when in fact a
    single seed measures nothing about corpus variance at all.
    """
    variance = seed_variance("exact_match", {1: 0.92})
    assert variance.sample_stdev is None
    assert "not measured" in str(variance)


def test_seed_variance_over_several_seeds() -> None:
    variance = seed_variance("exact_match", {1: 0.92, 2: 0.90, 3: 0.94, 4: 0.91, 5: 0.93})
    assert variance.mean == pytest.approx(0.92)
    assert variance.spread == pytest.approx(0.04)
    assert variance.sample_stdev == pytest.approx(0.0158, abs=1e-4)


def test_seed_variance_rejects_empty() -> None:
    with pytest.raises(ValueError):
        seed_variance("exact_match", {})


def test_interval_str_always_carries_n() -> None:
    """A proportion printed without its sample size is not a measurement."""
    rendered = str(Interval(point=0.5, low=0.4, high=0.6, n=123))
    assert "n=123" in rendered
    assert "[0.400, 0.600]" in rendered


def _report(model: str, outcomes: list[bool]):
    """Minimal ModelReport carrying only what the reporting path reads."""
    from research.evaluate import ModelReport

    n = len(outcomes)
    return ModelReport(
        model=model,
        n_requested=n,
        n_completed=n,
        n_errors=0,
        stopped_early=False,
        stop_reason=None,
        field_accuracy=sum(outcomes) / n if n else None,
        exact_match=sum(outcomes) / n if n else None,
        mean_reward=0.9,
        per_field={},
        per_tier={},
        per_lang={},
        day_month_transposed=0.0,
        answered_in_roman=0.0,
        schema_clean=1.0,
        mean_output_tokens=100.0,
        mean_total_tokens=400.0,
        mean_hidden_reasoning_tokens=0.0,
        truncated_rate=0.0,
        exact_outcomes=outcomes,
        failure_reasons={},
    )


def test_headline_table_never_prints_a_bare_proportion(capsys) -> None:
    """Exact match must always appear with its interval and its n.

    The whole project previously reported bare point estimates. This asserts the
    reporting path itself cannot regress to that, rather than trusting the caller to
    remember.
    """
    from research.evaluate import print_report

    print_report([_report("model-a", [True] * 92 + [False] * 8)])
    out = capsys.readouterr().out
    assert "0.920" in out
    assert "[" in out and "]" in out, "no interval in the headline table"
    assert "n=100" in out, "no sample size in the headline table"


def test_report_with_no_completions_says_not_measured(capsys) -> None:
    """An aborted run must not be presentable as a measurement of zero."""
    from research.evaluate import print_report

    print_report([_report("model-a", [])])
    out = capsys.readouterr().out
    assert "not measured" in out


def test_paired_comparison_refuses_misaligned_runs(capsys) -> None:
    """Two models evaluated on different numbers of documents are not paired.

    A run can stop at the budget partway through the second model. Silently truncating
    to the shorter vector would compare the models on different corpora and label it a
    paired test, which is worse than declining to compare.
    """
    from research.evaluate import print_report

    print_report([_report("a", [True] * 50), _report("b", [True] * 30)])
    out = capsys.readouterr().out
    assert "not comparable" in out
