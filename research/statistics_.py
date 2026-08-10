"""Interval estimation and model comparison for the evaluation results.

Every accuracy this project reports is a proportion estimated from a finite sample, and
a proportion without an interval is not a measurement -- it is a number that looks like
one. This module supplies the intervals and the comparison test, and `evaluate.py` is
written so that no reporting path can print a proportion without them.

Two methodological choices, both taken from the literature rather than from habit.

**Wilson score intervals, never the normal approximation.** *Position: Don't Use the CLT
in LLM Evals With Fewer Than a Few Hundred Datapoints* (arXiv:2503.01747) shows that
CLT-based error bars substantially understate uncertainty on the small, specialised
benchmarks typical of LLM evaluation. This project reproduces that failure on its own
data: at 38/40 correct, the normal approximation returns an upper bound of **1.018** --
an accuracy above 100 per cent. Wilson returns [0.835, 0.986] and stays inside [0, 1] by
construction, because it inverts the score test rather than assuming normality of the
estimate.

**Paired comparison, never two independent intervals.** Every model in a run sees the
identical document set, so the samples are paired and the right question is per-document
agreement, not whether two marginal intervals overlap. Comparing independent intervals
throws away that pairing and is strictly less powerful -- two intervals can overlap while
the paired difference is unambiguous. McNemar's test uses only the documents where the
models disagree, which is exactly the information the pairing provides.

A caveat this module cannot fix, recorded so nobody reads more into these numbers than
they carry: these intervals describe *sampling* error over documents. *Hidden Measurement
Error in LLM Pipelines* (arXiv:2604.11581) finds naive standard errors run 40-60% below
error-corrected ones once prompt phrasing, decoding temperature and other pipeline
choices are accounted for. Corpus-seed variance is reported separately by
`seed_variance` for that reason; the two together are still a floor, not a full accounting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A proportion with its sample size and a confidence interval."""

    point: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"

    @property
    def width(self) -> float:
        return self.high - self.low

    def overlaps(self, other: Interval) -> bool:
        """Whether two marginal intervals overlap.

        Deliberately *not* the way models are compared here -- see `mcnemar`. Overlap is
        a weak and conservative test that ignores pairing; it is exposed only so a report
        can say plainly when two intervals do overlap.
        """
        return not (self.low > other.high or other.low > self.high)


def wilson(successes: int, n: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because it stays within [0, 1] at any sample
    size and keeps its nominal coverage when the proportion is near 0 or 1 -- the regime
    every accuracy figure in this project lives in.

    >>> str(wilson(38, 40))
    '0.950 [0.835, 0.986] (n=40)'
    """
    if n <= 0:
        raise ValueError("n must be positive; an interval over zero samples is undefined")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")

    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(point=p, low=max(0.0, centre - half), high=min(1.0, centre + half), n=n)


def wilson_from_scores(scores: Sequence[float], z: float = Z_95) -> Interval:
    """Interval for a mean of per-document scores in [0, 1].

    Field accuracy is a mean of quarters rather than a raw success count, so it is
    treated as a binomial over `4n` field decisions -- which is what it actually is.
    Values outside [0, 1] are a programming error and are rejected rather than clamped.
    """
    if not scores:
        raise ValueError("cannot form an interval over an empty sample")
    if any(not 0.0 <= s <= 1.0 for s in scores):
        raise ValueError("scores must lie in [0, 1]")
    total = sum(scores)
    # Round rather than truncate: 0.25 * 4 can land at 0.9999999999999999.
    return wilson(round(total), len(scores), z=z)


@dataclass(frozen=True)
class PairedComparison:
    """Result of comparing two models on the same documents."""

    model_a: str
    model_b: str
    a_only: int
    """Documents A got right and B got wrong."""

    b_only: int
    both: int
    neither: int
    p_value: float

    @property
    def n(self) -> int:
        return self.a_only + self.b_only + self.both + self.neither

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (
            f"{self.model_a} vs {self.model_b}: "
            f"{self.a_only} / {self.b_only} discordant of {self.n}, "
            f"p = {self.p_value:.4g} ({verdict} at 0.05)"
        )


def _binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test against p = 0.5.

    Used instead of the chi-square approximation because the discordant count can be
    small, and the whole reason this module exists is that approximations misbehave in
    exactly that regime.
    """
    if n == 0:
        return 1.0
    coefficients = [math.comb(n, i) for i in range(n + 1)]
    total = float(2**n)
    observed = coefficients[k]
    # Sum the probability of every outcome no more likely than the one observed.
    tail = sum(c for c in coefficients if c <= observed)
    return min(1.0, tail / total)


def mcnemar(
    model_a: str, outcomes_a: Sequence[bool], model_b: str, outcomes_b: Sequence[bool]
) -> PairedComparison:
    """Exact McNemar test on paired per-document outcomes.

    The models must have been evaluated on the same documents in the same order; the
    caller is responsible for that alignment and a length mismatch is an error rather
    than something to silently truncate.

    Only discordant pairs carry information: documents both models got right, or both
    got wrong, say nothing about which is better. That is precisely the efficiency the
    pairing buys, and it is thrown away by comparing two marginal intervals.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError(
            f"paired comparison needs equal-length outcomes, "
            f"got {len(outcomes_a)} and {len(outcomes_b)}"
        )
    if not outcomes_a:
        raise ValueError("cannot compare over an empty sample")

    a_only = sum(1 for a, b in zip(outcomes_a, outcomes_b, strict=True) if a and not b)
    b_only = sum(1 for a, b in zip(outcomes_a, outcomes_b, strict=True) if b and not a)
    both = sum(1 for a, b in zip(outcomes_a, outcomes_b, strict=True) if a and b)
    neither = sum(1 for a, b in zip(outcomes_a, outcomes_b, strict=True) if not a and not b)

    return PairedComparison(
        model_a=model_a,
        model_b=model_b,
        a_only=a_only,
        b_only=b_only,
        both=both,
        neither=neither,
        p_value=_binomial_two_sided_p(a_only, a_only + b_only),
    )


@dataclass(frozen=True)
class SeedVariance:
    """Spread of a metric across independently seeded corpora."""

    metric: str
    per_seed: dict[int, float]

    @property
    def mean(self) -> float:
        return sum(self.per_seed.values()) / len(self.per_seed)

    @property
    def spread(self) -> float:
        return max(self.per_seed.values()) - min(self.per_seed.values())

    @property
    def sample_stdev(self) -> float | None:
        """None for a single seed, because one observation has no spread.

        Reporting 0.0 there would claim the corpus contributes no variance, which is the
        opposite of what one seed tells you -- namely nothing.
        """
        if len(self.per_seed) < 2:
            return None
        mean = self.mean
        variance = sum((v - mean) ** 2 for v in self.per_seed.values()) / (len(self.per_seed) - 1)
        return math.sqrt(variance)

    def __str__(self) -> str:
        stdev = self.sample_stdev
        stdev_text = "not measured (needs >= 2 seeds)" if stdev is None else f"{stdev:.4f}"
        return (
            f"{self.metric}: mean {self.mean:.3f} over {len(self.per_seed)} seeds, "
            f"range {self.spread:.3f}, sd {stdev_text}"
        )


def seed_variance(metric: str, per_seed: dict[int, float]) -> SeedVariance:
    """Summarise a metric measured on several independently seeded corpora.

    Distinct from the Wilson interval, and both are needed. The interval answers "how
    precisely did we measure this corpus"; seed variance answers "how much does the
    answer depend on which corpus we drew". A single-seed result silently conflates
    them, which is what every number in this project did before this existed.
    """
    if not per_seed:
        raise ValueError("cannot summarise variance over zero seeds")
    return SeedVariance(metric=metric, per_seed=dict(per_seed))
