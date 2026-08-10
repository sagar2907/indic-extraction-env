"""Run a corpus slice against one or more models and report measured results.

Deliberately a plain script rather than a verifiers harness invocation. The point of
this file is to produce the operating numbers quoted in the README and the report on a
free tier, which means it has to survive running out of quota partway through, resume
across days, and never silently substitute an estimate for a measurement.

Two rules it follows without exception:

* A metric that could not be computed is reported as None, never as zero. A run that
  stops at the budget reports what it measured and how many rollouts it actually made.
* Verification is never cached. Only generation is. Changing the verifier and
  re-running costs nothing and produces genuinely re-verified numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from indic_extraction_v1.corpus import FIELDS, Document, generate_many
from indic_extraction_v1.score import reward_terms
from indic_extraction_v1.verify import render_prompt, verify_document
from research.rollout import BudgetExhausted, Client, RolloutRequest
from research.statistics_ import Interval, mcnemar, seed_variance, wilson

DEFAULT_MODELS = ("openai/gpt-oss-120b", "llama-3.1-8b-instant", "gemini-3.6-flash")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class ModelReport:
    model: str
    n_requested: int
    n_completed: int
    n_errors: int
    stopped_early: bool
    stop_reason: str | None

    # Every figure below is None when n_completed is 0, so an aborted run cannot be
    # mistaken for a run that measured zero.
    field_accuracy: float | None
    exact_match: float | None
    mean_reward: float | None
    per_field: dict[str, float | None]
    per_tier: dict[str, float | None]
    per_lang: dict[str, float | None]
    day_month_transposed: float | None
    answered_in_roman: float | None
    schema_clean: float | None
    mean_output_tokens: float | None
    mean_total_tokens: float | None
    mean_hidden_reasoning_tokens: float | None
    truncated_rate: float | None
    """Fraction of replies cut off by the token ceiling.

    Reported separately because truncation is a harness failure, not a model failure.
    Two of forty gemini-3.6-flash rollouts were scored as wrong answers when they were
    actually correct answers cut off mid-field, the token ceiling having been consumed
    by hidden reasoning. A non-zero value here means the numbers understate the model."""

    exact_outcomes: list[bool]
    """Per-document correctness, in corpus order.

    Kept because model comparison must be paired: every model sees the identical
    documents, so the informative question is which ones they disagree on, not whether
    two marginal intervals happen to overlap. Discarding this vector would throw away
    exactly that information, and comparing marginals is strictly less powerful."""

    failure_reasons: dict[str, int]

    def exact_interval(self) -> Interval | None:
        """Wilson interval on exact match, or None when nothing was measured."""
        if not self.exact_outcomes:
            return None
        return wilson(sum(self.exact_outcomes), len(self.exact_outcomes))


def _mean(values: list[float]) -> float | None:
    """Mean, or None for an empty sample.

    Returning None rather than 0.0 is the whole point: a metric with no data must be
    visibly absent, not silently reported as the worst possible score.
    """
    return statistics.fmean(values) if values else None


def evaluate_model(
    client: Client,
    model: str,
    docs: list[Document],
    *,
    max_tokens: int,
    allow_network: bool,
) -> ModelReport:
    field_hits: list[float] = []
    exact: list[float] = []
    rewards: list[float] = []
    per_field: dict[str, list[float]] = defaultdict(list)
    per_tier: dict[str, list[float]] = defaultdict(list)
    per_lang: dict[str, list[float]] = defaultdict(list)
    transposed: list[float] = []
    roman: list[float] = []
    clean: list[float] = []
    out_tokens: list[float] = []
    tot_tokens: list[float] = []
    hidden: list[float] = []
    truncated: list[float] = []
    reasons: Counter[str] = Counter()

    errors = 0
    stopped_early = False
    stop_reason: str | None = None

    for doc in docs:
        request = RolloutRequest(model=model, prompt=render_prompt(doc), max_tokens=max_tokens)
        try:
            result = client.run(request, allow_network=allow_network)
        except BudgetExhausted as exc:
            stopped_early = True
            stop_reason = str(exc)
            break

        if result.error:
            errors += 1
            reasons[f"api:{result.error.split(':')[0]}"] += 1
            continue

        verdict = verify_document(result.reply, doc)
        terms = reward_terms(verdict, result.completion_tokens)

        accuracy = verdict.n_correct / len(FIELDS)
        field_hits.append(accuracy)
        exact.append(float(verdict.all_correct))
        rewards.append(terms.total)
        per_tier[doc.tier].append(accuracy)
        per_lang[doc.lang].append(accuracy)
        for field in verdict.fields:
            per_field[field.name].append(float(field.correct))
            if not field.correct:
                reasons[f"{field.name}:{field.reason}"] += 1
        transposed.append(float(any(f.reason == "day-month-transposed" for f in verdict.fields)))
        roman.append(float(any(f.name == "name" and f.reason == "roman" for f in verdict.fields)))
        clean.append(float(verdict.schema_clean))
        truncated.append(float(result.truncated))
        out_tokens.append(result.completion_tokens)
        tot_tokens.append(result.total_tokens)
        hidden.append(result.hidden_reasoning_tokens)

    return ModelReport(
        model=model,
        n_requested=len(docs),
        n_completed=len(field_hits),
        n_errors=errors,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        field_accuracy=_mean(field_hits),
        exact_match=_mean(exact),
        mean_reward=_mean(rewards),
        per_field={f: _mean(per_field[f]) for f in FIELDS},
        per_tier={k: _mean(v) for k, v in sorted(per_tier.items())},
        per_lang={k: _mean(v) for k, v in sorted(per_lang.items())},
        day_month_transposed=_mean(transposed),
        answered_in_roman=_mean(roman),
        schema_clean=_mean(clean),
        mean_output_tokens=_mean(out_tokens),
        mean_total_tokens=_mean(tot_tokens),
        mean_hidden_reasoning_tokens=_mean(hidden),
        truncated_rate=_mean(truncated),
        exact_outcomes=[bool(e) for e in exact],
        failure_reasons=dict(reasons.most_common(15)),
    )


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "not measured" if value is None else format(value, spec)


def _print_seed_variance(by_seed: dict[int, list[ModelReport]]) -> None:
    """Report how much each headline metric moves across independently seeded corpora.

    A Wilson interval says how precisely one corpus was measured. It says nothing about
    how much the answer depends on *which* corpus was drawn, and a single-seed result
    silently conflates the two. Both belong in a report; neither substitutes for the
    other.
    """
    print("Corpus-seed variance")
    models = [r.model for r in next(iter(by_seed.values()))]
    for model in models:
        for metric in ("exact_match", "field_accuracy"):
            per_seed = {}
            for seed, reports in by_seed.items():
                report = next((r for r in reports if r.model == model), None)
                value = getattr(report, metric, None) if report else None
                if value is not None:
                    per_seed[seed] = value
            if not per_seed:
                print(f"  {model} {metric}: not measured")
                continue
            summary = seed_variance(f"{model} {metric}", per_seed)
            print(f"  {summary}")
            print(
                "      per seed: " + "  ".join(f"{s}={v:.3f}" for s, v in sorted(per_seed.items()))
            )
    print()


def _print_paired_comparisons(reports: list[ModelReport]) -> None:
    """Compare every pair of models that saw the same documents.

    Paired rather than by overlapping intervals. Every model in a run is evaluated on an
    identical corpus, so the informative comparison is per-document agreement; McNemar
    uses only the documents where the two disagree, which is where all the evidence is.
    Two marginal intervals can overlap while the paired difference is unambiguous.
    """
    comparable = [r for r in reports if r.exact_outcomes]
    if len(comparable) < 2:
        return

    print("Paired model comparison (McNemar, exact match)")
    for i, first in enumerate(comparable):
        for second in comparable[i + 1 :]:
            if len(first.exact_outcomes) != len(second.exact_outcomes):
                # Not an error: a run can stop at the budget mid-model. Say so rather
                # than silently truncating to the shorter vector, which would compare
                # the models on different corpora and call it a paired test.
                print(
                    f"  {first.model} vs {second.model}: not comparable "
                    f"(n={len(first.exact_outcomes)} vs {len(second.exact_outcomes)}; "
                    f"a paired test needs the same documents)"
                )
                continue
            result = mcnemar(first.model, first.exact_outcomes, second.model, second.exact_outcomes)
            first_interval = first.exact_interval()
            second_interval = second.exact_interval()
            overlap = (
                first_interval.overlaps(second_interval)
                if first_interval and second_interval
                else False
            )
            note = "  <- paired test still separates them" if overlap and result.significant else ""
            print(f"  {result}")
            print(f"      marginal intervals overlap: {overlap}{note}")
    print()


def print_report(reports: list[ModelReport]) -> None:
    header = (
        f"{'model':<26}{'exact match (Wilson 95%)':<34}{'field acc':<12}{'reward':>9}{'out tok':>9}"
    )
    print(header)
    print("-" * len(header))
    for report in reports:
        interval = report.exact_interval()
        exact = "not measured" if interval is None else str(interval)
        print(
            f"{report.model:<26}{exact:<34}"
            f"{_fmt(report.field_accuracy):<12}"
            f"{_fmt(report.mean_reward):>9}{_fmt(report.mean_output_tokens, '.0f'):>9}"
        )
    print()
    print(
        "Intervals are Wilson score, not the normal approximation: at small n the normal\n"
        "approximation returns bounds outside [0, 1] (arXiv:2503.01747). They cover\n"
        "sampling error over documents only -- run several seeds for corpus variance."
    )
    print()

    _print_paired_comparisons(reports)

    def breakdown(values: dict[str, float | None]) -> str:
        return "  ".join(f"{k}={_fmt(v)}" for k, v in values.items())

    for report in reports:
        print(f"== {report.model}")
        print(f"   per field : {breakdown(report.per_field)}")
        print(f"   per tier  : {breakdown(report.per_tier)}")
        print(f"   per lang  : {breakdown(report.per_lang)}")
        print(
            f"   transposed: {_fmt(report.day_month_transposed)}   "
            f"roman name: {_fmt(report.answered_in_roman)}   "
            f"schema ok: {_fmt(report.schema_clean)}"
        )
        print(
            f"   tokens    : out {_fmt(report.mean_output_tokens, '.0f')}  "
            f"total {_fmt(report.mean_total_tokens, '.0f')}  "
            f"hidden reasoning {_fmt(report.mean_hidden_reasoning_tokens, '.0f')}"
        )
        if report.truncated_rate:
            print(
                f"   TRUNCATED : {report.truncated_rate:.3f} of replies were cut off by "
                f"the token ceiling; those are harness failures, not model failures"
            )
        if report.n_errors:
            print(f"   errors    : {report.n_errors}")
        if report.stopped_early:
            print(f"   STOPPED   : {report.stop_reason}")
        if report.failure_reasons:
            print(f"   top failures: {report.failure_reasons}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="documents to evaluate")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "evaluate several independently seeded corpora and report the spread. "
            "A Wilson interval covers sampling error within one corpus; this covers "
            "how much the answer depends on which corpus was drawn."
        ),
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--tier", default=None)
    parser.add_argument("--lang", default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="fail on cache miss instead of calling the API; for reproducing a run",
    )
    parser.add_argument("--out", default=None, help="write JSON results here")
    args = parser.parse_args(argv)

    client = Client()
    seeds = args.seeds if args.seeds else [args.seed]

    by_seed: dict[int, list[ModelReport]] = {}
    for seed in seeds:
        docs = generate_many(args.n, seed=seed, lang=args.lang, tier=args.tier)
        if len(seeds) > 1:
            print(f"\n########## seed {seed} ##########")
        reports = []
        for model in args.models:
            reports.append(
                evaluate_model(
                    client,
                    model,
                    docs,
                    max_tokens=args.max_tokens,
                    allow_network=not args.offline,
                )
            )
        by_seed[seed] = reports
        print_report(reports)

    reports = by_seed[seeds[0]]
    if len(seeds) > 1:
        _print_seed_variance(by_seed)

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"eval_seed{args.seed}_n{args.n}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "n": args.n,
                    "seed": seeds[0],
                    "seeds": seeds,
                    "models": args.models,
                    "tier": args.tier,
                    "lang": args.lang,
                    "max_tokens": args.max_tokens,
                },
                "reports": [asdict(r) for r in reports],
                "by_seed": {str(s): [asdict(r) for r in rs] for s, rs in by_seed.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
