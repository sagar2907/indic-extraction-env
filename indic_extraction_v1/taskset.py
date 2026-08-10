"""The verifiers.v1 taskset: task data, rewards, metrics and model-free validation.

Reward design is the part of this file worth reading carefully.

There is exactly one *positive* term -- `field_accuracy` -- and it can only be earned
by producing values that match the document. Every other term is a penalty, declared
with a negative weight and bounded in [0, 1], so its best possible contribution is
zero. This asymmetry is the whole defence against reward hacking: if formatting or
brevity could earn positive reward, the reward-maximising policy would be a terse,
well-formed, contentless reply. Under this scheme that policy scores exactly the
penalty floor and never competes with actually reading the document.

Concretely, for the degenerate `{}` answer: field_accuracy 0, format_violation 1/3
(its keys do not match the schema, though it is valid JSON with a single object),
verbosity 0 -> total -0.083. Reading the document correctly scores +1.0. The gap, not
the exact penalty, is the point.

`field_accuracy` gives partial credit rather than all-or-nothing. That is safe here,
but only because of a property established by measurement rather than assumption: no
field can be guessed for free. `heuristics.py` implements the obvious shortcuts and
`tests/test_corpus_is_not_shortcuttable.py` pins each of them near chance. If a later
change to the generator made some field free, partial credit would start paying for
it -- which is why that test exists and why it is not optional.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import verifiers.v1 as vf

from indic_extraction_v1 import score
from indic_extraction_v1.corpus import TIERS, Record, generate
from indic_extraction_v1.lang import LANGUAGES, LanguageCode
from indic_extraction_v1.verify import Verdict, render_prompt, verify


class IndicExtractionData(vf.TaskData):
    """One document's serialisable ground truth.

    Only the fields scoring needs are stored. The document text itself lives in
    `prompt` (inherited from `vf.TaskData`), so a trace carries everything required to
    re-verify a rollout offline without regenerating the corpus.
    """

    lang: str
    tier: str
    name_native: str
    name_roman: list[str]
    amount_inr: int
    due_date: str
    """ISO 8601. Stored as a string because task data round-trips through JSON."""

    reference: str
    features: dict[str, str]

    def to_record(self) -> Record:
        return Record(
            name_native=self.name_native,
            name_roman=tuple(self.name_roman),
            amount_inr=self.amount_inr,
            due_date=date.fromisoformat(self.due_date),
            reference=self.reference,
        )


class IndicExtractionTask(vf.Task[IndicExtractionData]):
    def _verdict(self, trace: vf.Trace) -> Verdict:
        reply = trace.last_reply or ""
        lang: LanguageCode = self.data.lang  # type: ignore[assignment]
        return verify(reply, self.data.to_record(), lang)

    # ------------------------------------------------------------------ rewards

    @vf.reward(weight=score.WEIGHT_FIELD_ACCURACY)
    async def field_accuracy(self, trace: vf.Trace) -> float:
        """Fraction of the four fields extracted correctly. The only positive term."""
        return score.field_accuracy(self._verdict(trace))

    @vf.reward(weight=score.WEIGHT_FORMAT_VIOLATION)
    async def format_violation(self, trace: vf.Trace) -> float:
        """Penalty in [0, 1] for anything wrong with the response's shape."""
        return score.format_violation(self._verdict(trace))

    @vf.reward(weight=score.WEIGHT_VERBOSITY)
    async def verbosity(self, trace: vf.Trace) -> float:
        """Penalty in [0, 1] for output tokens spent beyond the budget."""
        return score.verbosity(trace.num_output_tokens or 0)

    # ------------------------------------------------------------------ metrics

    @vf.metric
    async def exact_match(self, trace: vf.Trace) -> float:
        """All four fields correct. The number to quote when comparing models."""
        return float(self._verdict(trace).all_correct)

    @vf.metric
    async def schema_clean(self, trace: vf.Trace) -> float:
        return float(self._verdict(trace).schema_clean)

    @vf.metric
    async def day_month_transposed(self, trace: vf.Trace) -> float:
        """Fires when the due date was read month-first instead of day-first.

        Tracked separately because it is a specific, fixable locale failure rather than
        generic inaccuracy, and it is invisible in an aggregate score.
        """
        verdict = self._verdict(trace)
        return float(any(f.reason == "day-month-transposed" for f in verdict.fields))

    @vf.metric
    async def answered_in_roman(self, trace: vf.Trace) -> float:
        """The name was correct but transliterated into Latin rather than kept in script.

        Not penalised -- both are correct extractions -- but worth measuring, because
        the rate varies sharply with model and with reasoning effort.
        """
        verdict = self._verdict(trace)
        return float(any(f.name == "name" and f.reason == "roman" for f in verdict.fields))

    @vf.metric
    async def output_tokens(self, trace: vf.Trace) -> float:
        return float(trace.num_output_tokens or 0)

    # --------------------------------------------------------------- validation

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Model-free check that the verifier accepts this row's own ground truth.

        Feeds the gold answer back through the real scoring path. A row that fails here
        is ungradeable -- the verifier could not recognise its own reference answer --
        and would otherwise show up as a model error forever.
        """
        import json

        record = self.data.to_record()
        gold = json.dumps(
            {
                "name": record.name_native,
                "amount_inr": record.amount_inr,
                "due_date": record.due_date.isoformat(),
                "reference": record.reference,
            },
            ensure_ascii=False,
        )
        lang: LanguageCode = self.data.lang  # type: ignore[assignment]
        return verify(gold, record, lang).all_correct


class IndicExtractionConfig(vf.TasksetConfig):
    seed: int = 0
    """Master seed. Changing it produces a disjoint, equally valid corpus."""

    num_tasks: int = 500
    lang: LanguageCode | None = None
    """Restrict to one language. None mixes all four."""

    tier: str | None = None
    """Restrict to one difficulty tier. None mixes all three."""


class IndicExtractionTaskset(vf.Taskset[IndicExtractionTask, IndicExtractionConfig]):
    def load(self) -> Iterator[IndicExtractionTask]:
        """Yield tasks lazily, one document at a time.

        A generator rather than a list because `Taskset.select(num_tasks)` pulls from
        `load` lazily and only materialises what a run actually takes. Returning a list
        defeated that: a consumer asking for the twenty examples this package declares in
        `[tool.verifiers.eval]` still paid to generate all `num_tasks` documents.

        Generation is per-document and index-independent by construction (see
        `corpus.generate`), so yielding changes nothing about which documents a given
        seed produces -- task N is identical whether or not 0..N-1 were ever built.
        """
        config = self.config
        # Validated here, before the generator is returned, so a bad tier or language
        # still fails at load() rather than on the first pull. A plain generator
        # function would defer this whole body and turn a config error into a
        # confusing failure partway into a run.
        if config.tier is not None and config.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {config.tier!r}")
        if config.lang is not None and config.lang not in LANGUAGES:
            raise ValueError(f"lang must be one of {tuple(LANGUAGES)}, got {config.lang!r}")
        return self._generate_tasks()

    def _generate_tasks(self) -> Iterator[IndicExtractionTask]:
        config = self.config
        for idx in range(config.num_tasks):
            doc = generate(
                idx,
                seed=config.seed,
                lang=config.lang,
                tier=config.tier,  # type: ignore[arg-type]
            )
            yield IndicExtractionTask(
                IndicExtractionData(
                    idx=doc.idx,
                    prompt=render_prompt(doc),
                    lang=doc.lang,
                    tier=doc.tier,
                    name_native=doc.record.name_native,
                    name_roman=list(doc.record.name_roman),
                    amount_inr=doc.record.amount_inr,
                    due_date=doc.record.due_date.isoformat(),
                    reference=doc.record.reference,
                    features=doc.features,
                ),
                config.task,
            )


__all__ = ["IndicExtractionTaskset"]
