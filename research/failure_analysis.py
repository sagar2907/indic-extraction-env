"""Classify wrong answers by *why* they are wrong, using only cached rollouts.

An aggregate accuracy number says a model got the due date wrong. It does not say
whether the model misparsed a date format or read the wrong line, and those call for
completely different fixes -- one is a parser problem, the other is the actual task.

This script answers that question by checking each wrong value against the distractor
that was deliberately placed in the document. Runs offline against the rollout cache,
so it costs nothing and is reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from indic_extraction_v1.corpus import generate_many
from indic_extraction_v1.heuristics import candidate_amounts, candidate_dates
from indic_extraction_v1.lang import LANGUAGES
from indic_extraction_v1.normalize import parse_amount, parse_date
from indic_extraction_v1.verify import extract_json, render_prompt, verify_document
from research.rollout import ROLLOUT_DIR


def _is_power_of_ten_multiple(predicted: int, truth: int) -> bool:
    """Whether two amounts differ by a clean factor of 10, 100 or 1000.

    That is the signature of a lakh/crore conversion slip: 7 lakh read as 7 crore is
    exactly 100x, and 3.2 lakh written as 32 lakh is exactly 10x. Ordinary misreadings
    do not land on round powers of ten.
    """
    if predicted <= 0 or truth <= 0 or predicted == truth:
        return False
    high, low = max(predicted, truth), min(predicted, truth)
    if high % low:
        return False
    return (high // low) in (10, 100, 1000, 10_000)


def load_cache_by_prompt() -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for path in sorted(ROLLOUT_DIR.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("prompt") and record.get("model"):
            index[(record["model"], record["prompt"])] = record
    return index


def analyse(model: str, n: int, seed: int) -> None:
    cache = load_cache_by_prompt()
    docs = generate_many(n, seed=seed)

    date_reasons: Counter[str] = Counter()
    amount_reasons: Counter[str] = Counter()
    examined = 0

    for doc in docs:
        record = cache.get((model, render_prompt(doc)))
        if record is None or not record.get("reply"):
            continue
        examined += 1
        verdict = verify_document(record["reply"], doc)
        obj, _ = extract_json(record["reply"])
        if obj is None:
            continue

        for field in verdict.fields:
            if field.correct:
                continue

            if field.name == "due_date":
                predicted = parse_date(str(obj.get("due_date", "")), LANGUAGES[doc.lang].months)
                if predicted is None:
                    date_reasons["unparseable"] += 1
                    continue
                truth = doc.record.due_date
                rivals = [d for d in candidate_dates(doc) if d != truth]
                if predicted in rivals:
                    date_reasons["picked the rival date in the document"] += 1
                elif predicted.day == truth.month and predicted.month == truth.day:
                    date_reasons["day/month transposed"] += 1
                elif abs((predicted - truth).days) <= 1:
                    date_reasons["off by one day"] += 1
                elif predicted.year != truth.year:
                    date_reasons["wrong year"] += 1
                else:
                    date_reasons["other wrong date"] += 1

            if field.name == "amount_inr":
                predicted_amount = parse_amount(obj.get("amount_inr", ""))
                if predicted_amount is None:
                    amount_reasons["unparseable"] += 1
                    continue
                truth_amount = doc.record.amount_inr
                rivals = [a for a in candidate_amounts(doc) if a != truth_amount]
                if predicted_amount in rivals:
                    amount_reasons["picked a rival amount in the document"] += 1
                elif predicted_amount == sum(candidate_amounts(doc)):
                    amount_reasons["summed the amounts"] += 1
                elif _is_power_of_ten_multiple(predicted_amount, truth_amount):
                    # Distinguished because it is a specific, nameable capability gap
                    # rather than generic inaccuracy: converting Indian numbering words
                    # (lakh = 10^5, crore = 10^7) into an integer number of rupees.
                    # Reporting these as "wrong amount" would hide the pattern entirely.
                    amount_reasons["lakh/crore magnitude error"] += 1
                else:
                    amount_reasons["other wrong amount"] += 1

    print(f"\n### {model}   ({examined} cached rollouts examined)")
    for title, counter in (
        ("due_date failures", date_reasons),
        ("amount_inr failures", amount_reasons),
    ):
        total = sum(counter.values())
        print(f"\n  {title}: {total}")
        if not total:
            print("    none")
            continue
        for reason, count in counter.most_common():
            print(f"    {reason:<44}{count:>4}  ({count / total:.0%})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["openai/gpt-oss-120b", "llama-3.1-8b-instant", "gemini-3.6-flash"],
    )
    args = parser.parse_args(argv)
    for model in args.models:
        analyse(model, args.n, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
