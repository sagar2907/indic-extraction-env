"""Score every adversarial policy under every reward design and print the matrix.

The output is the central evidence for the environment's reward-hacking claims. Read a
row as "what this attack earns"; read a column as "what this reward design permits".
The `shipped` column is the one that ships; the rest are counterfactuals kept so the
mitigations can be shown to matter rather than merely asserted.

Fully offline and deterministic: no model, no network, no clock. It is therefore
reproducible by anyone and safe to run in CI, which is where the exploit ceilings are
pinned as assertions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from indic_extraction_v1.corpus import generate_many
from research.ablations import ABLATIONS
from research.adversarial import POLICIES, estimate_tokens

RESULTS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Policies that answer without comprehending the document. `honest_oracle` is the
# reference and `correct_but_padded` reads correctly too -- it is a calibration probe
# for the length penalty, not an attack. Lumping it in with the attacks makes every
# margin trivially about padding and hides what the attacks actually earn.
NON_READING_ATTACKS = (
    "empty_schema",
    "shotgun_objects",
    "shotgun_lists",
    "key_stuffing",
    "terse_plausible_guess",
)

# Which alternative reward each attack was written to exploit. The difference between
# the two columns is the measured value of the corresponding mitigation.
TARGETED_BY = {
    "empty_schema": "format_bonus",
    "shotgun_objects": "lenient_extractor",
    "shotgun_lists": "membership_credit",
    "key_stuffing": "lenient_extractor",
}


def score_matrix(n: int, seed: int) -> dict[str, dict[str, float]]:
    docs = generate_many(n, seed=seed)
    matrix: dict[str, dict[str, float]] = {}
    for policy_name, policy in POLICIES.items():
        replies = [(policy(doc), doc) for doc in docs]
        row = {}
        for reward_name, reward in ABLATIONS.items():
            row[reward_name] = statistics.fmean(
                reward(reply, doc, estimate_tokens(reply)) for reply, doc in replies
            )
        matrix[policy_name] = row
    return matrix


def print_matrix(matrix: dict[str, dict[str, float]]) -> None:
    rewards = list(next(iter(matrix.values())).keys())
    width = max(len(name) for name in matrix) + 2
    header = "policy".ljust(width) + "".join(name.rjust(20) for name in rewards)
    print(header)
    print("-" * len(header))
    for policy_name, row in matrix.items():
        line = policy_name.ljust(width)
        line += "".join(f"{row[name]:>20.3f}" for name in rewards)
        print(line)
    print()

    honest = matrix["honest_oracle"]

    print("Free floor: what a well-formed but CONTENTLESS reply earns.")
    print("Anything above zero is reward paid for producing braces, collectable")
    print("without reading the document at all.")
    print()
    for name in rewards:
        floor = matrix["empty_schema"][name]
        verdict = "LEAKS" if floor > 1e-9 else "clean"
        print(f"  {name:<20} empty_schema = {floor:>7.3f}   {verdict}")
    print()

    print("Best non-reading attack, as a fraction of honest extraction.")
    print()
    for name in rewards:
        attacks = {p: matrix[p][name] for p in NON_READING_ATTACKS}
        best = max(attacks, key=lambda k: attacks[k])
        ratio = attacks[best] / honest[name] if honest[name] else float("nan")
        print(
            f"  {name:<20} {best:<24} {attacks[best]:>7.3f} / honest {honest[name]:>7.3f}"
            f"  = {ratio:>5.1%}"
        )
    print()

    print("Mitigation value: how much the shipped design removes from each attack,")
    print("measured against the alternative reward that attack was written to exploit.")
    print()
    for attack, ablation in TARGETED_BY.items():
        exploited = matrix[attack][ablation]
        contained = matrix[attack]["shipped"]
        print(
            f"  {attack:<24} under {ablation:<20} {exploited:>6.3f}"
            f"  ->  shipped {contained:>6.3f}   removed {exploited - contained:>+6.3f}"
        )
    print()

    print("Calibration: a correct answer must survive its own penalties.")
    padded = matrix["correct_but_padded"]["shipped"]
    print(
        f"  correct_but_padded scores {padded:.3f} under the shipped reward, "
        f"still far above the best attack "
        f"({max(matrix[p]['shipped'] for p in NON_READING_ATTACKS):.3f})."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    matrix = score_matrix(args.n, args.seed)
    print_matrix(matrix)

    out_path = Path(args.out) if args.out else RESULTS_DIR / "reward_hacking_matrix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"n": args.n, "seed": args.seed, "matrix": matrix}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
