"""Measure the token cost of each script, to justify a token-based length penalty.

This settles one design question with data instead of intuition: should the verbosity
penalty count characters or tokens?

Counting characters is the obvious choice and it is wrong. This environment accepts an
answer either in the document's own script or in Latin transliteration -- both are
correct extractions, and the correctness term cannot tell them apart. If the length
penalty is denominated in characters, those two identical-in-meaning answers are
charged at very different effective rates, because Indic scripts and Latin do not cost
the same number of tokens per character. The result is a script preference expressed
through the length term, invisible in the correctness term, and impossible for a model
to satisfy except by switching scripts.

Measurement method matters here. A naive reading of `prompt_tokens` is confounded by
the chat template, which adds a fixed overhead -- 71 tokens on gpt-oss-120b, 35 on
llama-3.1-8b-instant -- large enough that short samples appear to cost more tokens than
they have characters. Every figure below is therefore a *difference*: tokens for
(baseline + payload) minus tokens for baseline alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from research.rollout import Client, RolloutRequest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Comparable samples: the same kind of content -- a run of personal names -- in each
# script, at roughly equal character length so the ratios are directly comparable.
SAMPLES: dict[str, str] = {
    "latin": "Shri Ramesh Kumar Sharma Verma Gupta Singh Yadav Mishra Sunita Priya",
    "devanagari": "श्री रमेश कुमार शर्मा वर्मा गुप्ता सिंह यादव मिश्रा सुनीता प्रिया",
    "tamil": "திரு லட்சுமி கிருஷ்ணன் சுப்ரமணியன் நடராஜன் ராமன் முருகன் கார்த்திக்",
    "bengali": "শ্রী দেবাশিস চট্টোপাধ্যায় বন্দ্যোপাধ্যায় মুখার্জি ঘোষ দাস সুব্রত অনিতা",
}

MODELS = ("openai/gpt-oss-120b", "llama-3.1-8b-instant")

# A one-character baseline whose only purpose is to expose the chat-template overhead.
BASELINE = "x"


def content_tokens(client: Client, model: str, text: str) -> int | None:
    """Tokens attributable to `text` alone, with the chat template subtracted off."""
    base = client.run(RolloutRequest(model=model, prompt=BASELINE, max_tokens=1))
    full = client.run(RolloutRequest(model=model, prompt=BASELINE + text, max_tokens=1))
    if base.error or full.error:
        return None
    return full.prompt_tokens - base.prompt_tokens


def measure(client: Client) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for model in MODELS:
        per_script: dict[str, float] = {}
        for script, text in SAMPLES.items():
            tokens = content_tokens(client, model, text)
            if tokens is None or tokens <= 0:
                continue
            per_script[script] = len(text) / tokens
        if per_script:
            results[model] = per_script
    return results


def main() -> int:
    client = Client()
    results = measure(client)
    if not results:
        print("No measurements: every probe failed. Not reporting estimated values.")
        return 1

    for model, per_script in results.items():
        print(f"\n### {model}")
        print(f"{'script':<14}{'chars/token':>13}{'vs Latin':>12}")
        print("-" * 39)
        latin = per_script.get("latin")
        for script, ratio in per_script.items():
            relative = f"{latin / ratio:.2f}x" if latin else "n/a"
            print(f"{script:<14}{ratio:>13.2f}{relative:>12}")

    print()
    print("Read 'vs Latin' as: how many times more tokens the same character count")
    print("costs in this script. A character-denominated length budget is that many")
    print("times more generous to this script than to Latin, for answers the")
    print("correctness term scores identically. Counting tokens removes the effect")
    print("by construction, which is why the shipped penalty is token-based.")

    worst = max(
        (per.get("latin", 0) / ratio, model, script)
        for model, per in results.items()
        for script, ratio in per.items()
        if script != "latin" and per.get("latin")
    )
    print()
    print(f"Largest distortion measured: {worst[0]:.2f}x ({worst[2]} on {worst[1]}).")

    out_path = RESULTS_DIR / "script_bias.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
