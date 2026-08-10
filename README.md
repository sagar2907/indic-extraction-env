# indic-extraction-v1

A verifiable RL environment for the [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments):
structured extraction from Indic-script documents, scored by a deterministic,
script-aware verifier.

Of 1,501 environments on the Hub, searching `indic` returns two — this one and an
instruction-following eval — while `hindi`, `tamil`, `bengali`, `marathi` and
`devanagari` each return zero.

A model is shown a Hindi, Marathi, Tamil or Bengali payment document and must return a
single JSON object with four fields — the addressee's name, the amount payable, the due
date, and the document's reference number. Each document also contains a *rival*
candidate for every one of those fields: a second person, a second amount, a second
date, a second reference. The task is discrimination, not spotting.

Built against **`verifiers` v1** (`Taskset` / `Task` / `@vf.reward`). The v0 API
(`SingleTurnEnv`, `Rubric`, `Parser`, `load_environment`) is deprecated upstream and is
not used here.

```bash
uv pip install -e .
```

The published package has **zero runtime dependencies**. The corpus is generated from
an integer seed rather than downloaded, and every normaliser is standard-library only,
so the environment installs and runs offline.

---

## Honest status

### Built and verified

| Component | Status | Evidence |
|---|---|---|
| Seeded corpus generator, 4 languages × 3 tiers | Working | `tests/test_corpus.py`, 190 tests pass |
| Deterministic verifier (digits, dates, amounts, names, references) | Working | `tests/test_normalize.py`, `tests/test_verify.py` |
| `verifiers` v1 taskset with 3 rewards + 5 metrics | Loads and scores | validated on Linux, see *Platform* below |
| Reward-hacking analysis, 3 exploits found and closed | Complete | `tests/test_reward_hacking.py`, table below |
| Shortcut-resistance audit | Complete | `tests/test_corpus_is_not_shortcuttable.py` |
| Cached, budget-aware rollout runner | Working | 1,100+ live rollouts, `tests/test_budget.py` |
| End-to-end live evaluation, 3 models | Complete | numbers below, all measured |
| Wheel builds, installs standalone, loads from a clean env | Verified | zero deps, `tests/test_packaging.py` |
| Published to the Environments Hub | Live, CI green | `sagar2907/indic-extraction-v1@0.1.3` |
| Wilson intervals, paired comparison, seed variance | Complete | `research/statistics_.py`, `tests/test_statistics.py` |
| Works on `verifiers` 0.2.1 **and** 0.3.0 | Verified | suite green on both, `tests/test_taskset.py` |

### Not done, and why

- **No GRPO training run.** The reward-hacking analysis uses hand-written adversarial
  policies and offline ablation rewards, not gradient-based search. Training even a
  0.6B model needs a GPU, and this project was built for $0. The substitute is
  defensible — the ablations show exactly what each mitigation is worth — but it is a
  substitute, and it is not the same as watching a policy discover an exploit during
  training. Nothing in this repository reports a training curve, because none was run.
- **`gemini-3.6-flash` measured on 61 documents, not 100.** The free tier will not
  sustain a longer run. At 10 rpm, 82 of 100 rollouts returned `RateLimitError`; at 5 rpm
  with bounded backoff a run reaches roughly sixty and then stalls against what appears
  to be a hard daily request cap that backoff cannot clear. Two attempts on separate days
  produced 40 and 61. The 61 are reported as 61 and are not extrapolated.
- **Both of Gemini's failures on that sample are harness truncation, not comprehension.**
  Hidden reasoning is charged against `max_tokens`: 670 reasoning tokens left 26 for the
  reply, which was cut off mid-field. Those two documents are the *only* errors in the
  sample — on the other 59 it got all four fields right every time. So 0.967 is a floor,
  and the model made no reading mistakes here at all.
- **The corrected-ceiling re-run still has not happened.** `RolloutRequest.max_tokens`
  was raised to 1400, but `evaluate.py` kept its own `--max-tokens` default of 700 and
  passes it explicitly, so a re-run intended to use the new ceiling silently used the old
  one and re-served the truncated documents from cache. The CLI now inherits the value
  instead of restating it (`tests/test_budget.py` pins this), but the clean run is
  blocked on tomorrow's quota.
- **The first push failed the Hub's own integration test.** Version 0.1.0 uploaded
  successfully and then failed CI: the Hub asserts on a `tags` key in `[project]`, a
  Prime Intellect extension rather than a PEP 621 field, so a pyproject carrying only
  the standard `keywords` passes every local build and fails remotely.
  `tests/test_packaging.py` now mirrors the Hub's metadata assertions locally so that
  class of failure surfaces from `pytest` rather than by email.
- **Cross-script name acceptance is untriggered in the headline runs.** All three
  models answered in the document's own script at `reasoning_effort=low`, so the
  romanisation-accepting code path scored 0.000 in every run. It is not dead code — the
  same model at `reasoning_effort=medium` returned `"Shri Ramesh Kumar Sharma"` for a
  Devanagari document during development, which is what motivated it — but the headline
  numbers do not exercise it.
- **No human validation of the generated Indic text.** The documents are templated from
  hand-written language data. A native speaker has not reviewed them for naturalness.
  They are structurally correct, not certified idiomatic. This is the cheapest remaining
  gap and the one where a reviewer could most reasonably say the artifact overclaims.
- **Four languages of the twenty-two scheduled**, and one document genre. Adding a
  language is a data change rather than a code change — a `Language` entry in `lang.py` —
  but breadth is deliberately gated on the review above: more unreviewed languages would
  multiply an existing weakness rather than reduce one.
- **The Hub listing is still PRIVATE.** `prime env push --visibility PUBLIC` is accepted
  by the CLI and does not take effect on an existing environment — the flag appears to
  apply only at creation, and `prime env` exposes no command to change visibility
  afterwards. The remaining step is a one-click change in the Hub dashboard. Everything
  else is done: 0.1.3 is live and its integration test passes.
- **No OCR-noise tier.** Real Indic documents arrive through OCR. The `render_variant`
  machinery could carry one, but the current corpus tests clean text only.

### Compatibility

Verified against **`verifiers` 0.2.1 and 0.3.0**. The two differ: 0.2.1 exposes
`Taskset.select(n)` returning a list, and 0.3.0 replaced it with `Taskset.head(n)`
returning a lazy *view*. The environment itself is unaffected — it exports a `Taskset`
subclass whose `load()` is a generator, which is what both APIs need — but a fresh
install resolves `verifiers>=0.2.1` to 0.3.0, and the test suite originally named the
removed method.

Two things are worth knowing if you consume this. On 0.3.0, `__iter__` is the read path
and applies a view's transform, while `load()` is the raw subclass hook: calling
`head(5).load()` bypasses the `islice` and yields the whole taskset. And a development
environment installed weeks ago pins silently while every new consumer gets the latest
release, so "the tests pass here" is not evidence that they pass for anyone else —
`tests/test_taskset.py` now exercises whichever API is present.

### Platform

`verifiers.v1` imports `fcntl` and **cannot be installed on Windows**. The deterministic
core — corpus, normalisers, verifier, reward, shortcut baselines — has no such
dependency and runs anywhere; only `taskset.py` needs a POSIX host. `__init__.py`
resolves the taskset export lazily (PEP 562) so that importing
`indic_extraction_v1.corpus` on Windows does not drag in `verifiers`. Tests and lint
were run on both Windows (190 passed, 1 skipped) and Linux/WSL (198 passed, ruff clean).

---

## Measured results

All numbers below are from real API calls, not estimates. Nothing here is extrapolated;
where a figure could not be measured it is reported as `not measured`.

### Model comparison — 200 documents, seed 1, mixed languages and tiers

| Model | Exact match (Wilson 95%) | Field accuracy | Mean reward | Output tokens |
|---|---|---:|---:|---:|
| `openai/gpt-oss-120b` | **0.920** [0.874, 0.950] (n=200) | 0.976 | 0.975 | 197 |
| `llama-3.1-8b-instant` | **0.620** [0.551, 0.684] (n=200) | 0.897 | 0.897 | 68 |
| `gemini-3.6-flash` | **0.967** [0.888, 0.991] (n=61) | 0.967 | 0.962 | 73 |

Intervals are Wilson score, not the normal approximation. At n=40 the normal
approximation puts Gemini's upper bound at **1.018** — an accuracy above 100% — which is
the small-sample failure described in [arXiv:2503.01747](https://arxiv.org/abs/2503.01747).

**The two fully measured models are cleanly separated, and the evidence is paired rather
than marginal.** Every model sees the identical corpus, so the comparison is McNemar's
exact test on per-document outcomes: `gpt-oss-120b` wins **63** documents that
`llama-3.1-8b-instant` loses and loses **3**, p = 1.3e-15. That rests on the 66 documents
where they disagree and would hold even if the intervals overlapped.

**`gemini-3.6-flash` cannot be ranked against the others here.** Its interval overlaps
`gpt-oss-120b`'s, it was measured on 40 documents rather than 200, and two of its
failures were replies truncated by my own token ceiling rather than wrong answers — so
0.950 is a floor. Listed with all three caveats attached rather than presented as
comparable.

### How much the answer depends on which corpus was drawn

Five independently seeded corpora, 100 documents each:

| Model | seed 1 | 2 | 3 | 4 | 5 | mean | range | sd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-oss-120b` exact | 0.920 | 0.920 | 0.950 | 0.950 | 0.978 | 0.944 | 0.058 | 0.024 |
| `llama-3.1-8b` exact | 0.580 | 0.620 | 0.690 | 0.660 | 0.670 | 0.644 | 0.110 | 0.044 |

Corpus choice alone moves `llama-3.1-8b`'s exact match by **11 points**. Seed 1 — the
seed every number in this repository came from before this table existed — is the
**minimum** of the range for both models. Those earlier figures were not wrong, and were
conservative rather than flattering, but they were under-determined and nothing let a
reader see it.

Exact match is far more seed-sensitive than field accuracy (sd 0.044 vs 0.010), which is
arithmetic rather than a surprise: it is a conjunction of four field decisions and
amplifies whatever moves them. The headline metric is the one most in need of several
seeds.

The paired result survives all of it. On every seed where both models covered the same
documents, discordant counts were 36/2, 31/1, 28/2 and 30/1, p between 5e-9 and 9e-7.

On the fifth seed, eleven transient API errors left `gpt-oss-120b` with 89 documents
against `llama`'s 100, and the reporting code **refused the paired test** rather than
truncating to the shorter vector — which would have compared two different corpora and
called the result paired.

### Per-field accuracy

| Model | name | amount_inr | due_date | reference |
|---|---:|---:|---:|---:|
| `openai/gpt-oss-120b` | 0.995 | 0.995 | 0.945 | 0.970 |
| `llama-3.1-8b-instant` | 0.990 | 0.870 | 0.755 | 0.975 |

### Why the failures happen

Classified against the specific distractor placed in each document
(`research/failure_analysis.py`, offline over the rollout cache):

| Failure | `gpt-oss-120b` | `llama-3.1-8b` |
|---|---|---|
| due_date — picked the rival date in the document | 10 / 10 (100%) | 46 / 49 (94%) |
| due_date — day/month transposed | 0 | 0 |
| amount — lakh/crore magnitude error | 0 | 10 / 26 (38%) |
| amount — picked a rival amount | 0 | 6 / 26 (23%) |

Two things worth noting. Not one date failure across 400 rollouts was a parsing or
day/month-ordering error — every one was choosing the wrong line. And llama-3.1-8b
produces a distinct, nameable failure the larger model never does: reading `7 லட்சம்`
as 70,00,000 instead of 7,00,000. That is Indian-numbering conversion, isolated as its
own error class.

### The corpus cannot be shortcut

Policies that never read the document, scored over 600 documents
(`indic_extraction_v1/heuristics.py`):

| Policy | Field accuracy | Exact match |
|---|---:|---:|
| `largest_amount` | 0.463 | 0.060 |
| `first_of_each` | 0.416 | 0.018 |
| `latest_date` | 0.416 | 0.028 |
| `positional` | 0.085 | 0.000 |
| `empty` | 0.000 | 0.000 |

With one rival per field, chance is 0.5 per field and 0.5⁴ = 0.0625 exact. The best
shortcut sits at 0.060 exact — at the chance floor.

**This was not true of the first design.** The original generator pinned the addressee
to line 0 and gave the easy and medium tiers no distractors at all, which made
`largest_amount` score 0.717 field / 0.327 exact and made the `name` field extractable
at accuracy **1.000** by the rule "take the first line". Difficulty came from the
*absence* of competitors rather than from surface complexity. The generator was
rebuilt so every field has exactly one rival at every tier, positioned to make each
shortcut a coin flip. `tests/test_corpus_is_not_shortcuttable.py` pins the ceilings so
the regression cannot return silently.

### Reward hacking: three exploits found and closed

Attack policies scored under the shipped reward and under four alternative designs
(`research/hack_report.py`, 400 documents, fully offline):

| Attack | Under the naive reward | Under shipped | Removed |
|---|---:|---:|---:|
| `empty_schema` — right keys, no content | 0.300 *(format bonus)* | **0.000** | 0.300 |
| `shotgun_objects` — one object per candidate combination | 0.734 *(lenient extractor)* | **0.245** | 0.489 |
| `shotgun_lists` — every field a list of all candidates | 0.846 *(membership credit)* | **0.000** | 0.846 |
| `key_stuffing` — answer plus every candidate as extra keys | 0.467 *(lenient extractor)* | 0.384 | 0.083 |

Honest extraction scores 1.000. The mitigations:

1. **Exactly one reward term can be positive.** Formatting and brevity are penalties
   bounded in `[0, 1]` with negative weights, so their best contribution is zero. A
   positive format bonus pays 0.300 — 23% of full marks — for producing braces.
2. **Grade the first JSON object, never the best-matching one,** and penalise
   multiplicity. Searching for the best object lets the model submit a slate of guesses
   and have the grader do the extraction.
3. **Require a scalar per field.** Crediting membership in a list turns "which of these
   is the due date" into "list the dates", removing the entire task.

A fourth design flaw was found by direct measurement rather than by attack. The
verbosity penalty originally counted characters. Measured tokens-per-character relative
to Latin, for identical content:

| Script | `gpt-oss-120b` | `llama-3.1-8b` |
|---|---:|---:|
| Devanagari | 1.16× | 1.94× |
| Bengali | 1.44× | 4.20× |
| Tamil | 1.66× | **4.52×** |

Since this environment accepts an answer in the source script *or* in Latin
transliteration, and the correctness term cannot tell them apart, a character budget
charges those two identical-in-meaning answers at rates differing by up to 4.5×. The
penalty is denominated in tokens instead, which removes the bias by construction.
(`research/script_bias.py`.)

---

## Running it

```bash
uv pip install -e ".[dev]"
python -m pytest -q          # 189 offline tests, no API key needed
```

Everything above runs offline with no API key.

To evaluate against live models, put keys in a gitignored `.env`:

```bash
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

```bash
python -m research.evaluate --n 200 --seed 1 --models openai/gpt-oss-120b
```

Rollouts are cached under `.cache/rollouts/`, keyed by every input that affects the
output. Re-running after a verifier change costs nothing, because only generation is
cached — verification always re-runs. The runner tracks a per-provider daily token
budget and stops cleanly at the cap rather than failing mid-run.

Offline analyses, all free:

```bash
python -m research.hack_report --n 400
python -m research.failure_analysis --n 200
python -m research.script_bias
```

## Publishing

```bash
uv tool install prime
prime login
prime env push
```

Requires a Prime Intellect account with a username set in your profile — push is
rejected without one.

Published from a `git archive` export rather than the working tree, so only tracked
files are uploaded and the rollout cache, virtualenv and `.env` cannot leak into the
registry:

```bash
git archive --format=tar HEAD | tar -x -C /tmp/stage && cd /tmp/stage && prime env push
```

## Report

`report/report.md` is the full write-up — concepts from first principles, every design
decision and the reasoning behind it, the decisions that turned out wrong, and the
measurements. Render it:

```bash
python scripts/render_report.py
```

## Layout

```
indic_extraction_v1/     the published package (zero dependencies)
  lang.py                per-language data: scripts, months, honorifics, name pools
  corpus.py              seeded deterministic document generator
  normalize.py           digit folding, date/amount/name/reference normalisation
  verify.py              JSON extraction and per-field verification
  score.py               reward composition, runtime-independent
  heuristics.py          no-comprehension baselines that audit the corpus
  taskset.py             the verifiers v1 Taskset
research/                not published; evaluation and analysis tooling
tests/                   190 tests, fully offline
```

## Licence

MIT.
