# indic-extraction-v1

## A verifiable RL environment, built from first principles

---

# Part 0 — What you are looking at

This document explains a piece of machine-learning infrastructure called a *verifiable
reinforcement-learning environment*, and specifically the one in this repository. It
assumes no background in reinforcement learning, language-model post-training, or
Indian language processing. Everything is built up from scratch.

The short version: modern language models are improved after their initial training by
letting them attempt a task thousands of times and rewarding the attempts that succeed.
That only works if "succeeded" can be decided by a program rather than a person. An
*environment* is the pairing of a task generator with such a program. This repository
contains one, for the task of reading Indian-language payment documents and extracting
four structured fields from them.

The interesting engineering is not the extraction. It is making the grader impossible
to cheat, and proving that with numbers.

---

# Part I — Concepts, from nothing

## I.1 How a language model is trained, in three stages

**Pre-training.** A model is shown an enormous amount of text and repeatedly asked to
predict the next word. This produces something with broad knowledge and fluent language
but no particular inclination to be useful. It will happily continue a question with
three more questions.

**Supervised fine-tuning (SFT).** The model is shown examples of the behaviour we want:
a prompt, and a good response written by a human. It learns to imitate. This is
effective and expensive, because every example costs human effort.

**Reinforcement learning (RL).** Instead of showing the model what to do, we let it try,
score its attempt, and adjust it to make high-scoring attempts more likely. This is
where the model gets genuinely better at things rather than merely more imitative,
because it can explore strategies no human demonstrated.

The third stage is where this project lives.

## I.2 The problem with scoring

RL needs a score for every attempt. Historically that score came from a *reward model*:
a second neural network trained on human preference judgements ("response A is better
than response B"). This is the R in RLHF, reinforcement learning from human feedback.

Reward models have two serious problems. They need a lot of human labelling, which is
slow and costly. And because they are themselves learned approximations, they can be
fooled — a policy can find inputs that score highly under the reward model while being
obviously bad to a human. The reward model is a proxy, and optimising hard against a
proxy eventually breaks it.

## I.3 RLVR: reward from a program, not a network

The alternative is **Reinforcement Learning with Verifiable Rewards**. Restrict yourself
to tasks where correctness can be *checked by running code*, and let that code produce
the reward. Some examples:

- Mathematics: does the final answer equal the reference answer?
- Code: do the unit tests pass?
- Structured extraction: does the emitted JSON match the known ground truth?

The reward is now a deterministic program. It costs nothing per evaluation, it never
needs a human, and it cannot be fooled in the way a neural reward model can — though, as
this project demonstrates at length, it can absolutely be fooled in *other* ways.

This is the paradigm behind the recent generation of reasoning models, and it is the
paradigm this environment is built for.

## I.4 What "an environment" actually is

Concretely, in the `verifiers` library used here, an environment is a Python package
that provides:

1. **A taskset** — something that produces a list of tasks. Each task carries a prompt
   and whatever ground truth is needed to grade it.
2. **Reward functions** — methods that look at what the model produced and return a
   number.
3. **Metrics** — like rewards, but recorded for analysis and not used for training.

The training infrastructure — sampling from the model, batching, the optimiser, the
GPUs — is somebody else's problem. You write the task, the checker, and the reward.

## I.5 Why environments are the bottleneck

Two things changed recently. Algorithms like **GRPO** (Group Relative Policy
Optimisation) removed the need for a separate value network: they sample a group of
responses to the same prompt, and use the group's own mean score as the baseline against
which each response is judged. Simpler, cheaper, and it works.

Once the algorithm is cheap and standardised, training speed is governed by how fast you
can generate *verified rollouts* — attempts with scores attached. That is exactly what
an environment produces. The bottleneck moved from algorithm design to environment
supply, and the supply is thin: writing a good one requires understanding both a task
domain and the failure modes of reward design, and very few people have done it.

## I.6 Reward hacking

This is the central concept of the project, so it is worth being precise.

**Reward hacking is when a policy finds a way to score highly that does not correspond
to doing the task.** The reward is a proxy for what you want. The policy optimises the
proxy. Any gap between the proxy and your intent is a gap the policy will find, because
finding it is exactly what optimisation does.

A canonical example: if you reward a document-extraction model partly for producing
well-formed JSON, then emitting `{"name": "", "amount": 0}` collects that reward without
reading anything. The model has not learned to extract. It has learned to type braces.

The literature confirms this is not hypothetical. Helff et al. (arXiv:2604.15149) show
RLVR-trained models abandoning general rule induction in favour of enumerating
instance-level labels that happen to satisfy the verifier, characterising it as
"imperfect verifiers that check only extensional correctness admit false positives".

A verifiable reward does not make you safe. It makes the exploit deterministic and
findable, which is better — you can go looking for it.

---

# Part II — The task

## II.1 Why Indic structured extraction

The brief offered several candidate domains. This one was chosen for four reasons.

**Coverage.** The Environments Hub has almost no Indian-language content. This was an
assumption when the task was chosen and is now a measurement: searching a Hub of 1,501
environments, `indic` returns two results -- this one and `adityapuranik/indic-ifeval`, an
instruction-following eval across 14 languages, which is a different task. `hindi`,
`tamil`, `bengali`, `marathi` and `devanagari` each return zero. An environment nobody
else has built is worth more than a fourth text-to-SQL environment.

**Deterministic verification.** Field extraction against known ground truth needs no
judge model, no similarity threshold, no human. The reward is exact.

**Reproducibility without a download.** Because the corpus can be *generated* from a
seed, the package has zero dependencies and runs offline. There is no dataset to host,
no licence to worry about, and no annotation noise — the ground truth is exact by
construction, so a disagreement between model and reference is always the model's.

**A rich exploit surface.** Multi-field extraction with partial credit is fertile ground
for reward hacking, which makes it a good vehicle for the analysis that is the real
deliverable.

## II.2 The task itself

The model receives a document in Hindi, Marathi, Tamil or Bengali — a payment demand, a
tax notice, a utility bill — and a schema. It must return one JSON object:

```json
{
  "name": "the addressee",
  "amount_inr": 58348,
  "due_date": "2026-04-21",
  "reference": "MH-REV-2026-67095"
}
```

Here is a real generated document:

```
देय दिनांक: 21/04/2026
जारी दिनांक: 15 मार्च 2026
देय रक्कम: रु. 58,348/-
संदर्भ क्रमांक: MH-REV-2026-67095
मागील थकबाकी: रु. 24,462/-
प्रति: श्री मंगला साळुंखे
मदत क्रमांक: DL/MUN-35819
अधिकारी: सुरेश कुलकर्णी
```

Eight lines. Two dates, two amounts, two reference numbers, two people. Only one of each
is the answer. The line order is shuffled.

## II.3 What makes it hard

**Discrimination, not detection.** Finding *a* date is trivial. Deciding which of two
dates is the due date rather than the issue date requires reading the label.

**Indian numbering.** Amounts group as `12,34,567` (lakh grouping), not `1,234,567`. They
may be written `₹2.4 லட்சம்` (2.4 lakh = 240,000) or `₹1.5 কোটি` (1.5 crore =
15,000,000). Converting these is a genuine capability, and — as the measurements later
show — a genuine failure mode.

**Day-first dates.** `03/04/2026` is 3 April in India, not 4 March. A model with a
US-centric prior gets this backwards, deterministically.

**Multiple digit systems.** `२०२६`, `௨௦௨௬`, `২০২৬` and `2026` are the same year in
Devanagari, Tamil, Bengali and Latin digits. Documents mix them.

**Names in two scripts.** A model may correctly answer `श्री रमेश शर्मा` or
`Shri Ramesh Sharma`. Both are right. This turns out to matter enormously — see IV.2.

---

# Part III — Architecture

Seven modules in the published package, plus research tooling that is not published.

## III.1 `lang.py` — language data

Pure data. For each of four languages: the digit glyphs, the twelve month names,
honorifics, pools of given names and surnames, currency and multiplier words, and the
labels that appear in documents.

The load-bearing part is that every name carries **its accepted romanisations**:

```python
Name("চট্টোপাধ্যায়", ("Chattopadhyay", "Chatterjee"))
```

Both spellings are in everyday use for that Bengali surname. A verifier accepting only
one is simply wrong. Because we generate the corpus, we know the full acceptable set by
construction — so the verifier can check membership in a closed set rather than resorting
to fuzzy string distance.

## III.2 `corpus.py` — the generator

Produces a `Document`: rendered text plus exact ground truth, from an integer seed.

Determinism rules it obeys: no clock reads, no `hash()` (salted per process), no
iteration over sets, and **one independently seeded RNG per task** so that generating
task 7 gives the same result whether or not tasks 0–6 were generated first. That
independence is what lets a run resume, shard across providers, or evaluate a subset
without silently producing a different corpus.

It also provides `render_variant(doc, k)`: the same ground truth re-rendered with a
different digit system, date format, amount wording, labels and line order. This is used
for invariance testing (see V.4).

## III.3 `normalize.py` — deterministic normalisation

Pure functions, standard library only. Digit folding across all Unicode decimal digits;
Indian and Western amount grouping; multiplier words; six date formats; reference
normalisation; name tokenisation with honorific stripping.

The design rule throughout is **normalise, then compare exactly**. Never fuzzy-match.
Fuzzy matching hands out partial credit for wrong answers and makes the reward a function
of a similarity threshold nobody can justify.

## III.4 `verify.py` — extraction and field checking

Two stages, deliberately separated:

- **Extraction** recovers exactly one JSON object from the raw reply, tolerating code
  fences and surrounding prose.
- **Field verification** checks each of the four values, returning a per-field verdict
  with a *reason* string.

The reasons are not decoration. `day-month-transposed` is a locale bug;
`picked the rival date` is a comprehension failure; `unparseable` is a formatting
failure. These need different fixes, and an aggregate accuracy number hides all three.

## III.5 `score.py` — the reward

Kept separate from the taskset so it can be tested on any platform and so there is one
definition of the reward rather than two that drift. Discussed in full in Part IV.

## III.6 `heuristics.py` — the shortcut baselines

Policies that answer *without reading*: pick the largest amount, pick the latest date,
take a fixed line position, emit an empty schema. These audit the corpus. If a shortcut
scores well, the environment is measuring the shortcut rather than the task.

## III.7 `taskset.py` — the verifiers v1 interface

Exports one `Taskset` class via `__all__`. Three rewards, five metrics, and a model-free
`validate()` hook that feeds each row's own gold answer back through the real scoring
path — catching rows the verifier cannot grade, which would otherwise look like permanent
model failures.

## III.8 `research/` — not published

The rollout runner (caching, budget, retry), the evaluation driver, the adversarial
policies, the ablation rewards, and the analysis scripts. Deliberately outside the
published wheel: hub users want the environment, not my experiment harness.

---

# Part IV — Design decisions and why

## IV.1 Build against v1, not v0

The brief described the `verifiers` API as `SingleTurnEnv` / `Rubric` / `Parser` /
`load_environment()`. Checking the library rather than trusting the description showed
that is **v0, and deprecated** — the upstream docs say it "will be fully removed in a
future release". Current v1 uses `Taskset` / `Task` / `@vf.reward`, and explicitly
forbids `load_environment()`.

The whole environment is built against v1. Had I trusted the brief, I would have shipped
against an API scheduled for deletion.

## IV.2 Accept both scripts for names

During early probing, the *same model* at two reasoning settings returned:

- `reasoning_effort=low` → `"श्री रमेश कुमार शर्मा"`
- `reasoning_effort=medium` → `"Shri Ramesh Kumar Sharma"`

Both are correct extractions. A naive exact-match verifier scores the second as zero,
and would therefore report that raising reasoning effort *hurt* accuracy — measuring
script preference and calling it comprehension.

The fix: every name carries its accepted romanisations, and honorifics are stripped from
both sides in both scripts. This is enumerable precisely because the corpus is generated.

## IV.3 Exactly one reward term may be positive

This is the single most important decision in the project.

```
reward = 1.00 × field_accuracy      (positive, requires reading)
       − 0.25 × format_violation    (penalty, bounded [0,1])
       − 0.10 × verbosity           (penalty, bounded [0,1])
```

Both penalties have a *best possible contribution of zero*. Nothing except being correct
can raise the score. The consequence is that a terse, perfectly formatted, contentless
reply scores the penalty floor and can never compete with reading the document.

The alternative — a positive bonus for schema conformance — is the natural first
implementation and is measurably exploitable. See V.1.

## IV.4 Partial credit, but only because it was earned

`field_accuracy` gives 0.25 per correct field rather than all-or-nothing. Partial credit
gives a denser learning signal, but it is *only safe if no field can be guessed for
free* — otherwise it pays for guessing.

That is not assumed here. It is measured by the shortcut baselines and pinned by a test.
The first version of the corpus failed this badly (V.2).

## IV.5 Grade the first JSON object, never the best one

If the reply contains several JSON objects, the verifier grades the first and records
the count. It never searches for the best-matching one.

Searching would convert the task into a multiple-choice quiz the model writes for
itself: emit one object per plausible reading and let the grader do the extraction. The
count feeds the format penalty instead.

## IV.6 Reject collection values

A field whose value is a list scores zero, even if the truth is in the list. Crediting
membership turns "which of these is the due date" into "list the dates", which removes
the entire task.

## IV.7 Count tokens, not characters

Discussed in V.3 — this one was wrong first and fixed by measurement.

## IV.8 Cache every rollout; never cache verification

Generation is expensive and metered. Verification is free. Keying the cache on
everything that affects the model's output means changing the verifier and re-running
costs nothing, which is what made it practical to iterate on scoring under a free tier.

---

# Part V — The decisions that were wrong

This is the most useful part of the document. Each of these was believed correct,
shipped, and then disproved by measurement.

## V.1 Wrong: difficulty from *fewer distractors*

**What I built.** Three difficulty tiers where "easy" documents had no competing
candidates at all, "medium" had two, and "hard" had four. Difficulty came from how many
distractors were present.

**How it failed.** Running the shortcut baselines against 600 documents:

| Policy | Field accuracy | Exact match | `name` field |
|---|---:|---:|---:|
| `largest_amount` | **0.717** | **0.327** | **1.000** |
| `first_of_each` | 0.660 | 0.145 | 1.000 |
| `positional` | 0.416 | 0.067 | 1.000 |

Two separate disasters. The `name` field scored **1.000** for every shortcut, because I
had pinned the addressee to line 0 and shuffled only the remaining lines — "take the
first line" was a perfect name extractor. And because easy and medium documents had no
rival candidates, those fields were free on two-thirds of the corpus. A policy that read
nothing scored 0.327 exact match.

**Why it was wrong in principle.** Difficulty should come from the *surface* being hard
to read, not from the answer being unopposed. With no rival, a field tests detection, not
comprehension — and detection is trivial.

**The fix.** Every field gets exactly one rival at every tier, each positioned to make
the corresponding shortcut a coin flip:

- the rival amount is drawn **log-uniformly around the target**, so it is larger exactly
  half the time and "pick the largest" is 50/50 by construction rather than by a guessed
  range;
- the rival date falls *after* the due date half the time, so neither "latest" nor
  "earliest" is informative;
- the addressee sits behind a label like everyone else and is shuffled with the rest;
- the rival reference uses the same format family, so format-sniffing narrows to two and
  no further.

**After:**

| Policy | Field accuracy | Exact match | `name` field |
|---|---:|---:|---:|
| `largest_amount` | 0.463 | **0.060** | 0.520 |
| `first_of_each` | 0.416 | 0.018 | 0.520 |
| `positional` | 0.085 | 0.000 | 0.000 |

With one rival per field, chance is 0.5 per field and 0.5⁴ = 0.0625 exact. The best
shortcut now sits at 0.060 — the chance floor. `tests/test_corpus_is_not_shortcuttable.py`
pins these ceilings, including per-tier, so the regression cannot return quietly.

## V.2 Wrong: a character-based length penalty

**What I built.** A verbosity penalty counting characters, with a comment asserting that
Indic scripts cost "roughly twice as many tokens per character as Latin".

**How it failed.** I had never measured it. When I did — differentially, subtracting the
chat-template overhead, which is 71 tokens on gpt-oss-120b and 35 on llama-3.1-8b and
large enough to make short samples appear to cost more tokens than they have characters:

| Script | gpt-oss-120b | llama-3.1-8b |
|---|---:|---:|
| Devanagari | 1.16× | 1.94× |
| Bengali | 1.44× | 4.20× |
| Tamil | 1.66× | **4.52×** |

*(tokens per character, relative to Latin, for identical content)*

My stated figure was wrong in both directions: too high for Devanagari on one tokeniser,
far too low for Tamil on the other.

**Why it matters.** This environment accepts an answer in the source script *or* in
Latin transliteration, and the correctness term cannot distinguish them. A
character-denominated budget therefore charges two identical-in-meaning answers at rates
differing by up to 4.5×. That is a script preference expressed through the length term,
invisible in the correctness term, and unsatisfiable by the model except by switching
scripts.

**The fix.** Denominate the penalty in tokens the provider actually billed. The bias
disappears by construction.

## V.3 Wrong: the token budget itself

**What I built.** `TOKEN_BUDGET = 160`, reasoned from "a four-field JSON answer is well
under that".

**How it failed.** Measuring 400 real rollouts: gpt-oss-120b answers *correctly* in
about 197 output tokens. So **170 of 199 correct rollouts were being charged a penalty**,
at a mean cost of 0.0068 reward.

That is not a guardrail, it is a tax on correct behaviour — noise applied to exactly the
answers that should score full marks.

**The fix.** `TOKEN_BUDGET = 256`, above measured correct-answer length for every model
evaluated (llama 68, gemini 76, gpt-oss 197), while still biting hard on the
`correct_but_padded` attack at ~670 tokens. Mean reward for gpt-oss rose 0.969 → 0.975
with accuracy unchanged — the difference was pure noise removal.

The general lesson: **a reward term that fires on the majority of correct answers is
miscalibrated**, regardless of how sensible its threshold looked on paper.

## V.4 Wrong: assuming the verifier was surface-invariant

**What I assumed.** That because normalisation was thorough, the verdict could not depend
on how a document happened to be typeset.

**What changed my mind.** Helff et al. (arXiv:2604.15149) propose *isomorphic
perturbation testing*: evaluate under transformations that preserve the logical content,
because genuine competence is invariant and shortcuts are not. I had no such test.

**The fix.** `render_variant(doc, k)` re-renders a document with identical ground truth
and a completely redrawn surface. `tests/test_surface_invariance.py` asserts that the
gold answer verifies identically across variants, that variants genuinely differ, and —
importantly — that no *single surface* makes a shortcut policy reliable. Aggregate
shortcut resistance can conceal one rendering on which a trivial rule works perfectly,
and a training run samples every rendering.

The verifier passed. But it passed *demonstrably*, which it did not before.

## V.5 Wrong: a token ceiling that silently truncated correct answers

**What I built.** `max_tokens=700` for every rollout, reasoned from the fact that a
four-field JSON answer needs perhaps eighty tokens.

**How it failed.** Two of forty gemini-3.6-flash rollouts produced no parseable JSON and
were scored as comprehension failures. They were nothing of the kind:

```
idx=22  prompt 266  completion 26  total 962  -> hidden reasoning 670
        reply: '```json
{
  "name": "திரு கார்த்திக் சுப்ரமணியன்",
  "amount_inr'

idx=26  prompt 251  completion 59  total 947  -> hidden reasoning 637
        reply: '...
  "reference": "UP/GST'
```

Both are correct answers cut off mid-field. The cause is that **`max_tokens` is shared
between hidden reasoning and the visible answer**. A ceiling that looks generous for the
answer is not, because the invisible portion is charged against the same budget: 670
tokens of thinking left 26 for the reply.

**Why it matters more than the two data points.** I was attributing my own configuration
error to the model, and doing so invisibly — a truncated reply and a wrong answer are
indistinguishable once both are "no JSON". Every accuracy number for a reasoning model
was therefore a lower bound of unknown tightness.

**The fix.** Three parts. The ceiling rose to 1400, which clears the largest reasoning
burst observed on this task (736) with room for an answer. `finish_reason` is now
captured, cached and exposed as `RolloutResult.truncated`. And the evaluation reports a
truncation rate as its own line, stating explicitly that a non-zero value means the
numbers understate the model. The corrected Gemini run is blocked on free-tier quota and
has not been repeated, so the 0.950 in Part VII stands with that caveat attached.

## V.6 Wrong: assuming a successful build meant a publishable package

**What I believed.** That if `uv build` produced a wheel, the wheel installed into an
empty environment, and the taskset loaded from it, the package was ready to publish. I
had verified all three, and they are all genuine evidence.

**How it failed.** The push to the Environments Hub succeeded. The CLI printed a
checksum and a URL. Nothing local had anything to say about it. Some minutes later an
email arrived saying an action on the environment had failed, and the Hub's own logs
gave the reason:

```
assert "tags" in pyproject["project"], "pyproject.toml does not have tags"
AssertionError: pyproject.toml does not have tags
```

The Hub runs its own integration suite against every published artefact, and it asserts
on a `tags` key inside `[project]`. `tags` is **not a PEP 621 field**. The standard one
is `keywords`, which is what I had used. Standard build backends ignore unknown keys in
that table entirely, so `keywords` builds cleanly, installs cleanly, imports cleanly --
and fails the registry's check.

**Why it is worth a section.** The general lesson is that *a build backend accepting
your file is not the same as the registry accepting it*, and the gap between them is
invisible to every local check. Three of the Hub's four tests passed, including
`test_install_and_import` -- which is itself a useful confirmation, because it ran on
Python 3.12 with `verifiers` absent and succeeded only because the taskset export is
resolved lazily (see the packaging discussion in Part III). An eager import would have
failed their CI outright for a completely different reason.

**The fix.** `tags` declared with the four language names, since Indic coverage is the
discoverability argument; `[tool.verifiers.eval]` defaults added to match what published
environments do; and three tests in `tests/test_packaging.py` that mirror the Hub's
metadata assertions locally, so this class of failure surfaces from `pytest` rather than
from an email. Version 0.1.1 passed the Hub's integration test.

## V.7 Wrong: trusting a development environment to represent a consumer's

**What I believed.** That a green test suite meant the package worked. It had passed on
Windows and Linux, from a wheel installed into an empty environment, and through the
Hub's own CI.

**How it failed.** Cloning the pushed repository from GitHub into a fresh environment --
which is what any consumer actually does -- produced four test failures immediately:

```
AttributeError: 'IndicExtractionTaskset' object has no attribute 'select'
```

`verifiers` had released 0.3.0. `pyproject.toml` asks for `>=0.2.1`, so the fresh install
resolved to the new release, while the development environment had been sitting on 0.2.1
for days and passing. 0.3.0 replaced `Taskset.select(n)` -- a list -- with
`Taskset.head(n)`, a lazy view.

**What the failure was, and was not.** The environment itself was fine. Checked directly
against 0.3.0: the taskset constructs, `load()` yields, `head(3)` returns three tasks,
every reward and metric resolves, `validate()` passes, and the lazy `__all__` export
works. Only the *tests* named the removed method. Making `load()` a generator turned out
to be what carried the package across the version boundary, since both APIs need exactly
that.

**A second, subtler thing the fix got wrong first.** The obvious repair is
`taskset.head(n).load()`. That is wrong, and it is wrong quietly: in 0.3.0 `__iter__` is
the read path and applies the view's transform, while `load()` is the raw subclass hook.
Calling `.load()` on a view therefore *bypasses* the `islice` -- `head(5).load()` returned
500 documents. The only test that caught it was the laziness assertion, because it is the
one that asks for far fewer tasks than the taskset holds; every test where the requested
count equalled `num_tasks` passed either way and proved nothing.

**The general lesson.** A development environment pins silently the moment it is created;
every new consumer gets the latest release. "The tests pass here" is evidence about here.
The check that matters is a clone of the pushed artefact into an environment resolved
today, and it is the one check that had never been run.

## V.8 Wrong: fixing a default in one place and leaving a copy in another

**What I did.** Raised `RolloutRequest.max_tokens` from 700 to 1400 after discovering
that hidden reasoning was truncating replies (V.5), added `finish_reason` capture, added
a truncation rate to the report, and wrote two regression tests. A thorough fix.

**What I missed.** `evaluate.py` declared its own `--max-tokens` default of 700 and
passes it explicitly to every request, overriding the dataclass entirely. So when quota
freed up and I re-ran Gemini expressly to get a clean measurement, the run used 700. It
therefore hit the same cache keys as before, re-served the two previously-truncated
documents from cache, and reported `truncated_rate: 0.0` -- because those cached entries
predate `finish_reason` and read as untruncated.

Every visible signal said the corrected ceiling was in use. The committed config said
otherwise: `'max_tokens': 700`.

**Why it survived the fix and the tests.** Both regression tests asserted on
`RolloutRequest`, which was correct. Neither asserted that the thing actually issuing
requests used it. A test that checks the definition and not the call site verifies the
half that was never broken.

**The fix.** `evaluate.py` reads `RolloutRequest.max_tokens` as its default instead of
restating a number, and a test now fails if a second literal reappears. The general
shape: a constant with two homes has no owner, and the copy that matters is whichever one
the caller actually passes.

**What it cost.** One wasted day of Gemini quota, and a paragraph in this report that
briefly claimed a corrected measurement I did not have.

## V.9 Wrong: three infrastructure bugs

**Word boundaries after Indic tokens.** `parse_amount("रु. 12500/-")` returned `None`.
Python's `\b` is defined via `str.isalnum()`, which is `False` for combining marks
(category `Mn`), so there is no boundary between the vowel sign in `रु` and the following
`.`. The assertion never matched, leaving a stray period that broke the numeric parse.
Word-boundary assertions are unreliable at the end of Indic tokens.

**Honorifics stripped in one script only.** `श्री` was removed but `Shri` survived, so a
model that transliterated the *whole* name including the honorific was compared against a
token list one element too long — penalising precisely the cross-script answer IV.2
exists to accept.

**Budget ledger clobbering.** Two evaluation processes ran concurrently, one per
provider. Each loaded the whole ledger at startup and wrote it back wholesale on every
charge, so the Groq process kept restoring its stale copy of the Google total and erasing
the other process's spend. The tracker reported ample Google headroom while real quota
drained. Fixed by re-reading immediately before each merge and replacing the file
atomically.

Each of these has a regression test whose docstring describes the failure, not just the
fix.

## V.10 Wrong: assuming API parameters are universal

`reasoning_effort="low"` cuts gpt-oss-120b's hidden reasoning from 366 tokens to 9 — a
5.5× cost reduction that made the free-tier budget viable. I applied it globally.

`llama-3.1-8b-instant` rejects it outright: `reasoning_effort is not supported with this
model`, a hard 400 on **100 out of 100 rollouts**. It looked like a blanket API outage.
It was one parameter.

Model capabilities are now declared per model, and the parameter is dropped before the
cache key is computed so a model that ignores it does not get two cache entries.

## V.11 Wrong: assuming NFC was enough to normalise Indic text

**What I believed.** That NFC normalisation plus digit folding and whitespace collapse
made two visually identical strings compare equal. NFC is the standard answer to Unicode
equivalence and it handles the composed/decomposed problem this corpus actually
generates, so it looked complete.

**How it failed.** NFC does not remove zero-width characters, and Indic scripts use them
for real. Measured:

| Case | Compared equal? |
|---|---|
| `रमेश शर्मा` vs the same name carrying ZWNJ (U+200C) | **No** |
| `क्ष` vs `क्` + ZWJ (U+200D) + `ष` | **No** |
| Bengali `অন্য` vs the same with ZWNJ | **No** |
| ZWSP, BOM, soft hyphen, word joiner | all survived normalisation |

ZWNJ and ZWJ control conjunct formation in Devanagari and Bengali. A model emitting one
produces output a reader cannot distinguish from output without it, and was scored as a
comprehension failure for the difference.

**Why it matters beyond the three rows.** It is the same defect as the honorific bug and
the romanisation bug: penalising a correct answer for an invisible presentation choice.
That is the single thing this verifier exists not to do, and I had shipped a third
instance of it. The lesson is that "we normalise Unicode" is not a property; NFC is one
specific equivalence and the ones it does not cover have to be named.

**The fix.** Strip Unicode category Cf plus ZWSP in `normalise_text`, before digit
folding so an invisible cannot split a token. Ten tests pin it. The defect was latent --
the generator never emits these characters -- so no committed number changed, which is
itself worth stating: it would have fired the moment anyone extended the corpus with real
scanned or web-sourced documents.

## V.12 Wrong: reporting point estimates from a single seed

**What I built.** Every accuracy in this report, quoted as a bare number. `0.920`.
`0.620`. `0.950`. All from `seed=1`, all with no interval.

**Why that is unsound.** Each of those is a proportion estimated from a finite sample.
Without an interval it looks like a measurement and is not one, and without a second seed
it cannot distinguish *how precisely this corpus was measured* from *how much the answer
depends on which corpus was drawn*.

The specific danger showed up immediately on our own data. Computing the naive
normal-approximation interval for `gemini-3.6-flash` at 38 of 40 correct gives
**[0.882, 1.018]** -- an upper bound above 100 per cent accuracy. That is exactly the
small-sample failure documented in *Position: Don't Use the CLT in LLM Evals With Fewer
Than a Few Hundred Datapoints* (arXiv:2503.01747). Wilson score intervals invert the
score test instead of assuming normality and stay inside [0, 1] at any sample size:
[0.835, 0.986].

**A second error the intervals exposed.** With intervals attached, `gemini-3.6-flash`
[0.835, 0.986] **overlaps** `gpt-oss-120b` [0.874, 0.950]. The earlier write-up presented
all three models in one table as though they were rankable. They are not, on this
evidence, and the claim had to be softened rather than merely annotated.

**And a better comparison.** Every model in a run sees the identical documents, so
comparing two marginal intervals throws the pairing away. McNemar's exact test on the
per-document outcomes uses only the documents where the models disagree, which is where
all the evidence lives. On 200 documents `gpt-oss-120b` beats `llama-3.1-8b-instant` on
**63** documents and loses on **3**, p = 1.3e-15 -- far stronger than "the intervals do
not overlap", and it would still hold if they did.

**The fix.** `research/statistics_.py` supplies Wilson intervals, exact McNemar and
seed-variance summaries; `research/evaluate.py` reports all three, and tests assert the
headline table cannot print a proportion without its interval and its `n`. A single seed
reports its standard deviation as "not measured" rather than as `0.0`, because one
observation has no spread and claiming zero asserts the opposite.

---

# Part VI — The reward-hacking analysis

## VI.1 Method

The brief asks for exploits found via GRPO rollouts. Training even a small model needs a
GPU, and this project was built for $0. The substitute is a two-part offline analysis,
and I want to be clear that it *is* a substitute.

**Adversarial policies** — replies engineered to maximise reward without doing the task,
written to be maximally effective rather than realistic. The question is not "would a
model stumble into this" but "what is the ceiling for a policy that optimises the reward
instead of the task".

**Ablation rewards** — four alternative reward designs this environment could plausibly
have shipped, three of which are the *default* first implementation. Scoring each attack
under each design gives the counterfactual: what the exploit would have earned without
the mitigation.

Claiming a mitigation matters is cheap. Showing what an attacker earns without it is not.

## VI.2 Results

400 documents, fully offline and deterministic:

| Attack | Under the naive reward | Under shipped | Removed |
|---|---:|---:|---:|
| `empty_schema` | 0.300 *(format bonus)* | **0.000** | 0.300 |
| `shotgun_objects` | 0.734 *(lenient extractor)* | **0.245** | 0.489 |
| `shotgun_lists` | 0.846 *(membership credit)* | **0.000** | 0.846 |
| `key_stuffing` | 0.467 *(lenient extractor)* | 0.384 | 0.083 |

Honest extraction scores 1.000.

**Exploit 1 — the free floor.** A reward paying 0.3 for schema conformance gives 0.300 —
23% of full marks — to a reply containing the right keys and no information. Mitigation:
only correctness may be positive. Floor removed entirely.

**Exploit 2 — grader-assisted extraction.** Emitting one object per candidate combination
scores 0.734, 73% of honest extraction, with no discrimination performed. Mitigation:
grade the first object and penalise multiplicity. Cut to 0.245.

**Exploit 3 — enumeration.** Listing every candidate per field scores 0.846, 85% of
honest extraction — and discriminating between those candidates is the entire task. This
is precisely the failure Helff et al. describe: an extensional check admitting a false
positive. Mitigation: require a scalar. Cut to exactly zero.

**Calibration.** `correct_but_padded` — the right answer buried in filler — scores 0.907:
below honest extraction, far above every attack. The length penalty discourages padding
without ever letting a guess beat a correct answer.

## VI.3 What this analysis cannot tell you

Best-of-n and hand-written attacks explore the space I thought to explore. A GRPO run
explores the space the *model* finds, which is not the same space and is the reason
reward hacking keeps surprising people. This analysis raises confidence; it does not
establish safety. The honest claim is: three specific exploits were found, measured, and
closed, and the environment is now instrumented so a fourth would show up as a metric
moving rather than as a silently inflated score.

## VI.4 A caution from the literature

*Reward Bias Substitution* (arXiv:2605.27996) argues that suppressing one reward bias
tends to redirect optimisation pressure onto correlated proxies rather than eliminate it
— "removing a feature does not remove the pressure to exploit it" — and that this is
invisible to single-axis evaluation.

Taking that seriously, I checked whether the length penalty had displaced pressure onto
accuracy, by bucketing real rollouts by output length:

| Output tokens | n | Field accuracy |
|---|---:|---:|
| 101–160 | 29 | 1.000 |
| 161–300 | 167 | 0.978 |
| >300 | 3 | 1.000 |

No relationship. But the more durable response to that paper is structural: the taskset
records five metrics beyond the reward — exact match, schema cleanliness, day/month
transposition, romanised answers, output tokens — so that displaced pressure has
somewhere to become visible.

---

# Part VII — Measured findings

## VII.1 Model comparison

Intervals are Wilson score at 95%, not the normal approximation, for the reason set out
in V.10. They cover sampling error over documents; corpus-seed variance is reported
separately in VII.2.

| Model | Exact match (Wilson 95%) | Field accuracy | Mean reward | Output tokens |
|---|---|---:|---:|---:|
| `openai/gpt-oss-120b` | **0.920** [0.874, 0.950] (n=200) | 0.976 | 0.975 | 197 |
| `llama-3.1-8b-instant` | **0.620** [0.551, 0.684] (n=200) | 0.897 | 0.897 | 68 |
| `gemini-3.6-flash` | **0.967** [0.888, 0.991] (n=61) | 0.967 | 0.962 | 73 |

**The two fully measured models are cleanly separated, and the right evidence for that is
not the intervals.** All models see the identical corpus, so the comparison is paired.
McNemar's exact test on the per-document outcomes: `gpt-oss-120b` wins **63** documents
that `llama-3.1-8b-instant` loses, and loses **3**, p = 1.3e-15. That conclusion rests on
the 66 documents where the two disagree, and would hold even if the marginal intervals
overlapped -- which is precisely why comparing marginals is the weaker test.

**`gemini-3.6-flash` cannot be ranked against the other two on this evidence.** Its
interval, [0.835, 0.986], overlaps `gpt-oss-120b`'s [0.874, 0.950]. It was measured on 40
documents rather than 200 because the free tier stalled (VII.4), and two of its failures
were replies my own token ceiling truncated rather than wrong answers (V.5), so 0.950 is
a floor. Three separate reasons to treat that row as a partial measurement rather than a
result, and it is listed here with all three attached rather than alongside the others as
though they were comparable.

An earlier 18-document sample of the same model reported a clean 1.000, which remains the
most useful thing in this section: n=18 is not a measurement, and a bare point estimate
gave no way to see that.

## VII.2 How much the answer depends on which corpus was drawn

Five independently seeded corpora, 100 documents each, same two models:

| Model | seed 1 | 2 | 3 | 4 | 5 | mean | range | sd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-oss-120b` exact | 0.920 | 0.920 | 0.950 | 0.950 | 0.978 | 0.944 | 0.058 | 0.024 |
| `llama-3.1-8b` exact | 0.580 | 0.620 | 0.690 | 0.660 | 0.670 | 0.644 | 0.110 | 0.044 |
| `gpt-oss-120b` field | 0.980 | 0.980 | 0.988 | 0.988 | 0.994 | 0.986 | 0.014 | 0.006 |
| `llama-3.1-8b` field | 0.890 | 0.892 | 0.910 | 0.910 | 0.905 | 0.902 | 0.020 | 0.010 |

Three things follow.

**Corpus choice moves exact match by up to eleven points.** `llama-3.1-8b` scores 0.580 on
one corpus and 0.690 on another, with nothing changed but the seed. That is larger than
many differences reported as meaningful in model comparisons, and it is invisible in any
single-seed result.

**Seed 1 is the minimum of the range for both models.** Every number this project
committed before this section existed came from seed 1 alone. They were not wrong, and as
it happens they were conservative rather than flattering -- but they were
*under-determined*, and nothing in the earlier write-up let a reader see that.

**Exact match is far more seed-sensitive than field accuracy.** sd 0.044 against 0.010 for
`llama-3.1-8b`. That is arithmetic rather than a surprise -- exact match is a conjunction
of four field decisions, so it amplifies whatever moves the underlying fields -- but it
means the headline metric is precisely the one most in need of several seeds.

The paired comparison survives all of it. On every seed where both models covered the
same documents, `gpt-oss-120b` beat `llama-3.1-8b-instant` decisively: discordant counts
of 36/2, 31/1, 28/2 and 30/1, with p between 5e-9 and 9e-7. A conclusion that holds across
five independently drawn corpora is worth considerably more than one drawn from one.

Seed 5 is missing from that list, and the reason is worth recording. Eleven transient API
errors left `gpt-oss-120b` with 89 documents against `llama`'s 100, and the reporting code
**refused to run a paired test on mismatched samples** rather than silently truncating to
the shorter vector -- which would have compared the two models on different corpora and
labelled the result paired. Declining to compare is the correct output there.

## VII.3 Per-field

| Model | name | amount_inr | due_date | reference |
|---|---:|---:|---:|---:|
| `openai/gpt-oss-120b` | 0.995 | 0.995 | 0.945 | 0.970 |
| `llama-3.1-8b-instant` | 0.990 | 0.870 | 0.755 | 0.975 |

`due_date` is the hardest field for both, by a wide margin.

## VII.4 Why the failures happen

Classified against the specific distractor placed in each document:

| Failure | gpt-oss-120b | llama-3.1-8b |
|---|---|---|
| due_date — picked the rival date | 10 / 10 (100%) | 46 / 49 (94%) |
| due_date — day/month transposed | 0 | 0 |
| amount — lakh/crore magnitude error | 0 | 10 / 26 (38%) |
| amount — picked a rival amount | 0 | 6 / 26 (23%) |

Two findings worth stating plainly.

**Not one date failure in 400 rollouts was a parsing error.** Every single one was
choosing the wrong line. The environment is measuring distractor discrimination, exactly
as designed — and the day/month-transposition trap I built, expecting it to fire, never
did.

**llama-3.1-8b has a specific, nameable capability gap the larger model does not:**
reading `7 லட்சம்` as 70,00,000 instead of 7,00,000. Ten of its 26 amount errors are
clean powers of ten. Reported as "wrong amount" this would be invisible; isolated as its
own error class it is an actionable finding about Indian-numbering conversion.

## VII.5 Operating notes from the free tier

- **Hidden reasoning dominates cost.** Left at defaults, gpt-oss-120b spent 366 of 400
  output tokens on reasoning; gemini-3.6-flash spent 736 of 792. `reasoning_effort=low`
  cut the former to 9.
- **Gemini hides reasoning in the total.** A probe returned prompt 11, completion 4,
  total 154. Budgeting on `prompt + completion` would undercount real consumption
  tenfold. Account in `total_tokens`.
- **Cloudflare blocks Python's default user-agent.** `api.groq.com` returns `403 error
  1010` to `urllib`, which looks exactly like an auth failure. The same key works with a
  normal user-agent. Diagnosed with a deliberately-invalid-key control that returned a
  clean 401.
- **Gemini's free tier is far tighter than 10 rpm.** 82 of 100 rollouts returned
  `RateLimitError`. Google no longer publishes free-tier limits; they are per-project and
  visible only in AI Studio. Reduced to 5 rpm with bounded exponential backoff.
- **`verifiers.v1` cannot be installed on Windows** — it imports `fcntl`. The
  deterministic core avoids the dependency entirely and the taskset export is resolved
  lazily, so development works anywhere and only rollouts need POSIX.

## VII.6 Publishing to the Hub

Published as `sagar2907/indic-extraction-v1@0.1.3`. The wheel is 33 KB, the source
archive 80 KB, and the package installs back out of the registry into a clean
environment.

Two things about the process are worth recording. The Hub runs its own integration suite
against every push -- pull the artefact, install it, import it, check the metadata -- and
that suite is the real acceptance gate, not the upload. Version 0.1.0 uploaded
successfully and failed it, for the reason in V.6. And uploads should be made from a
clean export rather than the working tree: `git archive HEAD` guarantees that the
virtualenv, the rollout cache and the gitignored `.env` cannot be swept into a public
registry, which is a much better guarantee than remembering to check.

Neither the push nor the two CI runs incurred any charge. The account's wallet shows a
zero balance and no billing rows, and the API token used carries `billing: read` without
write, so it is structurally incapable of spending.

## VII.7 What this cost

**$0.** Roughly 620 live rollouts across three models, entirely within free tiers. At
published rates the same work would have cost approximately $0.60 on Groq's
`gpt-oss-120b`. The brief's ~$40 estimate is essentially all GPU rental for a training
run that was not performed; inference for a project this size is a rounding error.

---

# Part VIII — Running and extending

## VIII.1 Running

```bash
uv pip install -e ".[dev]"
python -m pytest -q          # 190 tests, fully offline, no keys
```

Live evaluation needs keys in a gitignored `.env`:

```bash
python -m research.evaluate --n 200 --seed 1 --models openai/gpt-oss-120b
```

Offline analyses, all free:

```bash
python -m research.hack_report --n 400
python -m research.failure_analysis --n 200
python -m research.script_bias
```

## VIII.2 Extending

**Adding a language** is a data change, not a code change: add a `Language` to `lang.py`
with its digits, months, honorifics, name pools with romanisations, and labels.

**Adding a field** means extending `FIELDS`, `Record`, a `_CHECKS` entry, and the prompt
schema. Add a rival candidate for it in `_compose`, or the shortcut tests will —
correctly — start failing.

**Changing the reward** should be done in `score.py`, and the tests in `test_score.py`
and `test_reward_hacking.py` will tell you if the change reopens an exploit. That is
their purpose; if one fails, read its docstring before weakening the assertion.

---

# Part IX — Limitations

1. **No GRPO training run.** The exploit analysis is adversarial and ablation-based, not
   gradient-based. It explores the space I thought of.
2. **gemini-3.6-flash measured on 61 documents**, not 100. The free tier stalls against
   an apparent daily request cap that backoff cannot clear; two attempts on separate days
   reached 40 and 61. Reported as 61, not extrapolated — and 0.967 is a floor, because
   both failures in the sample are harness truncation rather than wrong answers. On the
   other 59 documents it got every field right.
3. **Synthetic documents.** Templated from hand-written language data, not scraped. They
   are structurally correct; no native speaker has reviewed them for naturalness.
4. **Four languages of the twenty-two scheduled** in India, and one document genre.
5. **Published privately, not publicly.** The environment is live on the Hub and its
   integration test passes, but visibility is PRIVATE pending an explicit decision to
   make it public. Nothing technical blocks the change.
6. **Cross-script name acceptance is untriggered in the headline runs.** All models
   answered in the source script at `reasoning_effort=low`. The code path exists because
   the same model at medium effort behaved differently, but the headline numbers do not
   exercise it.
7. **Two models carry the analysis.** Conclusions about tokeniser behaviour rest on two
   tokenisers.

---

# Part X — Glossary

**Ablation** — removing or altering one component to measure what it contributed.

**Chat template** — the fixed scaffolding a provider wraps around your message. Adds
tokens (71 on gpt-oss-120b, 35 on llama-3.1-8b) and confounds naive token measurement.

**Environment** — a task generator plus an automatic grader, packaged for RL training.

**Exact match** — all four fields correct. Stricter and more discriminative than field
accuracy.

**Extensional correctness** — checking that an output matches the expected answer,
without checking that it was produced for the right reason. Admits false positives.

**Field accuracy** — fraction of the four fields correct; the partial-credit signal.

**GRPO** — Group Relative Policy Optimisation. Samples a group of responses per prompt
and uses the group's mean score as the baseline, removing the need for a value network.

**Isomorphic perturbation testing** — evaluating under transformations that preserve
logical content. Genuine competence is invariant; shortcuts are not.

**Lakh / crore** — Indian numbering units. 1 lakh = 100,000; 1 crore = 10,000,000.
Digits group as `12,34,567`.

**NFC** — a Unicode normalisation form. The same visible Indic string can be encoded
composed or decomposed; these are unequal under `==` and must be normalised before
comparison.

**Post-training** — everything done to a model after pre-training: SFT, RLHF, RLVR.

**Reward hacking** — scoring highly without doing the task, by exploiting the gap between
the reward and the intent.

**RLVR** — Reinforcement Learning with Verifiable Rewards. Reward comes from running a
program, not from a learned reward model.

**Rollout** — one attempt: prompt in, response out, score attached.

**Taskset** — in verifiers v1, the class that loads and produces tasks.

**Tokeniser** — the component splitting text into model units. Indic scripts cost
substantially more tokens per character than Latin, and the ratio varies by model.

**Verifier** — the program deciding whether a response is correct.

---

*Repository: `indic-extraction-v1`. 190 tests, ruff clean, zero runtime dependencies.*
