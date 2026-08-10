"""Budget accounting, including the concurrency failure that motivated the re-read.

No network and no real clock dependence beyond "today is one day": every assertion is
about the ledger's arithmetic and its behaviour under interleaved writers.
"""

from __future__ import annotations

import json

import pytest

from research.rollout import Budget, BudgetExhausted, RolloutRequest


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "budget.json"


def test_charges_accumulate(ledger) -> None:
    budget = Budget(path=ledger)
    budget.charge("groq", 100)
    budget.charge("groq", 250)
    assert budget.spent("groq") == 350


def test_remaining_never_goes_negative(ledger) -> None:
    budget = Budget(path=ledger)
    budget.charge("groq", 10_000_000)
    assert budget.remaining("groq") == 0


def test_require_raises_when_exhausted(ledger) -> None:
    budget = Budget(path=ledger)
    budget.charge("groq", 10_000_000)
    with pytest.raises(BudgetExhausted):
        budget.require("groq", 1)


def test_concurrent_writers_do_not_clobber_each_other(ledger) -> None:
    """Regression: one provider's process erased another provider's spend.

    Two evaluation processes ran at once, one per provider. Each loaded the entire
    ledger at startup and wrote it back in full on every charge, so whichever process
    wrote last restored its own stale copy of the other's total. The Google counter sat
    frozen while real quota drained, which is precisely the situation the budget is
    supposed to make impossible.

    Simulated here with two Budget instances that were both constructed before either
    wrote -- the same stale-snapshot condition, without needing real processes or any
    timing dependence.
    """
    first = Budget(path=ledger)
    second = Budget(path=ledger)  # constructed while the file is still empty

    first.charge("groq", 500)
    second.charge("google", 700)
    first.charge("groq", 500)

    on_disk = json.loads(ledger.read_text(encoding="utf-8"))
    today = next(iter(on_disk))
    assert on_disk[today]["groq"] == 1000
    assert on_disk[today]["google"] == 700, "the other provider's spend was erased"


def test_ledger_is_written_atomically(ledger) -> None:
    """No temp files may survive a charge, and the result must be valid JSON."""
    budget = Budget(path=ledger)
    budget.charge("groq", 42)
    assert json.loads(ledger.read_text(encoding="utf-8"))
    leftovers = list(ledger.parent.glob("*.tmp"))
    assert leftovers == []


def test_cache_key_separates_repeated_samples() -> None:
    """Best-of-n sampling must not collide on one cache entry.

    Without a sample discriminator every draw of the same prompt shares a key, so a
    best-of-n search would score one cached response n times and report it as n
    independent samples -- silently turning a search into a constant.
    """
    base = RolloutRequest(model="openai/gpt-oss-120b", prompt="p", temperature=1.0)
    other = RolloutRequest(model="openai/gpt-oss-120b", prompt="p", temperature=1.0, sample_id=1)
    assert base.cache_key() != other.cache_key()


def test_cache_key_is_stable_across_processes() -> None:
    """The key must depend only on the request, never on dict ordering or hash seed."""
    a = RolloutRequest(model="m", prompt="hello नमस्ते")
    b = RolloutRequest(model="m", prompt="hello नमस्ते")
    assert a.cache_key() == b.cache_key()
    assert len(a.cache_key()) == 64


def test_reasoning_effort_is_dropped_for_models_that_reject_it() -> None:
    """Regression: llama-3.1-8b-instant 400s on every request carrying the parameter.

    `reasoning_effort` is not a universal chat-completions field. Sending it to a
    non-reasoning model produced `reasoning_effort is not supported with this model`
    and failed 100 out of 100 rollouts, which looked like a blanket API outage rather
    than one bad parameter.
    """
    from research.rollout import Client

    request = RolloutRequest(model="llama-3.1-8b-instant", prompt="p", reasoning_effort="low")
    assert Client.normalise(request).reasoning_effort == ""

    keeps = RolloutRequest(model="openai/gpt-oss-120b", prompt="p", reasoning_effort="low")
    assert Client.normalise(keeps).reasoning_effort == "low"


def test_retryable_classification() -> None:
    """A 400 must not be retried; a 429 must be.

    Retrying a permanent parameter error wastes the very rate limit the retry exists to
    survive. This split was added after 100 consecutive 400s from an unsupported
    `reasoning_effort` -- retrying those would have burned the minute budget too.
    """
    from research.rollout import _is_retryable

    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    class Weird(Exception):
        status_code = 503

    assert _is_retryable(RateLimitError("slow down"))
    assert not _is_retryable(BadRequestError("bad param"))
    assert _is_retryable(Weird("upstream"))


def test_retry_backoff_is_bounded_and_does_not_sleep_in_tests() -> None:
    """Retries stop after max_retries, and the delays grow exponentially.

    The sleep function is injected so this assertion is about the schedule rather than
    about elapsed wall-clock time. A test that actually slept would be slow and would
    fail intermittently on a loaded machine.
    """
    from research.rollout import Client

    waits: list[float] = []
    client = Client(max_retries=3, backoff_base=2.0, sleep=waits.append)

    class RateLimitError(Exception):
        pass

    calls = {"n": 0}

    class FakeCompletions:
        def create(self, **_kwargs):
            calls["n"] += 1
            raise RateLimitError("429")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    client._clients["groq"] = FakeClient()
    client._last_call["groq"] = 0.0

    from research.rollout import RolloutRequest

    result = client.run(
        RolloutRequest(model="openai/gpt-oss-120b", prompt="unique-prompt-for-retry-test")
    )
    assert result.error is not None
    assert calls["n"] == 4, "one initial attempt plus max_retries"
    assert waits == [2.0, 4.0, 8.0]


def test_failed_calls_are_not_cached() -> None:
    """A transient 429 must never be baked into the cache.

    Caching a failure would make one bad minute permanent: every later run would replay
    the error instead of retrying, and the dataset would carry a hole forever.
    """
    from research.rollout import Client, RolloutRequest, read_cache

    client = Client(max_retries=0, sleep=lambda _s: None)

    class RateLimitError(Exception):
        pass

    class FakeCompletions:
        def create(self, **_kwargs):
            raise RateLimitError("429")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    client._clients["groq"] = FakeClient()
    client._last_call["groq"] = 0.0

    request = RolloutRequest(model="openai/gpt-oss-120b", prompt="another-unique-prompt-xyz")
    assert client.run(request).error is not None
    assert read_cache(request) is None


def test_truncated_replies_are_distinguishable_from_wrong_answers() -> None:
    """A reply cut off by the token ceiling must be flagged, not silently scored wrong.

    Regression on a real contamination of a measurement. `max_tokens` is shared between
    hidden reasoning and the visible answer, and on gemini-3.6-flash 670 reasoning
    tokens left 26 for the reply, truncating it mid-field. Two of forty rollouts were
    counted as comprehension failures when they were correct answers the harness had cut
    off -- attributing a configuration error to the model.
    """
    from research.rollout import RolloutResult

    cut_off = RolloutResult(
        reply='{"name": "x", "amount_inr"',
        model="gemini-3.6-flash",
        prompt_tokens=266,
        completion_tokens=26,
        total_tokens=962,
        cached=False,
        finish_reason="length",
    )
    assert cut_off.truncated
    assert cut_off.hidden_reasoning_tokens == 670

    complete = RolloutResult(
        reply="{}",
        model="gemini-3.6-flash",
        prompt_tokens=266,
        completion_tokens=26,
        total_tokens=300,
        cached=False,
        finish_reason="stop",
    )
    assert not complete.truncated


def test_default_token_ceiling_clears_the_largest_observed_reasoning_burst() -> None:
    """The default must leave room for an answer after hidden reasoning.

    The largest reasoning burst measured on this task was 736 tokens. A ceiling of 700
    could not fit that plus any answer at all, which is exactly how the truncation above
    happened.
    """
    from research.rollout import RolloutRequest

    largest_observed_reasoning_burst = 736
    assert RolloutRequest(model="m", prompt="p").max_tokens > largest_observed_reasoning_burst
