"""Cached, budget-aware rollout execution against free-tier providers.

This module exists because the free tiers are small enough that a naive evaluation
loop simply cannot finish. The binding constraint is tokens per day, not requests per
day: Groq allows 200K TPD on `openai/gpt-oss-120b`, and at roughly 1.1K tokens per
rollout that is about 170 rollouts, not the 1000 the request limit implies. Three
consequences shaped the design.

**Everything is cached.** A rollout is keyed by every input that can change its
output. Re-running an evaluation after changing the verifier costs nothing, because
verification happens outside the cache -- only generation is paid for. This is what
makes it practical to iterate on scoring without re-spending quota.

**Budget is accounted in `total_tokens`, never `prompt + completion`.** Gemini's
OpenAI-compatible endpoint reports hidden reasoning tokens only in the total: a probe
returned prompt 11, completion 4, total 154. Summing the two visible fields would have
undercounted real consumption by a factor of ten and walked straight into a quota wall
with the budget tracker still reporting plenty of headroom.

**Reasoning is turned down explicitly.** Left at their defaults these models spend the
overwhelming majority of their output budget on hidden reasoning -- measured at 366 of
400 tokens for gpt-oss-120b and 736 of 792 for gemini-3.6-flash on this task. Setting
`reasoning_effort="low"` cut gpt-oss-120b to 9 reasoning tokens, a 5.5x reduction in
cost per rollout, with no loss of accuracy on a task this mechanical.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache"
ROLLOUT_DIR = CACHE_DIR / "rollouts"
BUDGET_FILE = CACHE_DIR / "budget.json"


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    daily_token_budget: int
    """Tokens per day the free tier allows for this provider's models, summed.

    Deliberately conservative: hitting the real ceiling returns 429s that are wasted
    round trips, and on some providers repeated 429s attract longer cooldowns.
    """

    requests_per_minute: int


GROQ = Provider(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    api_key_env="GROQ_API_KEY",
    # Published free-tier TPD across the models we use: gpt-oss-120b 200K,
    # llama-3.1-8b-instant 500K, llama-3.3-70b 100K. We stay under the sum.
    daily_token_budget=700_000,
    requests_per_minute=30,
)

GOOGLE = Provider(
    name="google",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key_env="GOOGLE_API_KEY",
    # Google no longer publishes a static free-tier table; limits are per-project and
    # visible only in AI Studio. This figure is a self-imposed cap, not a quoted one.
    daily_token_budget=300_000,
    # Measured, not quoted. At 10 rpm a 100-document run lost 82 rollouts to
    # RateLimitError, so the sustainable free-tier rate for gemini-3.6-flash is well
    # below that. Five leaves margin; bounded retry below absorbs the rest.
    requests_per_minute=5,
)

PROVIDERS = {p.name: p for p in (GROQ, GOOGLE)}


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    supports_reasoning_effort: bool
    """Whether the model accepts a `reasoning_effort` parameter at all.

    Not a universal OpenAI-API field. Sending it to a non-reasoning model is a hard
    400, not a warning: llama-3.1-8b-instant answers `reasoning_effort is not supported
    with this model` and fails every single request. Declaring the capability per model
    keeps that from looking like a mysterious blanket API failure.
    """


# Explicit rather than inferred from the model name, so a typo fails loudly instead of
# silently routing to the wrong endpoint or dropping a parameter.
MODELS: dict[str, ModelSpec] = {
    "openai/gpt-oss-120b": ModelSpec("groq", True),
    "openai/gpt-oss-20b": ModelSpec("groq", True),
    "llama-3.3-70b-versatile": ModelSpec("groq", False),
    "llama-3.1-8b-instant": ModelSpec("groq", False),
    "gemini-3.6-flash": ModelSpec("google", True),
    "gemini-2.5-flash-lite": ModelSpec("google", True),
}


def load_env(path: Path | None = None) -> None:
    """Read .env into os.environ without overwriting anything already set.

    Deliberately minimal and dependency-free. Keys live in exactly one place -- the
    gitignored .env -- and are never written to the cache, to logs, or to any artifact
    this repository produces.
    """
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class BudgetExhausted(RuntimeError):
    """Raised when a provider's self-imposed daily allowance is spent."""


# Error classes worth retrying. Matched on class name rather than by importing the
# provider SDK's exception types, so this module keeps working if the SDK reorganises
# them and so a second provider's differently-named 429 is still caught.
RETRYABLE_ERROR_NAMES = frozenset(
    {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "APIStatusError",
    }
)


def _is_retryable(exc: Exception) -> bool:
    """Whether a failed call is worth repeating.

    A 400 for an unsupported parameter is permanent and retrying it just burns the
    rate limit; a 429 or a dropped connection is transient and is the normal condition
    on a free tier rather than an exceptional one.
    """
    if type(exc).__name__ in RETRYABLE_ERROR_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    return status in (408, 409, 429, 500, 502, 503, 504)


@dataclass
class Budget:
    """Per-provider, per-UTC-day token accounting persisted across runs.

    Reading the clock is correct here and forbidden in the corpus generator, and the
    distinction matters: the corpus must be reproducible forever, whereas a quota is
    inherently a fact about a particular day.
    """

    path: Path = BUDGET_FILE
    _state: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path.exists():
            self._state = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def spent(self, provider: str) -> int:
        return self._state.get(self._today(), {}).get(provider, 0)

    def remaining(self, provider: str) -> int:
        return max(0, PROVIDERS[provider].daily_token_budget - self.spent(provider))

    def _reload(self) -> None:
        if self.path.exists():
            # A torn read can only happen if a writer was interrupted mid-file. Keeping
            # the in-memory state is safer than zeroing the ledger.
            with contextlib.suppress(json.JSONDecodeError):
                self._state = json.loads(self.path.read_text(encoding="utf-8"))

    def charge(self, provider: str, tokens: int) -> None:
        """Add `tokens` to today's total for `provider`, re-reading first.

        The re-read is load-bearing and was added after a real failure. Two evaluation
        processes were running concurrently, one per provider. Each had loaded the whole
        ledger at startup and wrote it back wholesale on every charge, so the Groq
        process kept restoring its stale copy of the Google total and erasing the other
        process's spend. The tracker reported plenty of Google headroom while the real
        quota was being consumed -- exactly the failure the budget exists to prevent.

        Re-reading immediately before the merge, and replacing the file atomically,
        makes concurrent single-writer-per-key updates safe. This is not a general
        cross-process lock and does not need to be: each process only ever increments
        its own provider's counter.
        """
        self._reload()
        day = self._today()
        self._state.setdefault(day, {}).setdefault(provider, 0)
        self._state[day][provider] += tokens
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file and replace, so a reader never observes a
        # half-written ledger.
        temp = self.path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.replace(temp, self.path)

    def spent_today(self) -> dict[str, int]:
        self._reload()
        return dict(self._state.get(self._today(), {}))

    def require(self, provider: str, estimate: int) -> None:
        if self.remaining(provider) < estimate:
            raise BudgetExhausted(
                f"{provider}: {self.spent(provider)} tokens spent today, "
                f"budget {PROVIDERS[provider].daily_token_budget}. Resume tomorrow; "
                f"cached rollouts are preserved."
            )


@dataclass(frozen=True)
class RolloutRequest:
    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 1400
    """Shared between hidden reasoning and the visible answer, which is the trap.

    Measured on gemini-3.6-flash: 670 reasoning tokens left 26 for the reply and cut it
    off mid-field. A ceiling that looks generous for a four-field JSON object is not,
    because the invisible portion is charged against the same budget. 1400 clears the
    largest reasoning burst observed (736) with room for the answer."""
    reasoning_effort: str = "low"
    seed: int | None = None
    """Passed to providers that honour it. Does not make sampling deterministic --
    see `sample_id` for how repeated samples are distinguished instead."""

    sample_id: int = 0
    """Distinguishes repeated samples of the same prompt at temperature > 0.

    Without it every sample in a best-of-n search would collide on the same cache key
    and the search would examine one response n times.
    """

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": self.prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
                "seed": self.seed,
                "sample_id": self.sample_id,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class RolloutResult:
    reply: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached: bool
    error: str | None = None
    finish_reason: str | None = None
    """Why generation stopped. "length" means the reply was cut off mid-answer.

    Recorded because a truncated response is an infrastructure failure, not a
    comprehension failure, and scoring the two identically attributes my own
    configuration error to the model."""

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def hidden_reasoning_tokens(self) -> int:
        """Tokens billed but not visible in prompt or completion counts.

        Non-zero for Gemini via the OpenAI-compatible layer, which reports thinking
        only inside the total. Worth surfacing because it is the single largest driver
        of free-tier consumption on this task.
        """
        return max(0, self.total_tokens - self.prompt_tokens - self.completion_tokens)


def _cache_path(key: str) -> Path:
    return ROLLOUT_DIR / key[:2] / f"{key}.json"


def read_cache(request: RolloutRequest) -> RolloutResult | None:
    path = _cache_path(request.cache_key())
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RolloutResult(
        reply=raw["reply"],
        model=raw["model"],
        prompt_tokens=raw["prompt_tokens"],
        completion_tokens=raw["completion_tokens"],
        total_tokens=raw["total_tokens"],
        cached=True,
        error=raw.get("error"),
        finish_reason=raw.get("finish_reason"),
    )


def write_cache(request: RolloutRequest, result: RolloutResult) -> None:
    path = _cache_path(request.cache_key())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "reply": result.reply,
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "error": result.error,
                "finish_reason": result.finish_reason,
                # The prompt is stored so a cache entry is self-describing and a stale
                # key can be diagnosed. No credential is ever written here.
                "prompt": request.prompt,
                "temperature": request.temperature,
                "sample_id": request.sample_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class Client:
    """Thin wrapper over the OpenAI SDK pointed at a free-tier, OpenAI-compatible host.

    The SDK is used rather than raw urllib for one non-obvious reason: Cloudflare in
    front of api.groq.com rejects Python's default urllib user-agent with a 403
    (error 1010), which looks exactly like an authentication failure. The SDK sends a
    proper user-agent and the same key succeeds.
    """

    def __init__(
        self,
        budget: Budget | None = None,
        *,
        max_retries: int = 4,
        backoff_base: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        load_env()
        self.budget = budget or Budget()
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # Injected so tests can exercise the retry path without wall-clock delays.
        # A test that really slept would be slow and, worse, timing-dependent.
        self._sleep = sleep
        self._clients: dict[str, Any] = {}
        self._last_call: dict[str, float] = {}

    def _client(self, provider: Provider):
        from openai import OpenAI

        if provider.name not in self._clients:
            key = os.environ.get(provider.api_key_env)
            if not key:
                raise RuntimeError(
                    f"{provider.api_key_env} is not set. Put it in the gitignored .env."
                )
            self._clients[provider.name] = OpenAI(api_key=key, base_url=provider.base_url)
        return self._clients[provider.name]

    def _throttle(self, provider: Provider) -> None:
        """Space requests to stay inside the provider's requests-per-minute ceiling.

        A plain sleep is adequate here: this is a single-process batch job, and the
        alternative -- discovering the limit through 429s -- burns quota on failures.
        """
        min_gap = 60.0 / provider.requests_per_minute
        last = self._last_call.get(provider.name)
        if last is not None:
            elapsed = time.monotonic() - last
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)
        self._last_call[provider.name] = time.monotonic()

    @staticmethod
    def normalise(request: RolloutRequest) -> RolloutRequest:
        """Drop parameters the target model cannot accept.

        Applied before the cache key is computed, so that a model which ignores
        reasoning effort does not get two cache entries for what is the same call.
        """
        spec = MODELS.get(request.model)
        if spec is not None and not spec.supports_reasoning_effort and request.reasoning_effort:
            return replace(request, reasoning_effort="")
        return request

    def run(self, request: RolloutRequest, *, allow_network: bool = True) -> RolloutResult:
        request = self.normalise(request)
        cached = read_cache(request)
        if cached is not None:
            return cached
        if not allow_network:
            raise RuntimeError(
                f"cache miss for {request.model} and network is disabled "
                f"(key {request.cache_key()[:12]})"
            )

        spec = MODELS.get(request.model)
        if spec is None:
            raise ValueError(f"unknown model {request.model!r}; add it to MODELS")
        provider_name = spec.provider
        provider = PROVIDERS[provider_name]

        # Charge against a conservative estimate before spending, so a run stops at the
        # budget rather than one request past it.
        estimate = len(request.prompt) // 2 + request.max_tokens
        self.budget.require(provider_name, estimate)
        self._throttle(provider)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.reasoning_effort:
            kwargs["reasoning_effort"] = request.reasoning_effort
        if request.seed is not None:
            kwargs["seed"] = request.seed

        response = None
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client(provider).chat.completions.create(**kwargs)
                break
            except Exception as exc:  # provider errors are varied and opaque
                last_error = exc
                if attempt == self.max_retries or not _is_retryable(exc):
                    break
                # Exponential backoff. Free tiers meter per minute, so the waits are
                # deliberately long enough to cross a window boundary rather than
                # hammering into the same exhausted bucket.
                self._sleep(self.backoff_base * (2**attempt))

        if response is None:
            # A failed call is recorded, not cached: transient 429s and 5xx must not be
            # baked in permanently, or a bad minute would poison the dataset forever.
            return RolloutResult(
                reply="",
                model=request.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cached=False,
                error=f"{type(last_error).__name__}: {last_error}"[:400],
            )

        usage = response.usage
        result = RolloutResult(
            reply=response.choices[0].message.content or "",
            model=response.model or request.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            cached=False,
            finish_reason=getattr(response.choices[0], "finish_reason", None),
        )
        self.budget.charge(provider_name, result.total_tokens)
        write_cache(request, result)
        return result
