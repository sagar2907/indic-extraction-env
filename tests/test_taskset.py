"""The verifiers v1 taskset itself.

Skipped wherever `verifiers` cannot be installed. That is not a hypothetical:
`verifiers.v1` imports `fcntl` and fails on Windows outright, so this file is the one
part of the suite that does not run everywhere. Everything it does not cover -- the
corpus, normalisation, verification, the reward -- is deliberately free of that
dependency and is tested on every platform.
"""

from __future__ import annotations

import pytest

vf = pytest.importorskip("verifiers.v1", reason="verifiers requires a POSIX host")

from pydantic import ValidationError  # noqa: E402

from indic_extraction_v1 import taskset as taskset_module  # noqa: E402
from indic_extraction_v1.taskset import (  # noqa: E402
    IndicExtractionConfig,
    IndicExtractionTaskset,
)


@pytest.fixture
def counted_generate(monkeypatch):
    """Count how many documents the generator is asked to build."""
    calls = {"n": 0}
    real = taskset_module.generate

    def counting(idx, **kwargs):
        calls["n"] += 1
        return real(idx, **kwargs)

    monkeypatch.setattr(taskset_module, "generate", counting)
    return calls


def test_select_only_generates_what_it_takes(counted_generate) -> None:
    """`select(k)` must cost k documents, not `num_tasks` documents.

    Regression on a real inefficiency. `load()` returned a list, so every consumer paid
    to build the whole taskset regardless of how much they asked for -- including anyone
    running the twenty examples this package declares in `[tool.verifiers.eval]`, who was
    silently generating five hundred documents to use twenty.
    """
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=500, seed=1))
    counted_generate["n"] = 0
    tasks = taskset.select(5)
    assert len(tasks) == 5
    assert counted_generate["n"] == 5, "select() pulled more documents than it needed"


def test_full_load_still_yields_everything(counted_generate) -> None:
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=40, seed=1))
    counted_generate["n"] = 0
    assert len(list(taskset.load())) == 40
    assert counted_generate["n"] == 40


def test_bad_tier_fails_at_load_not_on_first_pull() -> None:
    """A bad config must fail at load(), not partway through a run.

    `tier` is a plain `str | None`, so nothing rejects it before our own check. Making
    `load` a bare generator function would defer that check along with the rest of the
    body and turn a typo into a failure surfacing only once rollouts had begun -- which
    is why validation happens before the generator is returned.
    """
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=5, tier="impossible"))
    with pytest.raises(ValueError, match="tier must be one of"):
        taskset.load()


def test_bad_language_is_rejected_at_config_construction() -> None:
    """`lang` is a pydantic Literal, so it never reaches our own check.

    Worth pinning the layer that actually enforces it: the config cannot be built at all
    with an unknown language, which is stricter than validating inside load(). The check
    in `load` remains as defence for a config assembled through `model_construct`, which
    bypasses validation.
    """
    with pytest.raises(ValidationError, match="lang"):
        IndicExtractionConfig(num_tasks=5, lang="xx")  # type: ignore[arg-type]


def test_lazy_load_preserves_document_identity() -> None:
    """Yielding must not change which documents a seed produces.

    Generation is index-independent by construction, so this is expected -- but it is the
    property that makes laziness safe, and it is worth pinning rather than assuming.
    """
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=8, seed=1))
    first = [t.data.model_dump() for t in taskset.select(8)]
    second = [t.data.model_dump() for t in taskset.select(8)]
    assert first == second
    assert [t["idx"] for t in first] == list(range(8))


async def test_validate_accepts_every_row() -> None:
    """The model-free hook must accept the verifier's own reference answer."""
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=25, seed=3))
    for task in taskset.select(25):
        assert await task.validate(None), task.data.idx


def test_rewards_and_metrics_are_registered() -> None:
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=1, seed=1))
    task = taskset.select(1)[0]
    for name in ("field_accuracy", "format_violation", "verbosity"):
        assert hasattr(task, name), name
    for name in ("exact_match", "schema_clean", "day_month_transposed", "output_tokens"):
        assert hasattr(task, name), name
