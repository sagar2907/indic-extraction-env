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


def take(taskset, count: int) -> list:
    """Materialise `count` tasks using whichever lazy-selection API this version has.

    `verifiers` renamed this between releases: 0.2.x exposes `Taskset.select(n)`
    returning a list, and 0.3.0 replaced it with `Taskset.head(n)`, which returns a
    *view* -- a shallow copy carrying an `itertools.islice` transform. The environment
    supports both, because what either API actually needs is `load()` to be a generator.

    The 0.3.0 branch iterates the view rather than calling `.load()` on it, and the
    distinction is the whole point. In 0.3.0 `__iter__` is the read path and applies the
    view's transform; `load()` is the subclass hook, the raw generator we implement.
    Calling `.load()` on a view therefore bypasses the islice and yields the entire
    taskset: `head(5).load()` returned 500 documents, not 5. Only the laziness test
    caught it, because it is the one that asks for far fewer tasks than the taskset
    holds -- a test using `num_tasks == count` passes either way and proves nothing.

    This helper exists because a fresh clone caught the drift after 0.3.0 shipped:
    `pyproject.toml` asks for `verifiers>=0.2.1`, so a new install resolved to 0.3.0 and
    four tests failed on a method that no longer existed, while the development
    environment had been sitting on 0.2.1 for days and passing. The package itself was
    unaffected; only the tests named the removed method.
    """
    if hasattr(taskset, "head"):
        return list(taskset.head(count))
    return list(taskset.select(count))


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


def test_lazy_selection_api_is_available_under_some_name() -> None:
    """Whatever the version calls it, cheap selection must exist.

    If a future release removes both spellings, this fails loudly here rather than
    surfacing as a consumer silently generating the entire taskset.
    """
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=4, seed=1))
    assert hasattr(taskset, "head") or hasattr(taskset, "select")


def test_selection_only_generates_what_it_takes(counted_generate) -> None:
    """Taking k tasks must cost k documents, not `num_tasks` documents.

    Regression on a real inefficiency. `load()` returned a list, so every consumer paid
    to build the whole taskset regardless of how much they asked for -- including anyone
    running the twenty examples this package declares in `[tool.verifiers.eval]`, who was
    silently generating five hundred documents to use twenty.
    """
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=500, seed=1))
    counted_generate["n"] = 0
    tasks = take(taskset, 5)
    assert len(tasks) == 5
    assert counted_generate["n"] == 5, "selection pulled more documents than it needed"


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
    first = [t.data.model_dump() for t in take(taskset, 8)]
    # Same tasks via the plain generator, to show selection is not what determines them.
    second = [
        t.data.model_dump()
        for t in IndicExtractionTaskset(IndicExtractionConfig(num_tasks=8, seed=1)).load()
    ]
    assert first == second
    assert [t["idx"] for t in first] == list(range(8))


async def test_validate_accepts_every_row() -> None:
    """The model-free hook must accept the verifier's own reference answer."""
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=25, seed=3))
    for task in take(taskset, 25):
        assert await task.validate(None), task.data.idx


def test_rewards_and_metrics_are_registered() -> None:
    taskset = IndicExtractionTaskset(IndicExtractionConfig(num_tasks=1, seed=1))
    task = take(taskset, 1)[0]
    for name in ("field_accuracy", "format_violation", "verbosity"):
        assert hasattr(task, name), name
    for name in ("exact_match", "schema_clean", "day_month_transposed", "output_tokens"):
        assert hasattr(task, name), name
