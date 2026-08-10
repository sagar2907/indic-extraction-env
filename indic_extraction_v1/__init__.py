"""Structured extraction from Indic-script documents with a deterministic verifier.

The v1 loader discovers the taskset class through `__all__`, so the export below is
the package's entire public contract as far as verifiers is concerned.

That export is resolved lazily (PEP 562) rather than imported at module scope, and the
reason is practical: `verifiers.v1` imports `fcntl` and therefore cannot be installed
on Windows at all. Importing the taskset eagerly here would make the corpus generator,
the normalisers, the verifier and the shortcut baselines -- none of which depend on
verifiers -- unimportable on a platform where they otherwise work perfectly. Anyone
can develop and test the deterministic core anywhere; only running rollouts needs a
POSIX host.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indic_extraction_v1.taskset import IndicExtractionTaskset

__all__ = ["IndicExtractionTaskset"]


def __getattr__(name: str) -> Any:
    if name == "IndicExtractionTaskset":
        from indic_extraction_v1.taskset import IndicExtractionTaskset

        return IndicExtractionTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
