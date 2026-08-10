"""What ships in the published wheel, and what must not.

A hub user installs this environment to get a taskset. They should not also receive my
evaluation harness, my adversarial policies, or my test suite -- that is dependency and
attack surface they did not ask for, and it would make the package's zero-dependency
claim false the moment the research code grew an import.

Reading `pyproject.toml` rather than building a wheel keeps this fast enough to belong
in the normal suite. The full check -- build, install into an empty environment, import
with no other packages present -- is documented in the README and was run against the
built artefact; this guards the configuration that makes it true.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_only_the_environment_package_is_published() -> None:
    """research/, tests/ and scripts/ must never enter the wheel."""
    packages = _config()["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert packages == ["indic_extraction_v1"], packages


def test_published_package_declares_no_runtime_dependencies() -> None:
    """The zero-dependency promise, enforced.

    The corpus is generated rather than downloaded and every normaliser is
    standard-library only, so there is nothing legitimate to add here. `verifiers` in
    particular must stay out: it is supplied by whatever host loads the taskset, and
    pinning it here would force a version on every consumer.
    """
    assert _config()["project"]["dependencies"] == []


def test_dev_extra_carries_the_tooling() -> None:
    """Everything the research code needs lives behind the dev extra, not the default."""
    dev = _config()["project"]["optional-dependencies"]["dev"]
    joined = " ".join(dev)
    for required in ("verifiers", "openai", "pytest", "ruff"):
        assert required in joined, required


def test_python_floor_matches_what_verifiers_requires() -> None:
    """verifiers requires >=3.11,<3.14; publishing a wider range would be a lie.

    A consumer on 3.14 who installs this package succeeds, then fails to load the
    taskset -- a confusing failure one layer away from its cause.
    """
    assert _config()["project"]["requires-python"] == ">=3.11,<3.14"


def test_every_published_module_avoids_importing_verifiers_at_module_scope() -> None:
    """Only taskset.py may import verifiers, and the package must not do it eagerly.

    `verifiers.v1` imports fcntl and cannot be installed on Windows. Importing it at
    package scope would make the corpus generator, normalisers, verifier and reward --
    none of which need it -- unimportable on a platform where they otherwise work. The
    export in `__init__.py` is resolved lazily (PEP 562) for exactly this reason.
    """
    package_dir = PYPROJECT.parent / "indic_extraction_v1"
    for path in sorted(package_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        imports_verifiers = "import verifiers" in source
        if path.name == "taskset.py":
            assert imports_verifiers, "taskset.py is the one module that should"
            continue
        assert not imports_verifiers, f"{path.name} imports verifiers at module scope"

    init = (package_dir / "__init__.py").read_text(encoding="utf-8")
    assert "def __getattr__" in init, "the taskset export must stay lazy"


def test_pyproject_declares_hub_tags() -> None:
    """The Hub's integration test asserts on `tags`, which is not a PEP 621 field.

    Regression on a real CI failure. The first push carried only the standard
    `keywords` field. The upload succeeded, so nothing looked wrong locally, and the
    Hub then ran its own test suite against the published artefact and failed with
    "pyproject.toml does not have tags" -- arriving as an email rather than as anything
    the local build could have caught.

    Worth knowing that a build backend accepting the file is not the same as the Hub
    accepting it: `tags` is an extension the standard tooling ignores entirely.
    """
    project = _config()["project"]
    tags = project.get("tags")
    assert isinstance(tags, list) and tags, "the Hub requires a non-empty project.tags"
    assert all(isinstance(t, str) and t for t in tags)
    # The languages are the discoverability point; the Hub has near-zero Indic coverage.
    for expected in ("indic", "hindi", "tamil", "bengali"):
        assert expected in tags, expected


def test_pyproject_has_every_field_the_hub_checks() -> None:
    """Mirror of the Hub's `test_pyproject_has_metadata`, run locally.

    Cheaper to fail here than to discover it from a post-push email.
    """
    project = _config()["project"]
    for required in ("name", "version", "description", "tags"):
        assert required in project, required


def test_eval_defaults_are_declared() -> None:
    """`prime env eval` should have sane defaults so a first look is cheap."""
    eval_config = _config()["tool"]["verifiers"]["eval"]
    assert eval_config["num_examples"] > 0
    assert eval_config["rollouts_per_example"] > 0
