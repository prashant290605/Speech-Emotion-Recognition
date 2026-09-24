"""The --skip-reports staleness gate.

``make_results_doc.py`` runs dozens of 2000-replicate cluster bootstraps and
takes tens of minutes, which makes manuscript iteration painful enough that
somebody will eventually want to skip it. The danger is obvious: a paper built
from reports that describe a previous state.

So the skip does not trust the files. Every report must exist and be newer than
the ledger, the derived artifacts and its own generator. These tests pin the
gate itself -- that it accepts current reports, refuses stale ones, refuses
missing ones, and names what went wrong -- without running a build.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_paper  # noqa: E402


@pytest.fixture()
def fake_repo(tmp_path, monkeypatch):
    """A miniature repository whose mtimes we can control."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "src" / "ser").mkdir(parents=True)

    old = time.time() - 1000
    for rel in build_paper.REPORT_INPUTS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input", encoding="utf-8")
        import os
        os.utime(path, (old, old))
    for rel, generator in build_paper.REPORT_ARTIFACTS:
        (tmp_path / rel).write_text("report", encoding="utf-8")
        gen = tmp_path / generator
        gen.parent.mkdir(parents=True, exist_ok=True)
        gen.write_text("generator", encoding="utf-8")
        import os
        os.utime(gen, (old, old))

    monkeypatch.setattr(build_paper, "REPO_ROOT", tmp_path)
    return tmp_path


def test_current_reports_are_accepted(fake_repo):
    build_paper.stage_reports_verify()  # must not raise


def test_a_newer_input_makes_the_reports_stale(fake_repo):
    import os

    now = time.time() + 10
    os.utime(fake_repo / "results" / "runs.jsonl", (now, now))
    with pytest.raises(build_paper.BuildError) as excinfo:
        build_paper.stage_reports_verify()
    message = str(excinfo.value)
    assert "stale" in message
    assert "results/runs.jsonl" in message, "the gate must name the newer input"


def test_a_newer_generator_makes_its_own_report_stale(fake_repo):
    """Editing the generator invalidates its output even if no data moved."""
    import os

    now = time.time() + 10
    os.utime(fake_repo / "tools" / "make_results_doc.py", (now, now))
    with pytest.raises(build_paper.BuildError) as excinfo:
        build_paper.stage_reports_verify()
    message = str(excinfo.value)
    assert "reports/RESULTS.md" in message
    assert "tools/make_results_doc.py" in message


def test_a_missing_report_is_refused(fake_repo):
    (fake_repo / "reports" / "RESULTS.md").unlink()
    with pytest.raises(build_paper.BuildError, match="requires existing reports"):
        build_paper.stage_reports_verify()


def test_the_error_tells_the_user_what_to_do(fake_repo):
    import os

    now = time.time() + 10
    os.utime(fake_repo / "results" / "runs.jsonl", (now, now))
    with pytest.raises(build_paper.BuildError) as excinfo:
        build_paper.stage_reports_verify()
    assert "without --skip-reports" in str(excinfo.value)


# -- the flag cannot become the default -----------------------------------
def test_skip_reports_is_off_by_default():
    parser_args = build_paper.main.__doc__ or ""
    source = (REPO_ROOT / "tools" / "build_paper.py").read_text(encoding="utf-8")
    assert '"--skip-reports", action="store_true"' in source
    assert 'default=True' not in source.split("--skip-reports")[1][:400]


def test_check_only_does_not_accept_stale_reports_silently():
    """--check-only generates nothing, so it must not claim reports are current."""
    source = (REPO_ROOT / "tools" / "build_paper.py").read_text(encoding="utf-8")
    check_only = source[source.index("if args.check_only:"):]
    assert "stage_reports" not in check_only.split("return")[0]


def test_every_declared_report_input_exists_in_the_real_repository():
    """A typo in REPORT_INPUTS would silently weaken the gate."""
    for rel in build_paper.REPORT_INPUTS:
        assert (REPO_ROOT / rel).exists(), rel


def test_every_declared_report_and_generator_exists():
    for rel, generator in build_paper.REPORT_ARTIFACTS:
        assert (REPO_ROOT / rel).exists(), rel
        assert (REPO_ROOT / generator).exists(), generator
