"""The label-harmonisation table generator must reproduce the committed table.

Two generators in this repository have shipped the same latent defect: string
surgery written for a single-column float kept appending a second
``\\end{table*}`` after the shared helper began emitting ``table*`` itself. The
committed tables predated the change, so the existing checks, which read the
committed files, could not see it. This test runs the generator instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

SUMMARY_SPEC = REPO_ROOT / "configs" / "label_harmonisation_summary.yaml"


@pytest.fixture(scope="module")
def rendered():
    if not SUMMARY_SPEC.exists():
        pytest.skip(f"{SUMMARY_SPEC} not present")
    from make_label_harmonisation_table import render

    spec = yaml.safe_load(SUMMARY_SPEC.read_text(encoding="utf-8"))
    return spec, render(spec)


def test_generator_emits_exactly_one_float(rendered):
    _, text = rendered
    assert text.count("\\begin{table*}") == 1
    assert text.count("\\end{table*}") == 1
    assert text.rstrip().endswith("\\end{table*}")


def test_generator_reproduces_the_committed_table(rendered):
    spec, text = rendered
    committed = REPO_ROOT / spec["table"]
    assert text == committed.read_text(encoding="utf-8")
