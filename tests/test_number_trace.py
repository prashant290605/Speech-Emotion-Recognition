from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_number_trace import NUMBER, is_traced, text_without_noncontent  # noqa: E402


def test_latex_scientific_notation_is_traced_as_one_number():
    text = text_without_noncontent(r"$1.15\times10^{-8}$")
    assert NUMBER.findall(text) == ["1.15e-8"]


def test_scientific_notation_accepts_display_rounding():
    from decimal import Decimal

    assert is_traced("1.15e-8", {Decimal("1.1514e-8")})
