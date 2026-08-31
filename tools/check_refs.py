#!/usr/bin/env python
"""Reference integrity checker (Phase 1).

Reports only. Never edits the bibliography.

    python tools/check_refs.py
    python tools/check_refs.py --offline          # cache only, no network
    python tools/check_refs.py --tex path.tex --bib path.bib --out reports/refs_report.md

Equivalent to `ser check-refs`; both call into ser.refs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.refs import run_audit  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default="legacy/SER_Report.tex")
    parser.add_argument("--bib", default=None,
                        help="Optional BibTeX database; citations are read from --tex and its inputs")
    parser.add_argument("--out", default="reports/refs_report.md")
    parser.add_argument("--cache", default=".cache/crossref.json")
    parser.add_argument("--mailto", default=None, help="Contact for Crossref's polite pool")
    parser.add_argument("--offline", action="store_true", help="Use the cache only")
    args = parser.parse_args(argv)

    return run_audit(
        tex_path=REPO_ROOT / args.tex,
        out_path=REPO_ROOT / args.out,
        cache_path=REPO_ROOT / args.cache,
        bib_path=(REPO_ROOT / args.bib) if args.bib else None,
        mailto=args.mailto,
        offline=args.offline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
