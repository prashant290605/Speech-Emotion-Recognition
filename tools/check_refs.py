#!/usr/bin/env python
"""Reference integrity checker.

Reports only. Never edits the bibliography.

    python tools/check_refs.py                    # the current paper + supplement
    python tools/check_refs.py --offline          # cache only, no network
    python tools/check_refs.py --legacy           # the archived pre-rebuild report
    python tools/check_refs.py --tex path.tex --bib path.bib --out report.md

**The default target is the current manuscript.** It used to be
``legacy/SER_Report.tex``, which meant a bare run audited the superseded
pre-rebuild document and wrote its findings over the current report. That is
worse than no check: it reports clean on a paper nobody is submitting.

The supplement is audited alongside the article by default, because the
submission is both documents. Without it, ``lipton2018bbse`` and
``saerens2002adjusting`` -- cited only in supplementary Section S2 -- are
reported as uncited, and the obvious "fix" would be to delete two live entries.

Equivalent to ``ser check-refs``; both call into ser.refs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.refs import CURRENT_PAPER, LEGACY_PAPER, run_audit  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tex", default=None,
                        help=f"manuscript root (default: {CURRENT_PAPER['tex']})")
    parser.add_argument("--bib", default=None,
                        help=f"BibTeX database (default: {CURRENT_PAPER['bib']})")
    parser.add_argument("--out", default=None,
                        help=f"report path (default: {CURRENT_PAPER['out']})")
    parser.add_argument("--supplement", default=None,
                        help=f"additional manuscript root (default: {CURRENT_PAPER['supplement']})")
    parser.add_argument("--no-supplement", action="store_true",
                        help="audit the article alone; entries cited only in the "
                             "supplement will be reported as uncited")
    parser.add_argument("--legacy", action="store_true",
                        help="audit the archived pre-rebuild report instead")
    parser.add_argument("--cache", default=".cache/crossref.json")
    parser.add_argument("--mailto", default=None, help="Contact for Crossref's polite pool")
    parser.add_argument("--offline", action="store_true", help="Use the cache only")
    args = parser.parse_args(argv)

    if args.legacy and (args.tex or args.bib):
        parser.error("--legacy selects its own --tex/--bib; do not pass both")
    defaults = LEGACY_PAPER if args.legacy else CURRENT_PAPER

    tex = args.tex or defaults["tex"]
    bib = args.bib if args.bib is not None else defaults["bib"]
    out = args.out or defaults["out"]

    extras = []
    if not args.no_supplement:
        supplement = args.supplement or defaults["supplement"]
        if supplement and (REPO_ROOT / supplement).exists():
            extras.append(REPO_ROOT / supplement)

    print(f"Auditing {tex}" + (f" + {bib}" if bib else "")
          + (f" + {extras[0].name}" if extras else " (article only)"))

    return run_audit(
        tex_path=REPO_ROOT / tex,
        out_path=REPO_ROOT / out,
        cache_path=REPO_ROOT / args.cache,
        bib_path=(REPO_ROOT / bib) if bib else None,
        extra_tex_paths=extras,
        mailto=args.mailto,
        offline=args.offline,
    )


if __name__ == "__main__":
    raise SystemExit(main())
