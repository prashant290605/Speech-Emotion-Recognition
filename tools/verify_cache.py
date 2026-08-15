#!/usr/bin/env python
"""Feature cache verification (Phase 3).

Asserts row count, finiteness, shapes, and that utterance ordering matches the
manifest exactly.

    python tools/verify_cache.py
    python tools/verify_cache.py --corpora ravdess --backbones hubert

Equivalent to `ser verify-cache`; both call into ser.features.verify.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ser.cli import main as cli_main  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", default=None)
    parser.add_argument("--backbones", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    forwarded = ["verify-cache"]
    if args.config:
        forwarded = ["-c", args.config, "verify-cache"]
    if args.corpora:
        forwarded += ["--corpora", args.corpora]
    if args.backbones:
        forwarded += ["--backbones", args.backbones]
    return cli_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
