"""`ser` command-line entrypoint.

One command per phase. Commands belonging to phases that have not been built
yet exist as stubs that exit with the phase number that owns them -- so the
shape of the pipeline is visible from `ser --help` on day one, and running
something out of order fails loudly instead of half-working.

On Windows, where `make` is usually absent, this CLI is the canonical
interface; the Makefile is a thin wrapper that delegates to it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

from .config import ConfigError, load_config, repo_root
from .utils.results import (
    SCHEMA_VERSION,
    SMOKE_CORPUS,
    append_row,
    count_rows,
    make_run_id,
    new_row,
    read_rows,
    schema_as_markdown,
    validate_row,
)
from .utils.runmeta import capture_runmeta
from .utils.seeding import set_all_seeds

# Phase that owns each not-yet-built command. `ser <cmd>` exits 2 with a
# pointer rather than a traceback or, worse, a partial result.
PENDING = {
    "check-refs": (1, "tools/check_refs.py: Crossref verification of the .bib"),
    "manifest": (2, "walk raw corpora -> data/manifest.csv"),
    "splits": (2, "speaker-disjoint splits, deterministic given a seed"),
    "dataset-stats": (2, "reports/dataset_stats.{md,csv}"),
    "extract": (3, "all-layer SSL + MFCC feature caches"),
    "verify-cache": (3, "tools/verify_cache.py: shape, NaN, ordering assertions"),
    "baselines": (4, "chance, majority, and prior-matched floors"),
    "align-check": (5, "end-to-end alignment sanity run on one corpus pair"),
    "classify-check": (6, "equal-budget classifier search on one corpus pair"),
    "run-grid": (7, "the full resumable grid"),
    "select": (8, "validated vs oracle selection protocols + headline tables"),
    "label-shift": (9, "prior-shift analysis and EM correction"),
    "figures": (10, "regenerate every figure from results"),
    "tables": (11, "emit LaTeX tables from results/runs.jsonl"),
    "verify": (11, "consolidated leakage and reproducibility assertions"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command in PENDING:
        phase, description = PENDING[args.command]
        print(
            f"`ser {args.command}` is not implemented yet.\n"
            f"  Owned by Phase {phase}: {description}\n"
            f"  See PHASES.md and PROGRESS.md.",
            file=sys.stderr,
        )
        return 2

    handler = {
        "inventory": _cmd_inventory,
        "schema": _cmd_schema,
        "smoke": _cmd_smoke,
    }[args.command]

    try:
        return handler(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ser",
        description="Cross-corpus speech emotion recognition: reproducibility rebuild.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to a YAML config (default: configs/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("inventory", help="[0] report legacy assets, config state, environment")
    sub.add_parser("schema", help="[0] print the frozen result schema")

    smoke = sub.add_parser("smoke", help="[0] write one synthetic row and validate it")
    smoke.add_argument(
        "--out",
        default=None,
        help="Override the results path (default: project.results_path)",
    )

    for name, (phase, description) in PENDING.items():
        sub.add_parser(name, help=f"[{phase}] {description}")

    return parser


def _epilog() -> str:
    return (
        "Phase order is enforced by PHASES.md, not by this tool. Commands for\n"
        "unbuilt phases exit 2 with the phase that owns them."
    )


# --------------------------------------------------------------------------
# Phase 0 commands
# --------------------------------------------------------------------------
def _cmd_inventory(args: argparse.Namespace) -> int:
    """Report what exists: legacy assets, caches, results, open decisions."""
    root = repo_root()
    config = load_config(args.config)
    meta = capture_runmeta(config.config_hash)

    print("=" * 72)
    print("INVENTORY")
    print("=" * 72)

    print(f"\nrepo root       : {root}")
    print(f"config          : {config.source_path}")
    print(f"config_hash     : {config.config_hash}")
    print(f"label_map_hash  : {config.label_map_hash}")
    print(f"split_spec_hash : {config.split_spec_hash}")
    print(f"git             : {meta.git_sha[:12]} ({meta.git_branch})"
          f"{' [DIRTY]' if meta.git_dirty else ''}")
    print(f"python          : {meta.python_version} on {platform.system()}")

    print("\n-- legacy assets " + "-" * 55)
    legacy = root / "legacy"
    if not legacy.exists():
        print("  (none)")
    else:
        files = sorted(p for p in legacy.rglob("*") if p.is_file())
        total = sum(p.stat().st_size for p in files)
        print(f"  {len(files)} files, {total / 1024:.0f} KiB, preserved untouched")
        for path in files:
            print(f"    {path.relative_to(root).as_posix()}  ({path.stat().st_size:,} B)")

    print("\n-- data " + "-" * 63)
    for label, key in (
        ("RAVDESS", "raw_ravdess"),
        ("CREMA-D", "raw_cremad"),
        ("IEMOCAP", "raw_iemocap"),
    ):
        path = config.resolve(getattr(config.paths, key))
        print(f"  {label:<8} {'present' if path.exists() else 'ABSENT '}  {path}")

    cache = config.resolve(config.paths.cache_dir)
    n_cached = len(list(cache.rglob("*"))) if cache.exists() else 0
    print(f"  cache    {n_cached} entries  {cache}")

    manifest = config.resolve(config.paths.manifest)
    print(f"  manifest {'present' if manifest.exists() else 'ABSENT '}  {manifest}")

    print("\n-- results " + "-" * 60)
    results = config.results_path
    print(f"  {count_rows(results)} rows  {results}")

    print("\n-- library versions " + "-" * 51)
    for name, version in sorted(meta.lib_versions.items()):
        print(f"  {name:<14} {version}")

    undecided = config.undecided()
    print("\n-- open decisions " + "-" * 53)
    if not undecided:
        print("  none: every label decision is set")
    else:
        for name in undecided:
            print(f"  UNDECIDED  labels.{name}")
        print(
            "\n  These are paper-level decisions. Phase 2 must surface them and a\n"
            "  human must set them in the config before the manifest is built."
        )

    print()
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    print(schema_as_markdown())
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Write one synthetic row end to end: config -> seed -> runmeta -> JSONL.

    This is the Phase 0 acceptance test. It exercises the real writer against
    the real results path, so a schema or provenance regression shows up here
    rather than 900 runs into the grid.
    """
    started = time.perf_counter()

    config = load_config(args.config)
    seed = set_all_seeds(config.seed)
    meta = capture_runmeta(config.config_hash)

    out_path = Path(args.out) if args.out else config.results_path
    class_names = list(config.labels.spaces["six"])

    coords = {
        "config_hash": config.config_hash,
        "label_map_hash": config.label_map_hash,
        "split_spec_hash": config.split_spec_hash,
        "seed": seed,
        "source_corpus": SMOKE_CORPUS,
        "target_corpus": SMOKE_CORPUS,
        "backbone": "none",
        "layer_agg": "n/a",
        "layer_index": None,
        "feature_branch": "ssl",
        "alignment": "none",
        "blending": "none",
        "blend_alpha": None,
        "n_groups": None,
        "classifier": "baseline_smoke",
        "split_id": "smoke-0",
    }

    # coords and the provenance stamp both carry config_hash; merge rather than
    # splat so the duplicate key resolves instead of raising.
    row = new_row(
        **{**coords, **meta.as_row_fields()},
        run_id=make_run_id(coords),
        n_classes=len(class_names),
        class_names=class_names,
        hyperparams_json=json.dumps({}, sort_keys=True),
        n_train=0,
        n_val=0,
        n_target_adapt=0,
        n_target_test=0,
        macro_f1=0.0,
        accuracy=0.0,
        uar=0.0,
        per_class_f1_json=json.dumps({name: 0.0 for name in class_names}, sort_keys=True),
        confusion_json=json.dumps([[0] * len(class_names) for _ in class_names]),
        chance_macro_f1=None,
        majority_macro_f1=None,
        prior_matched_macro_f1=None,
        selection_source_val_macro_f1=None,
        wall_seconds=round(time.perf_counter() - started, 6),
        status="ok",
        error=None,
    )

    append_row(out_path, row)

    # Read it back and revalidate: proves the round trip, not just the write.
    written = list(read_rows(out_path))[-1]
    validate_row(written)
    if written["run_id"] != row["run_id"]:
        raise AssertionError("round-trip mismatch on run_id")

    print(f"schema v{SCHEMA_VERSION}: OK  ({len(row)} fields)")
    print(f"seed          : {seed}")
    print(f"config_hash   : {config.config_hash[:16]}...")
    print(f"git_sha       : {meta.git_sha[:12]}{' [DIRTY]' if meta.git_dirty else ''}")
    print(f"run_id        : {row['run_id']}")
    print(f"appended to   : {out_path}  ({count_rows(out_path)} rows total)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
