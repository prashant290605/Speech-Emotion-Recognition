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
        "check-refs": _cmd_check_refs,
        "manifest": _cmd_manifest,
        "dataset-stats": _cmd_dataset_stats,
        "splits": _cmd_splits,
        "extract": _cmd_extract,
        "verify-cache": _cmd_verify_cache,
        "baselines": _cmd_baselines,
        "align-check": _cmd_align_check,
        "effective-rank": _cmd_effective_rank,
        "ladder-table": _cmd_ladder_table,
        "classify-check": _cmd_classify_check,
        "run-grid": _cmd_run_grid,
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

    refs = sub.add_parser(
        "check-refs", help="[1] audit the bibliography against Crossref (reports only)"
    )
    refs.add_argument("--tex", default="legacy/SER_Report.tex")
    refs.add_argument("--out", default="reports/refs_report.md")
    refs.add_argument("--cache", default=".cache/crossref.json")
    refs.add_argument("--mailto", default=None, help="Contact for Crossref's polite pool")
    refs.add_argument("--offline", action="store_true", help="Use the cache only")

    manifest = sub.add_parser(
        "manifest", help="[2] walk the raw corpora -> data/manifest.csv"
    )
    manifest.add_argument(
        "--corpora",
        default=None,
        help="Comma-separated subset to build (default: grid.corpora)",
    )
    manifest.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Record a corpus whose size disagrees with the published expectation "
        "instead of halting. Use only when you know why it differs.",
    )

    stats = sub.add_parser(
        "dataset-stats", help="[2] per-class counts, priors and prior KL from the manifest"
    )
    stats.add_argument("--corpora", default=None, help="Comma-separated subset")

    splits = sub.add_parser(
        "splits", help="[2] build every split, run the leakage assertions, report priors"
    )
    splits.add_argument("--corpora", default=None, help="Comma-separated subset")

    extract = sub.add_parser("extract", help="[3] build the all-layer feature caches")
    extract.add_argument("--corpora", default=None, help="Comma-separated subset")
    extract.add_argument(
        "--backbones",
        default=None,
        help="Comma-separated subset of features.backbones plus 'mfcc' "
        "(default: all of them)",
    )
    extract.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch CPU threads. Lower it when running one process per backbone.",
    )
    extract.add_argument(
        "--plan", action="store_true", help="Show the work plan and exit"
    )

    verify = sub.add_parser("verify-cache", help="[3] shape, finiteness, ordering assertions")
    verify.add_argument("--corpora", default=None, help="Comma-separated subset")
    verify.add_argument("--backbones", default=None, help="Comma-separated subset")

    baselines = sub.add_parser(
        "baselines", help="[4] chance floors per pair and seed -> results/runs.jsonl"
    )
    baselines.add_argument("--corpora", default=None, help="Comma-separated subset")
    baselines.add_argument(
        "--force", action="store_true", help="Recompute rows that already exist"
    )

    align = sub.add_parser(
        "align-check", help="[5] fit every ladder rung on one pair and report shift"
    )
    align.add_argument("--source", default="ravdess")
    align.add_argument("--target", default="cremad")
    align.add_argument("--seed", type=int, default=0)
    align.add_argument("--backbone", default="hubert")
    align.add_argument("--layer-spec", default="layer:6")
    clf = sub.add_parser(
        "classify-check", help="[6] every family at equal budget, selection on source_val"
    )
    clf.add_argument("--source", default="ravdess")
    clf.add_argument("--target", default="cremad")
    clf.add_argument("--seed", type=int, default=0)
    clf.add_argument("--backbone", default="hubert")
    clf.add_argument("--layer", type=int, default=6)
    clf.add_argument("--families", default=None, help="Comma-separated subset")

    grid = sub.add_parser("run-grid", help="[7] the staged, resumable grid")
    grid.add_argument("--stage", type=int, required=True, choices=[0, 1, 2])
    grid.add_argument("--corpora", default=None, help="Comma-separated subset")
    grid.add_argument("--dry-run", action="store_true", help="Enumerate and exit")
    grid.add_argument(
        "--no-freeze",
        action="store_true",
        help="Allow an unfrozen config. For development only; the grid proper "
        "must run against a tagged config.",
    )

    ladder = sub.add_parser(
        "ladder-table", help="[5] effect size per rung at 5 bandwidths + invariants"
    )
    ladder.add_argument("--source", default="ravdess")
    ladder.add_argument("--target", default="cremad")
    ladder.add_argument("--seed", type=int, default=0)
    ladder.add_argument("--backbone", default="hubert")
    ladder.add_argument("--layer-spec", default="layer:6")

    rank = sub.add_parser(
        "effective-rank", help="[5] effective rank per backbone per layer"
    )
    rank.add_argument("--corpora", default=None, help="Comma-separated subset")
    rank.add_argument("--seed", type=int, default=0)

    align.add_argument(
        "--lambdas",
        default=None,
        help="Comma-separated subset of the MMD lambda grid, to keep a check quick",
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
        "label_map_hash": config.label_map_hash,
        "split_spec_hash": config.split_spec_hash,
        "feature_spec_hash": config.feature_spec_hash,
        "search_spec_hash": config.search_spec_hash,
        "seed": seed,
        "source_corpus": SMOKE_CORPUS,
        "target_corpus": SMOKE_CORPUS,
        "backbone": "none",
        "layer_agg": "n/a",
        "layer_index": None,
        "feature_branch": "ssl",
        "alignment": "none",
        "alignment_eps": None,
        "alignment_lambda": None,
        "blending": "none",
        "blend_alpha": None,
        "n_groups": None,
        "classifier": "baseline_smoke",
        "split_id": "smoke-0",
    }

    # coords and the provenance stamp both carry config_hash; merge rather than
    # splat so the duplicate key resolves instead of raising.
    row = new_row(
        # coords carries the run_id facets; meta.as_row_fields() supplies
        # config_hash, which is recorded but is no longer a coordinate.
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
        cov_condition_number=None,
        cov_effective_rank=None,
        n_search_trials=None,
        marginal_mmd_raw=None,
        marginal_mmd_normalised=None,
        freeze_tag=None,
        per_class_precision_json=None,
        per_class_recall_json=None,
        per_class_support_json=None,
        n_collapsed_classes=None,
        epochs_run=None,
        predictions_path=None,
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


def _cmd_check_refs(args: argparse.Namespace) -> int:
    """Phase 1: audit the bibliography. Reports only, never edits the .tex."""
    from .refs import run_audit  # noqa: PLC0415 - keeps urllib off the import path

    root = repo_root()
    return run_audit(
        tex_path=root / args.tex,
        out_path=root / args.out,
        cache_path=root / args.cache,
        mailto=args.mailto,
        offline=args.offline,
    )


_CORPUS_PATH_KEYS = {
    "ravdess": "raw_ravdess",
    "cremad": "raw_cremad",
    "iemocap": "raw_iemocap",
}


def _selected_corpora(config, override: str | None) -> list[str]:
    if override:
        selected = [name.strip() for name in override.split(",") if name.strip()]
    else:
        selected = list(config.grid.corpora)
    unknown = [name for name in selected if name not in _CORPUS_PATH_KEYS]
    if unknown:
        raise ConfigError(f"unknown corpus/corpora: {unknown}")
    return selected


def _cmd_manifest(args: argparse.Namespace) -> int:
    """Phase 2: walk the raw corpora into data/manifest.csv."""
    from .labels import LabelPolicy  # noqa: PLC0415
    from .manifest import CountMismatch, build_manifest, write_manifest  # noqa: PLC0415

    config = load_config(args.config)
    seed = set_all_seeds(config.seed)
    policy = LabelPolicy.from_config(config)

    corpora = _selected_corpora(config, args.corpora)
    roots = {
        name: config.resolve(getattr(config.paths, _CORPUS_PATH_KEYS[name]))
        for name in corpora
    }

    print(f"seed {seed} | label_map_hash {config.label_map_hash}")
    for name, root in roots.items():
        print(f"  {name:<8} {root}")
    print()

    def progress(corpus: str, count: int) -> None:
        print(f"  {corpus}: {count} files...", flush=True)

    try:
        rows = build_manifest(
            roots, policy, verify_counts=not args.allow_count_mismatch, progress=progress
        )
    except CountMismatch as exc:
        print(f"\nHALTED: {exc}", file=sys.stderr)
        return 1

    out = config.resolve(config.paths.manifest)
    written = write_manifest(rows, out)
    print(f"\nwrote {written} rows to {out}")

    for name in corpora:
        subset = [r for r in rows if r.corpus == name]
        speakers = {r.speaker_id for r in subset}
        hours = sum(r.duration_s for r in subset) / 3600.0
        print(
            f"  {name:<8} {len(subset):>5} utterances | {len(speakers):>3} speakers "
            f"| {hours:5.2f} h"
        )
    return 0


def _cmd_splits(args: argparse.Namespace) -> int:
    """Phase 2: build every split and run the leakage assertions."""
    from .splitreport import run_splits_report  # noqa: PLC0415

    config = load_config(args.config)
    set_all_seeds(config.seed)
    return run_splits_report(config, _selected_corpora(config, args.corpora))


def _selected_backbones(config, override: str | None) -> list[str]:
    from .features.extract import MFCC_BACKBONE  # noqa: PLC0415

    available = list(config.features.backbones) + [MFCC_BACKBONE]
    if override:
        selected = [name.strip() for name in override.split(",") if name.strip()]
    else:
        selected = available
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ConfigError(f"unknown backbone(s) {unknown}; available: {available}")
    return selected


def _cmd_extract(args: argparse.Namespace) -> int:
    """Phase 3: build the feature caches. A key hit is a no-op."""
    from .features.audio import warm_up_audio_stack  # noqa: PLC0415

    # MUST precede anything that imports torch. See warm_up_audio_stack.
    warm_up_audio_stack()

    from .features.extract import plan_extraction, extract_one  # noqa: PLC0415
    from .manifest import read_manifest  # noqa: PLC0415

    config = load_config(args.config)
    set_all_seeds(config.seed)

    if args.threads:
        import torch  # noqa: PLC0415

        torch.set_num_threads(args.threads)

    rows = read_manifest(config.resolve(config.paths.manifest))
    corpora = _selected_corpora(config, args.corpora)
    backbones = _selected_backbones(config, args.backbones)

    plan = plan_extraction(rows, config, corpora, backbones)
    todo = [unit for unit in plan if not unit["exists"]]

    print(f"{len(plan)} work unit(s); {len(todo)} to extract, {len(plan)-len(todo)} cached")
    for unit in plan:
        state = "cached" if unit["exists"] else "TO EXTRACT"
        print(f"  {unit['corpus']:<8} {unit['backbone']:<10} {unit['n_rows']:>5} utts  {state}")
    if args.plan:
        return 0
    print()

    def progress(corpus, backbone, done, total, elapsed):
        rate = done / elapsed if elapsed else 0
        remaining = (total - done) / rate if rate else 0
        print(
            f"  [{corpus}/{backbone}] {done}/{total} "
            f"({done/total:5.1%}) {rate:5.2f} utt/s  eta {remaining/60:5.1f} min",
            flush=True,
        )

    total_wall = 0.0
    for unit in plan:
        result = extract_one(
            rows, config, unit["corpus"], unit["backbone"], progress=progress
        )
        total_wall += result["wall_seconds"]
        verb = "cached  " if result["status"] == "cached" else "extracted"
        print(
            f"{verb} {result['corpus']:<8} {result['backbone']:<10} "
            f"n={result['n']:<5} {result['wall_seconds']/60:6.1f} min  {result['path'].name}",
            flush=True,
        )

    print(f"\ntotal extraction wall time this invocation: {total_wall/60:.1f} min")
    return 0


def _cmd_verify_cache(args: argparse.Namespace) -> int:
    """Phase 3: assert count, finiteness, shapes, and manifest ordering."""
    from .features.verify import verify_all  # noqa: PLC0415
    from .manifest import read_manifest  # noqa: PLC0415

    config = load_config(args.config)
    rows = read_manifest(config.resolve(config.paths.manifest))
    corpora = _selected_corpora(config, args.corpora)
    backbones = _selected_backbones(config, args.backbones)

    results = verify_all(rows, config, corpora, backbones)
    failures = [r for r in results if r["problems"]]
    total_bytes = sum(r["bytes"] for r in results)

    for result in results:
        status = "OK" if not result["problems"] else "FAIL"
        print(
            f"  {result['corpus']:<8} {result['backbone']:<10} "
            f"{result['bytes']/1e6:8.1f} MB  {status}"
        )
        for problem in result["problems"]:
            print(f"      - {problem}")

    print(f"\n{len(results) - len(failures)}/{len(results)} caches verified"
          f" | {total_bytes/1e9:.2f} GB total")
    if failures:
        print("VERIFICATION FAILED", file=sys.stderr)
        return 1
    return 0


def _cmd_baselines(args: argparse.Namespace) -> int:
    """Phase 4: chance floors for every pair and seed."""
    from .baselinerun import run_baselines  # noqa: PLC0415

    config = load_config(args.config)
    set_all_seeds(config.seed)
    return run_baselines(
        config, _selected_corpora(config, args.corpora), force=args.force
    )


def _cmd_align_check(args: argparse.Namespace) -> int:
    """Phase 5: fit every ladder rung on one pair. Trains and selects nothing."""
    from .features.audio import warm_up_audio_stack  # noqa: PLC0415

    warm_up_audio_stack()

    from .alignrun import run_alignment_check  # noqa: PLC0415

    config = load_config(args.config)
    lambdas = (
        [float(v) for v in args.lambdas.split(",")] if args.lambdas else None
    )
    return run_alignment_check(
        config,
        args.source,
        args.target,
        seed=args.seed,
        backbone=args.backbone,
        layer_spec=args.layer_spec,
        lambdas=lambdas,
    )


def _cmd_run_grid(args: argparse.Namespace) -> int:
    """Phase 7: the staged grid. Refuses to start against a drifted config."""
    from .features.audio import warm_up_audio_stack  # noqa: PLC0415

    warm_up_audio_stack()

    from .freeze import ConfigDrift  # noqa: PLC0415
    from .run_grid import run_grid  # noqa: PLC0415

    config = load_config(args.config)
    try:
        return run_grid(
            config,
            args.stage,
            corpora=_selected_corpora(config, args.corpora),
            dry_run=args.dry_run,
            require_freeze=not args.no_freeze,
        )
    except ConfigDrift as exc:
        print(f"REFUSING TO RUN: {exc}", file=sys.stderr)
        return 2


def _cmd_classify_check(args: argparse.Namespace) -> int:
    """Phase 6: all families at equal budget, selection on source_val only."""
    from .classifyrun import run_classifier_check  # noqa: PLC0415

    config = load_config(args.config)
    families = [f.strip() for f in args.families.split(",")] if args.families else None
    return run_classifier_check(
        config, args.source, args.target, seed=args.seed,
        backbone=args.backbone, layer=args.layer, families=families,
    )


def _cmd_ladder_table(args: argparse.Namespace) -> int:
    """Phase 5: the paper's ladder table, with bandwidth robustness."""
    from .ladderreport import run_ladder_table  # noqa: PLC0415

    config = load_config(args.config)
    return run_ladder_table(
        config, args.source, args.target, seed=args.seed,
        backbone=args.backbone, layer_spec=args.layer_spec,
    )


def _cmd_effective_rank(args: argparse.Namespace) -> int:
    """Effective rank of source_train covariance, per backbone and layer."""
    from .rankreport import run_rank_report  # noqa: PLC0415

    config = load_config(args.config)
    return run_rank_report(
        config, _selected_corpora(config, args.corpora), seed=args.seed
    )


def _cmd_dataset_stats(args: argparse.Namespace) -> int:
    """Phase 2: per-class counts, priors, and the A8 prior-KL verification."""
    from .datastats import run_dataset_stats  # noqa: PLC0415

    config = load_config(args.config)
    return run_dataset_stats(config, _selected_corpora(config, args.corpora))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
