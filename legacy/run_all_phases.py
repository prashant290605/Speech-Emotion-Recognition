from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shutil

from progress_utils import log_step
from results_utils import save_result


PHASE_RUNNERS = [
    {
        "phase": "phase0_baseline",
        "module": "phase0_baseline",
        "result_file": "phase0_baseline.json",
    },
    {
        "phase": "phase1_baseline",
        "module": "phase1_baseline",
        "result_file": "phase1_baseline.json",
    },
    {
        "phase": "phase2_alpha_sweep_forward",
        "module": "phase2_alpha_sweep_forward",
        "result_file": "phase2_alpha_sweep_forward.json",
    },
    {
        "phase": "phase2_alpha_sweep_reversed",
        "module": "phase2_alpha_sweep_reversed",
        "result_file": "phase2_alpha_sweep_reversed.json",
    },
    {
        "phase": "phase3_mlp_baseline",
        "module": "phase3_mlp_baseline",
        "result_file": "phase3_mlp_baseline.json",
    },
    {
        "phase": "phase4_mlp_method",
        "module": "phase4_mlp_method",
        "result_file": "phase4_mlp_method.json",
    },
    {
        "phase": "phase5_fwaa_vector_alpha",
        "module": "phase5_fwaa_vector_alpha",
        "result_file": "phase5_fwaa_vector_alpha.json",
    },
    {
        "phase": "phase6_gaa_groupwise_alpha",
        "module": "phase6_gaa_groupwise_alpha",
        "result_file": "phase6_gaa_groupwise_alpha.json",
    },
    {
        "phase": "phase7_mlp_gaa",
        "module": "phase7_mlp_gaa",
        "result_file": "phase7_mlp_gaa.json",
    },
    {
        "phase": "phase8_gaa_reverse",
        "module": "phase8_gaa_reverse",
        "result_file": "phase8_gaa_reverse.json",
    },
    {
        "phase": "phase9_gaa_in_domain",
        "module": "phase9_gaa_in_domain",
        "result_file": "phase9_gaa_in_domain.json",
    },
    {
        "phase": "phase10_svm_baseline",
        "module": "phase10_svm_baseline",
        "result_file": "phase10_svm_baseline.json",
    },
    {
        "phase": "phase11_svm_method",
        "module": "phase11_svm_method",
        "result_file": "phase11_svm_method.json",
    },
    {
        "phase": "phase12_svm_gaa",
        "module": "phase12_svm_gaa",
        "result_file": "phase12_svm_gaa.json",
    },
    {
        "phase": "phase13_svm_gaa_reverse",
        "module": "phase13_svm_gaa_reverse",
        "result_file": "phase13_svm_gaa_reverse.json",
    },
    {
        "phase": "in_domain_results",
        "module": "in_domain_results",
        "result_file": "in_domain_results.json",
    },
]


FEATURE_CACHE_DIRS = [
    Path("features/ravdess"),
    Path("features/crema_d"),
]
RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SER phases end-to-end."
    )
    parser.add_argument(
        "--preserve-cache",
        action="store_true",
        help="Keep existing wav2vec2 cache and results instead of clearing them first.",
    )
    return parser.parse_args()


def clear_generated_artifacts() -> None:
    for cache_dir in FEATURE_CACHE_DIRS:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)


def main() -> None:
    args = parse_args()

    if not args.preserve_cache:
        clear_generated_artifacts()

    log_step("===== RUNNING ALL SER PHASES =====")
    log_step("")

    executed: list[dict[str, str]] = []
    for entry in PHASE_RUNNERS:
        phase = entry["phase"]
        module_name = entry["module"]
        result_file = entry["result_file"]

        log_step(f"[START] {phase}")
        module = importlib.import_module(module_name)
        module.main()
        log_step(f"[DONE]  {phase}")
        log_step("")

        executed.append(
            {
                "phase": phase,
                "module": module_name,
                "result_file": result_file,
            }
        )

    save_result(
        "all_phases_manifest.json",
        {
            "runner": str(Path(__file__).name),
            "from_scratch": not args.preserve_cache,
            "phases": executed,
        },
    )

    log_step("===== ALL PHASES COMPLETE =====")
    log_step("Results written under: results/")


if __name__ == "__main__":
    main()
